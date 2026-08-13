#!/usr/bin/env python3
"""
Find ByteTrack ID SWITCHES from the STORE alone -- no clips, no model, no GPU.

    # validation arm first (run whose switches the operator confirmed by eye)
    python tests/calibration/find_switches_from_store.py 20260804_094039 \
        --url http://192.168.1.35:6333

    # then the ground-truth run, naming the tracklets CLAUDE.md 3a/3b rests on
    python tests/calibration/find_switches_from_store.py 20260804_064551 \
        --url http://192.168.1.35:6333 \
        --watch cam_219:20,cam_219:8,cam_213:31,cam_213:35,cam_224:1,cam_224:30

WHY THIS EXISTS, GIVEN find_track_switches.py ALREADY DOES THIS
==============================================================
`find_track_switches.py` re-embeds crops, so it needs `._live_src_<cam>.mp4` plus a
sidecar and the ReID model. Run `20260804_064551` -- the ONLY run carrying operator
ground truth (three confirmed pairs, 2238 observations) -- has **no clips**. They
were overwritten by two later runs, `._live_src_<cam>.mp4` carries no run_id, and
CLAUDE.md 7 records that the loss is permanent. So the clip-based scanner can never
be pointed at the one run whose answers we trust.

But every one of that run's 2048-d per-observation vectors is still in Qdrant, and
the switch signature from commit 9ce94ced9e is PURE ARITHMETIC over those vectors.
Nothing here loads a model or decodes a frame, so it runs on the CPU-only dev box
and gives the same answer as the A6000 -- the same standing that
`measure_rank_and_bias.py` claims in its own docstring. CLAUDE.md's "dev box
conclusions are wrong" rule is about running the MODEL.

WHAT IT MEASURES, AND WHY THE RANDOM ARM IS THE WHOLE TRICK
===========================================================
A track's two temporal halves can disagree because the person turned, because the
light changed, or because the track covers TWO PEOPLE. Scanning every cut point and
taking the minimum finds the most-divided split; but a low minimum alone cannot
distinguish those three causes.

A RANDOM partition of the same observations, at the SAME two sizes, controls for all
three at once: it puts the same mixture on both sides, so it stays high no matter
which cause is operating. Therefore

    gap = median(random partitions) - min(temporal cuts)

isolates TEMPORAL STRUCTURE. Commit 9ce94ced9e established the reading empirically:
on run 20260804_094039 the two tracklets the operator confirmed were two people each
took the two largest gaps (0.368 for cam_219:6, 0.348 for cam_219:14), while clean
tracklets sat at 0.038-0.044.

`spread` (p90 - min of the scan curve) separates a STEP from a SLOPE. A switch is a
step: high on both sides of one frame, low across it, so the curve is bimodal and
spread is large. A person slowly turning gives a shallow bowl -- smaller spread at
the same minimum. Suggestive, not proof.

THIS IS A PROXY FOR THE CLIP-BASED SCAN, AND IT MUST BE VALIDATED AS ONE
=======================================================================
The store holds only observations that passed the crop-quality gate and the
`reid.interval` time filter, so a store tracklet is a SUBSAMPLE of the clip's crops.
Absolute gaps will therefore not match `find_track_switches.py`'s. Only the RANKING
is claimed to transfer, and even that is a claim to be tested, not assumed -- which
is what the validation arm on 20260804_094039 is for. Run it there FIRST; if the two
operator-confirmed chimeras do not surface, this proxy is invalid and nothing it says
about 064551 may be used.

EDGE CUTS: RUN IT BOTH WAYS
===========================
`--min-side-frac` exists because of commit 8c3641898b: with no floor, the scan cuts
off the first or last few crops -- entering/leaving view, clipped, motion-blurred --
and reports a false near-zero. But both operator-confirmed switches cut near an EDGE
(37/40 and 3/40), which is what a handover looks like. So a floor may blunt exactly
the real case. Run 0.25 for a credible rate and 0.05 for sensitivity; a tracklet
flagged at BOTH is the confident set.

NOT A FIX, AND DELIBERATELY NOT WIRED TO ANYTHING. It reports; you look.
"""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import arg, bootstrap, flag, header, validate_flags     # noqa: E402

bootstrap()

from database.store import PersonVectorStore                         # noqa: E402
from identity import reconcile as R                                  # noqa: E402


# The reading established by commit 9ce94ced9e on run 20260804_094039. Stated here,
# and printed before any result, because AGENT_BRIEF rule 4 requires the
# falsification criterion to be pre-registered -- it is what killed the scale/blur
# hypothesis cleanly.
GAP_SUSPECT = 0.20

# The two tracklets the operator confirmed, by eye, are each two people, and the
# gaps the CLIP-based scanner gave them. The validation arm checks this proxy
# reproduces them.
CONFIRMED_SWITCHES = {
    ("20260804_094039", "cam_219", 6): 0.368,
    ("20260804_094039", "cam_219", 14): 0.348,
}


def unit_rows(mat):
    """Row-wise L2 normalization -- `_prototype`'s first step, done once."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def scan_track(vectors, min_side_frac, trials, rng):
    """
    -> (temporal_min, cut_index, rand_median, gap, spread, n_cuts) or None.

    Identical arithmetic to `reconcile._prototype` (mean of L2-normalized vectors,
    renormalized) but via prefix sums, so all n cuts cost O(n) in total rather than
    O(n^2). Renormalizing makes the 1/k mean scaling irrelevant, so a prefix SUM and
    a prefix MEAN give the same unit vector -- the two forms are exactly equal, not
    approximately.
    """
    mat = unit_rows(np.stack(vectors).astype(np.float32))
    n = len(mat)

    lo = max(1, int(np.ceil(n * min_side_frac)))
    hi = n - lo
    if hi < lo:
        return None                      # too short to cut at this floor

    prefix = np.cumsum(mat, axis=0)       # prefix[i-1] = sum of first i rows
    total = prefix[-1]

    cuts = np.arange(lo, hi + 1)
    left = prefix[cuts - 1]               # sum of rows [0, cut)
    right = total - left                  # sum of rows [cut, n)
    ln = np.linalg.norm(left, axis=1, keepdims=True)
    rn = np.linalg.norm(right, axis=1, keepdims=True)
    scan = np.sum((left / np.clip(ln, 1e-12, None)) *
                  (right / np.clip(rn, 1e-12, None)), axis=1)

    k = int(np.argmin(scan))
    temporal_min = float(scan[k])
    cut_index = int(cuts[k])
    spread = float(np.percentile(scan, 90) - scan.min())

    # Random arm at the SAME two sizes as the winning temporal cut, so size is
    # controlled and the only difference is whether the partition respects time.
    rand = np.empty(trials, dtype=np.float64)
    for t in range(trials):
        idx = rng.permutation(n)[:cut_index]
        l = mat[idx].sum(axis=0)
        r = total - l
        rand[t] = float(unit(l) @ unit(r))
    rand_median = float(np.median(rand))

    return temporal_min, cut_index, rand_median, rand_median - temporal_min, \
        spread, len(cuts)


def parse_watch(spec):
    """'cam_219:20,cam_213:31' -> {('cam_219',20), ('cam_213',31)}"""
    out = set()
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise SystemExit(f"[switch] --watch entry {tok!r} must be CAM:TRACK, "
                             f"e.g. cam_219:20")
        cam, tid = tok.rsplit(":", 1)
        try:
            out.add((cam.strip(), int(tid)))
        except ValueError:
            raise SystemExit(f"[switch] --watch entry {tok!r} has a non-integer "
                             f"track id")
    return out


def main():
    validate_flags(extra=("--url", "--path", "--min-obs", "--min-side-frac",
                          "--trials", "--seed", "--cam", "--watch", "--all"))
    run_ids = [a for a in sys.argv[1:] if not a.startswith("--")]
    # arg() values are argv tokens too; drop anything consumed as a flag value.
    consumed = set()
    for f in ("--url", "--path", "--min-obs", "--min-side-frac", "--trials",
              "--seed", "--cam", "--watch"):
        v = arg(f)
        if v is not None:
            consumed.add(v)
    run_ids = [r for r in run_ids if r not in consumed]
    if not run_ids:
        raise SystemExit(__doc__.strip().split("\n\n")[1])

    url = arg("--url", "http://localhost:6333") or None
    min_obs = int(arg("--min-obs", 12))
    frac = float(arg("--min-side-frac", 0.25))
    trials = int(arg("--trials", 200))
    seed = int(arg("--seed", 0))
    cam_filter = arg("--cam")
    watch = parse_watch(arg("--watch"))
    show_all = flag("--all")

    if not 0.0 <= frac < 0.5:
        raise SystemExit(f"[switch] --min-side-frac must be in [0, 0.5), got {frac}")

    store = PersonVectorStore(path=arg("--path", "qdrant_data"), url=url)
    total_points = store.count()

    header("ID-SWITCH SCAN FROM THE STORE -- no clips, no model, no GPU")
    print(f"  store:          {url or arg('--path', 'qdrant_data')} "
          f"-- {total_points} point(s) total")
    print(f"  min-obs:        {min_obs}    min-side-frac: {frac}    "
          f"trials: {trials}    seed: {seed}")
    print(f"  suspect gap:    > {GAP_SUSPECT}")

    # The empty-store trap, made loud. `.env` ships QDRANT_URL=localhost:6333 and a
    # local container answers there with a 2048-d `persons` holding ZERO points,
    # while the corpus lives on another host. A silent empty read is how this
    # project got two "the store is empty" false alarms.
    if total_points == 0:
        raise SystemExit(
            "\n[switch] THIS STORE IS EMPTY (0 points) -- refusing to report.\n"
            "         A reachable-but-empty Qdrant is the standing trap here: the\n"
            "         dev box runs its own container on 6333 with the right SHAPE\n"
            "         (2048-d `persons`) and no data. Pass the host that holds the\n"
            "         corpus, e.g. --url http://192.168.1.35:6333")

    print("\n  PRE-REGISTERED READING (stated before any result):")
    print(f"    gap > {GAP_SUSPECT}          suspected chimera -- one track_id, two people")
    print("    large spread          the scan curve is a STEP  -> switch")
    print("    small spread          the curve is a shallow bowl -> a turn, not a switch")
    print("    gap ~ 0               no temporal structure -> one person")
    print("  A gap is EVIDENCE, never proof. Only the crops at `cut_frame` settle it,")
    print("  which is why the frame is printed.")

    for run_id in run_ids:
        _report_run(store, run_id, min_obs, frac, trials, seed, cam_filter,
                    watch, show_all)
    return 0


def _report_run(store, run_id, min_obs, frac, trials, seed, cam_filter, watch,
                show_all):
    tracklets = R._gather_tracklets(store, run_id)
    if not tracklets:
        raise SystemExit(
            f"\n[switch] run_id {run_id!r} has NO observations in this store.\n"
            f"         Check the id against the run log -- and check you are "
            f"pointed at the host that holds it.")

    header(f"RUN {run_id}")
    rows, skipped = [], []
    for (cam, tid), info in sorted(tracklets.items()):
        if cam_filter and cam != cam_filter:
            continue
        vectors, frames = info["vectors"], info["frames"]
        if len(vectors) < min_obs:
            skipped.append((cam, tid, len(vectors)))
            continue
        rng = np.random.default_rng(seed + abs(hash((cam, tid))) % 10_000)
        res = scan_track(vectors, frac, trials, rng)
        if res is None:
            skipped.append((cam, tid, len(vectors)))
            continue
        tmin, cut_i, rmed, gap, spread, n_cuts = res
        rows.append({
            "cam": cam, "tid": tid, "n": len(vectors),
            "tmin": tmin, "cut_i": cut_i,
            "cut_frame": frames[cut_i] if cut_i < len(frames) else frames[-1],
            "rmed": rmed, "gap": gap, "spread": spread, "n_cuts": n_cuts,
            "span": info["span"],
        })

    print(f"  {len(rows)} tracklet(s) scanned, {len(skipped)} below min-obs "
          f"or too short to cut")

    if not rows:
        print("  NOTHING SCANNABLE. Lower --min-obs, or this run's tracklets are "
              "all short.")
        return

    rows.sort(key=lambda r: -r["gap"])
    _print_table("ALL TRACKLETS, ranked by gap (largest = most switch-like)", rows)

    flagged = [r for r in rows if r["gap"] > GAP_SUSPECT]
    print(f"\n  FLAGGED: {len(flagged)} of {len(rows)} tracklet(s) exceed "
          f"gap {GAP_SUSPECT}")
    if flagged:
        print("  Look at these frames before believing anything downstream of them:")
        for r in flagged:
            print(f"    {r['cam']}:{r['tid']:<4} cut at frame {r['cut_frame']:<7} "
                  f"({r['cut_i']}/{r['n']})  gap {r['gap']:.3f}  "
                  f"spread {r['spread']:.3f}")

    # ---- validation arm ----------------------------------------------------
    known = {(c, t): g for (r, c, t), g in CONFIRMED_SWITCHES.items() if r == run_id}
    if known:
        print("\n  VALIDATION ARM -- operator-confirmed chimeras in this run.")
        print("  This proxy is only usable if these surface. Clip-based gaps shown")
        print("  for orientation only: the store is a SUBSAMPLE, so absolute values")
        print("  are not expected to match -- the RANKING is what must transfer.")
        by_key = {(r["cam"], r["tid"]): r for r in rows}
        order = {(r["cam"], r["tid"]): i + 1 for i, r in enumerate(rows)}
        ok = True
        for (cam, tid), clip_gap in sorted(known.items()):
            r = by_key.get((cam, tid))
            if r is None:
                print(f"    {cam}:{tid:<4} NOT SCANNED (below min-obs) -- "
                      f"validation INCONCLUSIVE")
                ok = False
                continue
            rank = order[(cam, tid)]
            verdict = "OK" if r["gap"] > GAP_SUSPECT else "MISSED"
            if verdict == "MISSED":
                ok = False
            print(f"    {cam}:{tid:<4} store gap {r['gap']:.3f} "
                  f"(rank {rank}/{len(rows)})   clip gap {clip_gap:.3f}   "
                  f"{verdict}")
        print(f"\n    VALIDATION: {'PASSED' if ok else 'FAILED'}")
        if not ok:
            print("    The store-based proxy did NOT reproduce a confirmed switch.")
            print("    Do not use its verdicts on any other run.")

    # ---- watchlist ---------------------------------------------------------
    if watch:
        print("\n  WATCHLIST -- tracklets named on the command line.")
        print("  These carry published numbers. A chimera here voids that number,")
        print("  because a chimeric tracklet's prototype is a mean over two people.")
        by_key = {(r["cam"], r["tid"]): r for r in rows}
        short = {(c, t): n for c, t, n in skipped}
        present = {(c, t) for (c, t) in tracklets}
        for cam, tid in sorted(watch):
            r = by_key.get((cam, tid))
            if r is not None:
                verdict = "CHIMERA SUSPECTED" if r["gap"] > GAP_SUSPECT else "clean"
                print(f"    {cam}:{tid:<4} n={r['n']:<5} gap {r['gap']:.3f}  "
                      f"spread {r['spread']:.3f}  cut@frame {r['cut_frame']:<7} "
                      f"{verdict}")
            elif (cam, tid) in short:
                print(f"    {cam}:{tid:<4} n={short[(cam, tid)]:<5} "
                      f"NOT SCANNABLE (below min-obs {min_obs}) -- "
                      f"no verdict, and none should be inferred")
            elif (cam, tid) not in present:
                print(f"    {cam}:{tid:<4} ABSENT from this run's store")
            else:
                print(f"    {cam}:{tid:<4} filtered out by --cam")

    if skipped and show_all:
        print(f"\n  below min-obs ({min_obs}):")
        for cam, tid, n in sorted(skipped):
            print(f"    {cam}:{tid:<4} n={n}")


def _print_table(title, rows):
    print(f"\n  {title}")
    print(f"    {'tracklet':<14}{'n':>5}{'temporal':>10}{'random':>9}"
          f"{'gap':>8}{'spread':>8}{'cut':>10}{'cut_frame':>11}{'span':>16}")
    print(f"    {'-' * 14}{'-' * 5:>5}{'-' * 10:>10}{'-' * 9:>9}"
          f"{'-' * 8:>8}{'-' * 8:>8}{'-' * 10:>10}{'-' * 11:>11}{'-' * 16:>16}")
    for r in rows:
        mark = " *" if r["gap"] > GAP_SUSPECT else "  "
        print(f"    {r['cam'] + ':' + str(r['tid']):<14}{r['n']:>5}"
              f"{r['tmin']:>10.3f}{r['rmed']:>9.3f}{r['gap']:>8.3f}"
              f"{r['spread']:>8.3f}{str(r['cut_i']) + '/' + str(r['n']):>10}"
              f"{r['cut_frame']:>11}"
              f"{str(r['span'][0]) + '-' + str(r['span'][1]):>14}{mark}")


if __name__ == "__main__":
    sys.exit(main())
