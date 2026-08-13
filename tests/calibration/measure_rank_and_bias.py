"""
measure_rank_and_bias.py  --  RANK-based identity measurement, with a camera-bias
control arm and a LEAVE-ONE-RUN-OUT holdout mode.

    python tests/calibration/measure_rank_and_bias.py <run_id> [<run_id> ...]
    python tests/calibration/measure_rank_and_bias.py --holdout <run_id> <run_id> ...

NEEDS NO MODEL AND NO GPU. It reads embeddings already in Qdrant and does pure
arithmetic, so it gives the same answer on the dev box and on the A6000 -- unlike
anything that re-embeds. (CLAUDE.md's "dev box conclusions are wrong" rule is about
running the MODEL.)

============================ WHY IT REPORTS RANKS =============================
`same_min - diff_max` is not decision-relevant, and this project has repeatedly been
misled by it:

  * `diff_max` is an extreme value that GROWS with sample size (CLAUDE.md 6.3:
    0.819 -> 0.936 at 48 vs 90 frames). So does any gap built on it.
  * pooling cameras mixes populations under DIFFERENT bars. Measured on run
    20260804_064551: pooled min-vs-MAX = -0.135 while cam_219 = +0.171 and
    cam_224 = +0.197 against their own p95. The "empty window" was the statistic.
  * a monotone rescaling of every score moves every margin and changes NO decision.

reconcile picks the top-scoring partner and requires it to be mutual, so this reports
Rank@1 and MRR. A transform that improves margins but leaves every rank untouched has
bought nothing.

============================ CROSS-SPACE COMPARISON ==========================
Comparing an absolute cosine in RAW against one in GLOBAL/CAMERA is the same class of
error as comparing across backbones or feature taps, which config.yaml's `reid.model`
/ `reid.tap` notes and AGENT_BRIEF rule 5 forbid. So every space is judged ONLY on
statistics internal to itself (rank, and margin against its OWN stranger reference).
Absolute scores across spaces are printed for orientation and are labelled as such.

============================ THE CONTROL ARM =================================
Per-camera mean subtraction is standard practice, not a discovery (camera-aware
normalisation; cf. Camera-based Batch Normalization, ECCV'20, and the camera-proxy
family). AGENT_BRIEF already listed it as hypothesis #3. It therefore gets tested
against the arm that must be excluded first:

    RAW      -- no transform
    GLOBAL   -- subtract ONE mean shared by all cameras
    CAMERA   -- subtract each camera's own mean

If GLOBAL matches CAMERA there is no camera story, only a generic person-crop offset.

MEANS ARE ALWAYS LEAVE-ONE-RUN-OUT. Estimating from the run being scored destroyed a
confirmed same-person pair (cam_213 0.630 -> -0.193) because with 34 vectors the
"camera mean" is mostly that person.

============================== LABELS ========================================
`calibration/tracklet_pairs.jsonl`, one JSON object per line:

    {"run_id": "...", "a": ["cam_219", 20], "b": ["cam_219", 8],  "same": true}
    {"run_id": "...", "a": ["cam_219", 20], "b": ["cam_224", 1],   "same": true}

BOTH sides carry their OWN camera. An earlier version of this file took the camera
from `a` and the track from `b`, which silently turned every cross-camera label into a
same-camera pair between unrelated tracks -- track ids are per-camera integers, so
collisions are near-certain. Cross-camera labels are the ones everything is gated on,
so they are validated here and reported in their own section.

n IS TINY. With no label file this falls back to the three operator-confirmed pairs in
CLAUDE.md 3b and prints their published scores as a self-check. Every margin is then a
one- to three-point estimate and is printed with its n. The stranger side gets a
bootstrap CI; the same-person side cannot have one at n<=3 and says so.
"""

import argparse
import collections
import itertools
import json
import os

import numpy as np

from _common import bootstrap, header

ROOT = bootstrap()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "persons")
MIN_OBS = 3                  # identity.reconcile.min_tracklet_observations
MIN_OVERLAP_SEC = 0.5        # a "provably different" pair needs REAL co-presence
P95_MIN_N = 20               # below this, a p95 IS the max -- refuse to print one
LABEL_PATH = "calibration/tracklet_pairs.jsonl"

# CLAUDE.md 3b's table. The doc writes "0020 vs 0008", which reads like reconcile's
# U-handles and is NOT -- they are TRACK IDs (run 20260804_064551 has 28 tracklets, so
# U-0031 cannot exist). Confirmed by reproducing all three published scores.
FALLBACK_PAIRS = [
    ("20260804_064551", ("cam_219", 20), ("cam_219", 8), 0.574),
    ("20260804_064551", ("cam_213", 31), ("cam_213", 35), 0.630),
    ("20260804_064551", ("cam_224", 1), ("cam_224", 30), 0.907),
]


# ---------------------------------------------------------------- data loading

def scroll(run_id):
    import requests
    vecs, meta = [], []
    offset = None
    while True:
        body = {"limit": 500, "with_vector": True,
                "with_payload": ["camera", "track_id", "frame", "ts"],
                "filter": {"must": [{"key": "run_id", "match": {"value": run_id}}]}}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                          json=body, timeout=300)
        r.raise_for_status()
        res = r.json()["result"]
        for p in res["points"]:
            pl = p["payload"]
            vecs.append(p["vector"])
            meta.append((pl.get("camera"), int(pl.get("track_id")),
                         pl.get("frame"), pl.get("ts")))
        offset = res.get("next_page_offset")
        if offset is None:
            break
    if not vecs:
        return None, []
    X = np.asarray(vecs, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    if float(np.abs(n - 1.0).max()) > 1e-3:
        print(f"  WARNING: stored vectors not unit norm (max dev "
              f"{float(np.abs(n - 1.0).max()):.3g}) -- normalising.")
    return X / np.clip(n, 1e-12, None), meta


class Run:
    """One run's tracklets, keyed by (camera, track) END TO END."""

    def __init__(self, run_id, X, meta):
        self.run_id, self.X, self.meta = run_id, X, meta
        self.cams = np.array([m[0] for m in meta])
        idx = collections.defaultdict(list)
        for i, m in enumerate(meta):
            idx[(m[0], m[1])].append(i)
        self.all_idx = dict(idx)
        self.idx = {k: v for k, v in idx.items() if len(v) >= MIN_OBS}
        self.fspan = {k: (min(meta[i][2] for i in v), max(meta[i][2] for i in v))
                      for k, v in self.all_idx.items()}
        self.tspan = {}
        for k, v in self.all_idx.items():
            t = [meta[i][3] for i in v if meta[i][3] is not None]
            if t:
                self.tspan[k] = (min(t), max(t))
        self.cameras = sorted(set(self.cams))

    def keys(self, camera=None):
        return sorted(k for k in self.idx if camera is None or k[0] == camera)


def proto(V):
    m = V.mean(0)
    n = np.linalg.norm(m)
    return None if n <= 0 else m / n


def _renorm(Y):
    return Y / np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12, None)


def frames_disjoint(a, b):
    return a[1] < b[0] or b[1] < a[0]


def overlap_sec(run, a, b):
    ta, tb = run.tspan.get(a), run.tspan.get(b)
    if ta is None or tb is None:
        return None
    return min(ta[1], tb[1]) - max(ta[0], tb[0])


# ------------------------------------------------------------------- the spaces

def build_spaces(run, other_X, other_cams):
    out = [("RAW", run.X)]
    if other_X is None or len(other_X) == 0:
        print("  NOTE: no held-out run -> GLOBAL/CAMERA SKIPPED. Self-estimated "
              "means are not a control (see docstring).")
        return out
    out.append(("GLOBAL", _renorm(run.X - other_X.mean(0))))
    mu = {}
    for c in run.cameras:
        sel = other_cams == c
        if sel.sum() >= 30:
            mu[c] = other_X[sel].mean(0)
        else:
            print(f"  NOTE: {c} has {int(sel.sum())} held-out vectors (<30) -- its "
                  f"CAMERA mean is not estimable; left uncentred.")
    out.append(("CAMERA", _renorm(np.stack(
        [run.X[i] - mu[run.cams[i]] if run.cams[i] in mu else run.X[i]
         for i in range(len(run.X))]))))
    return out


# ------------------------------------------------------------------ statistics

def quantile(v, q, min_n=P95_MIN_N):
    """None when n is too small for the quantile to differ from the max.

    int(0.95*n) >= n-1 for every n <= 20, so the previous hand-rolled version
    returned the TOP ORDER STATISTIC under a `p95` header in 2 of 5 held-out rows
    and in every cam_224 row. Refusing is the only honest option.
    """
    if not v or len(v) < min_n:
        return None
    return float(np.quantile(np.asarray(v, dtype=np.float64), q))


def boot_quantile(v, q, iters=2000, seed=0):
    """(lo, hi) 5-95% bootstrap interval for a quantile, or None when n is small."""
    if not v or len(v) < P95_MIN_N:
        return None
    rng = np.random.default_rng(seed)
    a = np.asarray(v, dtype=np.float64)
    draws = [np.quantile(rng.choice(a, size=len(a), replace=True), q)
             for _ in range(iters)]
    return float(np.quantile(draws, .05)), float(np.quantile(draws, .95))


def stranger_scores(run, P, camera, min_overlap=MIN_OVERLAP_SEC):
    """Provably-different pairs: SAME camera, co-present for > min_overlap seconds.

    Cross-camera co-presence proves nothing here (CLAUDE.md 6.1, the cameras
    overlap), so it is never used.

    THE OVERLAP FLOOR MATTERS AND WAS MISSING. `frames_disjoint` needs ONE frame of
    span overlap to declare two tracklets "provably two people", so a track that was
    dropped and re-acquired with a single double-detected frame qualified -- and that
    pair is ONE person, scoring high, sitting in the middle of the stranger
    reference. Measured on runs 064551/094039/120409 the floor removes 1-3 pairs per
    run and does NOT move any tail (0.962/0.843/0.710 all survive it), so it changes
    no published number -- but the hole was real and a future run can fall in it.

    STILL CIRCULAR WITH THE CHIMERA HYPOTHESIS: if one side is chimeric (two people
    under one track_id) it can share a person with its "stranger" partner. Nothing
    here can detect that; only labels can.
    """
    out, dropped = [], 0
    for a, b in itertools.combinations(run.keys(camera), 2):
        if frames_disjoint(run.fspan[a], run.fspan[b]):
            continue
        ov = overlap_sec(run, a, b)
        if ov is not None and ov <= min_overlap:
            dropped += 1
            continue
        out.append(float(P[a] @ P[b]))
    return sorted(out), dropped


def candidate_pool(run, query, partner_camera):
    """Who `query` competes against.

    SAME camera  -> time-DISJOINT tracklets in that camera (reconcile's Phase 1 pool;
                    a time-overlapping peer is provably another person and is
                    excluded before scoring).
    CROSS camera -> every tracklet in the partner camera. Cross-camera co-presence
                    proves nothing here, so nothing is excluded for it.

    The previous version filtered every pool to `k[0] == query[0]`, so the script
    could not do cross-camera retrieval at all while its own metric contract promised
    per-camera-pair numbers.
    """
    if partner_camera == query[0]:
        return [k for k in run.keys(query[0]) if k != query
                and frames_disjoint(run.fspan[k], run.fspan[query])]
    return [k for k in run.keys(partner_camera) if k != query]


def rank_of(run, P, query, truth):
    pool = candidate_pool(run, query, truth[0])
    order = [k for _, k in sorted(((float(P[query] @ P[k]), k) for k in pool),
                                  reverse=True)]
    if truth not in order:
        return None, len(pool)
    return order.index(truth) + 1, len(pool)


# ------------------------------------------------------------------- reporting

def load_labels(run_id):
    """[(a, b, published_or_None)] for this run. Validates both sides exist."""
    if not os.path.exists(LABEL_PATH):
        return [(a, b, pub) for r, a, b, pub in FALLBACK_PAIRS if r == run_id], True
    out = []
    with open(LABEL_PATH) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("run_id") != run_id or not rec.get("same"):
                continue
            try:
                a = (str(rec["a"][0]), int(rec["a"][1]))
                b = (str(rec["b"][0]), int(rec["b"][1]))
            except (KeyError, IndexError, TypeError, ValueError):
                print(f"  {LABEL_PATH}:{ln}: malformed a/b -- skipped ({rec!r})")
                continue
            out.append((a, b, None))
    return out, False


def report_run(run, others):
    oX = np.concatenate([o.X for o in others]) if others else None
    oc = np.concatenate([o.cams for o in others]) if others else None
    header(f"RUN {run.run_id}   {len(run.X)} obs   "
           f"{len(run.idx)} tracklets with >= {MIN_OBS} obs   cameras {run.cameras}")
    SPACES = build_spaces(run, oX, oc)
    P = {n: {k: proto(Z[v]) for k, v in run.idx.items()} for n, Z in SPACES}

    labels, is_fallback = load_labels(run.run_id)
    kept = []
    for a, b, pub in labels:
        missing = [x for x in (a, b) if x not in run.idx]
        if missing:
            print(f"  label {a} <-> {b}: {missing} absent or under {MIN_OBS} obs "
                  f"-> skipped")
            continue
        kept.append((a, b, pub))
    same_cam = [(a, b, p) for a, b, p in kept if a[0] == b[0]]
    cross_cam = [(a, b, p) for a, b, p in kept if a[0] != b[0]]
    print(f"  labelled same-person pairs: {len(same_cam)} same-camera, "
          f"{len(cross_cam)} CROSS-camera"
          + ("   [fallback: CLAUDE.md 3b]" if is_fallback else ""))
    if not cross_cam:
        print("  NO cross-camera same-person label -> every cross-camera number in "
              "this project is unmeasurable. That is the binding gate, not a caveat.")

    if any(p is not None for _, _, p in kept):
        print("\n  SELF-CHECK vs published scores (RAW):")
        for a, b, pub in kept:
            if pub is None:
                continue
            got = float(P["RAW"][a] @ P["RAW"][b])
            print(f"    {a[0]}:{a[1]} + {b[0]}:{b[1]}  published {pub}  "
                  f"measured {got:.3f}  {'ok' if abs(got - pub) < 0.005 else 'MISMATCH'}")

    # ---- edge lists
    if kept:
        header("EDGE LISTS -- is there a CHAIN to close, or is the tracklet isolated?")
        for a, b, _ in kept:
            for q, truth in ((a, b), (b, a)):
                pool = candidate_pool(run, q, truth[0])
                sc = sorted(((float(P["RAW"][q] @ P["RAW"][k]), k) for k in pool),
                            reverse=True)
                lane = "same-camera" if truth[0] == q[0] else f"-> {truth[0]}"
                print(f"\n  [RAW] {q[0]}:{q[1]}  ({lane}, {len(pool)} candidates)")
                for s, k in sc[:6]:
                    star = "  <-- TRUE PARTNER" if k == truth else ""
                    print(f"      {k[0]}:{k[1]:<6}{s:>8.3f}   {run.fspan[k]}{star}")

    # ---- the table
    header("PER-CAMERA TABLE -- ranks, and margins against each space's OWN reference")
    print(f"  strangers: same camera, co-present > {MIN_OVERLAP_SEC}s. "
          f"p95 refused below n={P95_MIN_N} (it would BE the max).")
    print(f"  {'space':<8}{'camera':<9}{'n_str':>6}{'p95':>8}{'p95 CI':>17}{'MAX':>8}"
          f"{'n_same':>7}{'same scores':>22}{'R@1':>7}{'MRR':>7}{'n_q':>5}")
    for n, _Z in SPACES:
        Pz = P[n]
        for c in run.cameras:
            strangers, dropped = stranger_scores(run, Pz, c)
            p95 = quantile(strangers, .95)
            ci = boot_quantile(strangers, .95)
            mine = [(a, b) for a, b, _ in kept if a[0] == c or b[0] == c]
            sames = [float(Pz[a] @ Pz[b]) for a, b in mine]
            rr = []
            for a, b in mine:
                for q, truth in ((a, b), (b, a)):
                    rk, _ = rank_of(run, Pz, q, truth)
                    if rk:
                        rr.append(rk)
            print(f"  {n:<8}{c:<9}{len(strangers):>6}"
                  + (f"{p95:>8.3f}" if p95 is not None else f"{'n<20':>8}")
                  + (f"{f'[{ci[0]:+.3f},{ci[1]:+.3f}]':>17}" if ci else f"{'--':>17}")
                  + (f"{strangers[-1]:>8.3f}" if strangers else f"{'--':>8}")
                  + f"{len(mine):>7}"
                  + f"{('  '.join(f'{s:+.3f}' for s in sames) or '--'):>22}"
                  + (f"{sum(1 for r in rr if r == 1) / len(rr):>7.2f}"
                     f"{sum(1 / r for r in rr) / len(rr):>7.2f}{len(rr):>5}"
                     if rr else f"{'--':>7}{'--':>7}{0:>5}"))
    print("\n  Same-person scores are listed individually, not reduced to a min: with")
    print("  n<=3 a min is one point. There is no CI for that side and none is faked.")
    print("  Absolute scores are NOT comparable BETWEEN spaces (see docstring).")

    # ---- bar reachability
    header("BAR REACHABILITY -- can a quorum/rounds sweep show anything?")
    print("  reconcile.py:1398 `if s < cam_bar(cam): continue` runs immediately before")
    print("  1401 `if not all_member_pairs_clear(...)`, so quorum is UNREACHABLE when")
    print("  nothing clears the bar. And 1405 requires MUTUAL BEST on top, so a joint")
    print("  sweep is bar x quorum x rounds x same_camera_reciprocal_best.")
    print("  NOTE `same_camera_member_quorum` is NOT in config.yaml -- it is a code")
    print("  default (reconcile.py:269). Add the key before sweeping it.\n")
    bars = load_bars()
    for c in run.cameras:
        bar = bars.get(c, bars["_global"])
        ks = run.keys(c)
        pr = sorted((float(P["RAW"][a] @ P["RAW"][b])
                     for a, b in itertools.combinations(ks, 2)
                     if frames_disjoint(run.fspan[a], run.fspan[b])), reverse=True)
        if not pr:
            continue
        cl = sum(1 for v in pr if v >= bar)
        src = "per_camera" if c in bars else "GLOBAL (no override)"
        print(f"  {c}  bar={bar:.2f} [{src}]  eligible={len(pr)}  clearing={cl} "
              f"({cl / len(pr):.0%})  best={pr[0]:.3f}")


def report_holdout(runs):
    """The multi-run table that was previously a scratchpad script.

    Every cell is (derived on: every OTHER run / tested on: this run).
    """
    header("HELD-OUT: means derived on the OTHER runs, tested on run Z")
    print(f"  Provably-different pairs only (same camera, co-present > "
          f"{MIN_OVERLAP_SEC}s).")
    print("  This side has usable n. The same-person side has labels in at most one")
    print("  run, so it CANNOT be held out -- stated, not hidden.")
    print(f"\n  {'tested on':<17}{'cam':<9}{'n':>4}{'dropped':>8}"
          f"{'RAW p95/MAX':>18}{'GLOBAL p95/MAX':>18}{'CAMERA p95/MAX':>18}")
    agg = collections.defaultdict(list)
    for run in runs:
        others = [o for o in runs if o.run_id != run.run_id]
        oX = np.concatenate([o.X for o in others])
        oc = np.concatenate([o.cams for o in others])
        SPACES = build_spaces(run, oX, oc)
        for c in run.cameras:
            cells, drop = [], None
            enough = True
            for n, Z in SPACES:
                Pz = {k: proto(Z[v]) for k, v in run.idx.items()}
                s, drop = stranger_scores(run, Pz, c)
                if len(s) < 3:
                    enough = False
                    break
                p95 = quantile(s, .95)
                cells.append(f"{('n<20' if p95 is None else f'{p95:+.3f}')}"
                             f"/{s[-1]:+.3f}")
                agg[n].append((p95, s[-1], len(s)))
            if not enough:
                print(f"  {run.run_id:<17}{c:<9}{'--':>4}{'--':>8}   "
                      f"fewer than 3 provably-different pairs -> UNMEASURABLE")
                continue
            print(f"  {run.run_id:<17}{c:<9}{len(s):>4}{drop:>8}"
                  + "".join(f"{x:>18}" for x in cells))
    print("\n  pooled over held-out (run, camera) cells:")
    for n in ("RAW", "GLOBAL", "CAMERA"):
        v = agg.get(n) or []
        if not v:
            continue
        p95s = [x[0] for x in v if x[0] is not None]
        print(f"    {n:<8} cells={len(v)}  mean MAX {np.mean([x[1] for x in v]):+.3f}"
              f"  worst MAX {max(x[1] for x in v):+.3f}"
              + (f"  mean p95 {np.mean(p95s):+.3f} (over {len(p95s)} cells with n>=20)"
                 if p95s else "  no cell reached n>=20, so NO p95 is reportable"))
    print("\n  READ THIS BEFORE CONCLUDING ANYTHING. Comparing a MAX in RAW against a")
    print("  MAX in CAMERA is a cross-space cosine comparison and licenses nothing.")
    print("  Each space may only be judged against its OWN same-person reference, and")
    print("  that reference exists in ONE run. So this table can show that a transform")
    print("  compresses a stranger distribution; it CANNOT show it helps or hurts.")


def load_bars():
    try:
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        rc = (cfg.get("identity") or {}).get("reconcile") or {}
        bars = {"_global": float(rc.get("same_camera_threshold", 0.90))}
        for cam, ov in (rc.get("per_camera") or {}).items():
            if isinstance(ov, dict) and "same_camera_threshold" in ov:
                bars[str(cam)] = float(ov["same_camera_threshold"])
        return bars
    except Exception as e:                                        # noqa: BLE001
        print(f"  (could not read config.yaml bars: {e}; assuming 0.90)")
        return {"_global": 0.90}


def main():
    global MIN_OVERLAP_SEC
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--holdout", action="store_true",
                    help="multi-run leave-one-run-out stranger table only")
    ap.add_argument("--min-overlap", type=float, default=MIN_OVERLAP_SEC)
    args = ap.parse_args()
    MIN_OVERLAP_SEC = args.min_overlap

    print(f"[calib] measure_rank_and_bias  qdrant={QDRANT_URL} "
          f"collection={COLLECTION} min_obs={MIN_OBS} "
          f"min_overlap={MIN_OVERLAP_SEC}s p95_min_n={P95_MIN_N}")
    runs = []
    for r in args.runs:
        X, meta = scroll(r)
        if X is None:
            print(f"[calib] run {r}: no points -- skipped")
            continue
        run = Run(r, X, meta)
        runs.append(run)
        print(f"[calib] run {r}: {len(X)} obs, "
              + ", ".join(f"{c}={int((run.cams == c).sum())}" for c in run.cameras))
    if not runs:
        print("[calib] nothing to measure.")
        return 1
    if len(runs) < 2:
        print("[calib] WARNING: one run only -> no held-out mean is possible.")

    if args.holdout:
        report_holdout(runs)
        return 0
    for run in runs:
        report_run(run, [o for o in runs if o.run_id != run.run_id])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
