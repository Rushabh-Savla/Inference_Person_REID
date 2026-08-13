"""
Why did these two identities NOT merge?

    python tests/calibration/explain_merge_failure.py <run_id> <id_a> <id_b>
    python tests/calibration/explain_merge_failure.py <run_id> --tracklets cam_219:7,cam_219:46
    python tests/calibration/explain_merge_failure.py <run_id> 1 11 --no-covisibility

WHY THIS EXISTS (REMEDIATION_PLAN.md Part M). The operator reports one person
carrying two reids; reconcile is supposed to merge them and does not. Four rounds
of threshold tuning were reverted because nobody could answer the only question
that matters: WHICH RULE refused, on WHICH pair of tracklets, by HOW MUCH.

The decision log cannot answer it. `conflict()` runs BEFORE scoring in
`mergeable_cross`, so a cluster pair blocked by a physical veto is dropped from
`root_scores` entirely -- no score, no candidate record, no gate. It leaves no
trace at all. And `conflict_reason` returns only the gate NAME, discarding the
member pair that triggered it.

So this script reconstructs the refusal from outside, using reconcile's own
primitives at the PRODUCTION settings, and prints:

  * both clusters, with observation counts, frame spans and wall-clock spans
  * the bar that applies to the pair, and WHY it is that bar (this is usually the
    surprise: two clusters that share a camera are judged at the SAME-camera bar,
    the strictest one over every shared camera -- not the cross-camera threshold)
  * the pair score under all three scoring modes
  * the FULL conflict grid -- every member pair, its gate, and its overlap in
    seconds -- so the blocking witness is named
  * for each blocking same-camera pair, whether the overlap is REAL (sampled
    observations actually interleave) or a PHANTOM of the min/max envelope (a gap
    in one tracklet swallowing the other). That distinction decides the fix:
    phantom -> M.2 (occupancy, not envelope); real -> M.4 (a wrong member
    poisoned the cluster).

READ-ONLY. Like the sweep, it runs reconcile through a store proxy that discards
every write, so diagnosing a run can never rewrite that run's ids.
"""

import sys
from collections import defaultdict

import numpy as np

from _common import arg, bootstrap, header, load_config, reconcile_settings

bootstrap()

from database.store import PersonVectorStore                        # noqa: E402
from identity import reconcile as R                                 # noqa: E402
from identity.reconcile import describe_reconcile_kwargs            # noqa: E402


class _ReadOnlyStore:
    """Real observations in, no writes out (same guard as the sweep)."""

    def __init__(self, store):
        self._store = store
        self.client = store.client
        self.collection = store.collection

    def set_global_id(self, point_ids, global_id):
        pass

    def clear_global_id(self, point_ids):
        pass


def _fmt_key(k):
    return f"{k[0]}:{k[1]:04d}"


def _fmt_span(span):
    return "-" if not span else f"{span[0]:.1f}..{span[1]:.1f}"


def _sampled_cooccurrences(a_times, b_times, tol):
    """How many of A's sampled instants land within `tol` seconds of one of B's.

    This is the question the same-camera veto MEANS to ask ("were these two boxes
    on screen at the same moment") as opposed to the one it actually asks ("do
    their min/max envelopes touch"). Zero co-occurrences with a positive envelope
    overlap is the signature of a phantom veto.
    """
    if not a_times or not b_times:
        return None
    hits = 0
    for t in a_times:
        if any(abs(t - u) <= tol for u in b_times):
            hits += 1
    return hits


def describe_pair(a, b, tracklets, covis_enabled, covis_pairs, sample_tol,
                  geo_envelopes=None):
    """One row of the conflict grid: (gate, detail) for this member pair.

    THIS FUNCTION MUST STAY IN STEP WITH reconcile.conflict_reason. It deliberately
    reimplements the grid -- reconcile drops a vetoed pair before scoring, leaving no
    trace -- so a gate added there and forgotten here makes this tool report "no rule
    refused" about a pair that WAS refused. That is worse than not having the tool,
    because it sends the next investigation to the wrong rule. The geometric check
    below was added for exactly that reason; check reconcile.conflict_reason's gate
    list against this one whenever either changes.
    """
    ta, tb = tracklets[a], tracklets[b]

    # Checked FIRST, matching conflict_reason's order, so the reported gate is the
    # one reconcile would actually have returned.
    if geo_envelopes:
        impossible, detail = R.reachability_verdict(ta.get("floor"),
                                                    tb.get("floor"), geo_envelopes)
        if impossible:
            return (R.dlog.GEOMETRIC_UNREACHABLE,
                    f"physically unreachable: median "
                    f"{detail['median_required_speed']:.2f} units/s over "
                    f"{detail['pairings']} pairing(s) vs limit {detail['limit']:.2f} "
                    f"({detail['n_impossible']} individually impossible)")
    if a[0] == b[0]:
        frame_overlap = (min(ta["span"][1], tb["span"][1])
                         - max(ta["span"][0], tb["span"][0]))
        if R._spans_disjoint(ta["span"], tb["span"]):
            return None, f"same-cam, frame envelopes disjoint (gap {-frame_overlap})"
        co = _sampled_cooccurrences(ta["times"], tb["times"], sample_tol)
        secs = R.temporal_overlap_sec(ta["span_ts"], tb["span_ts"])
        verdict = ("REAL co-presence" if co else
                   "PHANTOM -- envelopes overlap but NO sampled instant does")
        return (R.dlog.TEMPORAL_CONFLICT_SAME_CAMERA,
                f"same-cam, envelope overlap {frame_overlap} frame(s)"
                + (f" / {secs:.1f}s" if secs is not None else "")
                + f", co-occurring sampled instants={co} -> {verdict}")
    if not covis_enabled:
        return None, "cross-cam, veto disabled"
    pair = frozenset((a[0], b[0]))
    if pair not in covis_pairs:
        return None, "cross-cam, pair UNLISTED -> unconstrained (D1)"
    tol = covis_pairs[pair]
    if tol is None:
        return None, "cross-cam, declared covisible -> never vetoed"
    secs = R.temporal_overlap_sec(ta["span_ts"], tb["span_ts"])
    if secs is None:
        return None, "cross-cam, no wall-clock span -> cannot judge"
    if secs > tol:
        return (R.dlog.TEMPORAL_CONFLICT_CROSS_CAMERA,
                f"cross-cam, wall-clock overlap {secs:.1f}s > tolerance {tol:.1f}s")
    return None, f"cross-cam, overlap {secs:.1f}s <= tolerance {tol:.1f}s"


def _mentions(line, key):
    """True when a reconcile merge-log line names this tracklet."""
    return f"({key[0]!r}, {key[1]})" in line


def report_already_merged(a, b, gid, tracklets, kw, lines):
    """They ARE one identity at these settings -- so report BY WHICH MERGE.

    This is the mode that validates a candidate fix, and the distinction it draws
    is the one that decides whether a fix will generalise:

      PHASE 1 united them  -> the two fragments cleared their own camera's bar on
                              their own pairwise evidence. Order-independent, so it
                              will behave the same on other footage.
      PHASE 2 united them  -> they arrived in one identity through a CLUSTER-level
                              score, after other merges had reshaped both sides.
                              Phase 2 re-scores per round and merges in score
                              order, so this outcome depends on what merged first.
                              It can be right and still be luck.

    A bar set at or below the fragment score converts the second case into the
    first, which is why the number matters and not just the outcome.
    """
    header("ALREADY ONE IDENTITY -- by which merge, and how robustly")
    per_cam = kw["same_camera_thresholds"] or {}
    cap = kw["max_observations_per_side"]
    proto_a = R._prototype(tracklets[a]["vectors"])
    proto_b = R._prototype(tracklets[b]["vectors"])
    rows_a = R._subsample_rows(R._unit_rows(tracklets[a]["vectors"]), cap)
    rows_b = R._subsample_rows(R._unit_rows(tracklets[b]["vectors"]), cap)
    score = R.score_observation_sets(rows_a, rows_b, proto_a, proto_b,
                                     mode=kw["scoring"],
                                     top_frac=kw["consensus_top_frac"], cap=cap)
    same_cam = a[0] == b[0]
    bar = (per_cam.get(a[0], kw["same_camera_threshold"]) if same_cam
           else kw["threshold"])
    print(f"  both {_fmt_key(a)} and {_fmt_key(b)} are reid {gid} here.")
    print(f"  fragment score = {score:.3f} ({kw['scoring']})   "
          f"applicable bar = {bar:.2f} "
          f"({'same-camera, ' + a[0] if same_cam else 'cross-camera'})")

    direct = [ln.strip() for ln in lines
              if _mentions(ln, a) and _mentions(ln, b) and "merge" in ln]
    if direct:
        print("\n  A merge line names BOTH fragments:")
        for ln in direct:
            print(f"    {ln}")
        if any("cluster merge" not in ln for ln in direct):
            print("\n  ROBUST. Phase 1 merged the two fragments directly, on their "
                  "own pairwise\n  evidence, before any cluster existed. That does "
                  "not depend on merge order.")
        else:
            print("\n  PHASE 2, not Phase 1: both fragments are named, but as "
                  "CLUSTER representatives.\n  The score that admitted it is a "
                  "cluster mean, and Phase 2 merges in score order,\n  so this "
                  "outcome is contingent on what merged before it.")
        return
    print("\n  NO merge line names both fragments, so they were united by a Phase 2 "
          "CLUSTER\n  union (Phase 2 logs cluster representatives, not members).")
    if same_cam and score < bar:
        print(f"  And Phase 1 CANNOT have done it: {score:.3f} < {bar:.2f}.")
    elif same_cam and kw["same_camera_reciprocal_best"]:
        # Phase 1 was ELIGIBLE and still did not fire. With mutual-best on, each
        # fragment must be the OTHER's single best above-bar partner -- and a person
        # with three fragments in one camera has two that look more like each other
        # than either looks like the third. Naming that partner is the whole
        # explanation, so compute it rather than leaving the reader to guess.
        best = {}
        for k in (a, b):
            cands = []
            for other in tracklets:
                if other == k or other[0] != k[0]:
                    continue
                if not R._spans_disjoint(tracklets[k]["span"],
                                         tracklets[other]["span"]):
                    continue
                p = R._prototype(tracklets[other]["vectors"])
                if p is None:
                    continue
                r = R._subsample_rows(R._unit_rows(tracklets[other]["vectors"]), cap)
                s = R.score_observation_sets(
                    R._subsample_rows(R._unit_rows(tracklets[k]["vectors"]), cap), r,
                    R._prototype(tracklets[k]["vectors"]), p, mode=kw["scoring"],
                    top_frac=kw["consensus_top_frac"], cap=cap)
                if s >= bar:
                    cands.append((s, other))
            best[k] = max(cands) if cands else None
        print(f"  Phase 1 was ELIGIBLE ({score:.3f} >= {bar:.2f}) and still did not "
              f"fire, because\n  same_camera_reciprocal_best requires each to be the "
              f"OTHER's best above-bar partner:")
        for k in (a, b):
            if best[k] is None:
                print(f"    {_fmt_key(k)}: no above-bar partner")
            else:
                s, other = best[k]
                print(f"    {_fmt_key(k)}'s best partner is {_fmt_key(other)} "
                      f"at {s:.3f}"
                      + ("  <-- not the other fragment"
                         if other != (b if k == a else a) else ""))
        print("\n  So lowering this camera's bar alone can NEVER make Phase 1 merge "
              "these two.\n  Whatever united them did so at CLUSTER level in Phase 2.")
    if same_cam:
        print("\n  ORDER-DEPENDENT either way. The right answer arrived through "
              "cluster-level\n  scoring after other merges had reshaped both sides -- "
              "correct here, but\n  contingent on what merged first, and the number "
              "that admitted it is a mean of\n  cluster means, not the "
              f"{score:.3f} these two fragments actually score.\n  That is the case "
              "for M.3: judge the shared-camera claim on the fragment pair.")
    print("\n  merge lines mentioning either fragment:")
    for ln in lines:
        if _mentions(ln, a) or _mentions(ln, b):
            print(f"    {ln.strip()}")


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        raise SystemExit(__doc__.strip().split("\n\n")[1])
    run_id = sys.argv[1]

    kw = reconcile_settings(extra_flags=("--url", "--path", "--tracklets"))
    url = arg("--url", "http://localhost:6333") or None
    store = PersonVectorStore(path=arg("--path", "qdrant_data"), url=url)

    header(f"WHY THESE TWO DID NOT MERGE -- run {run_id}")
    print(f"  settings: {describe_reconcile_kwargs(kw)}")
    print(f"  store:    {url or arg('--path', 'qdrant_data')} "
          f"-- {store.count()} point(s) total\n")

    tracklets = R._gather_tracklets(store, run_id)
    if not tracklets:
        raise SystemExit(f"[explain] run_id {run_id!r} has NO observations in this "
                         f"store -- check the id against the run log.")

    # Reconcile read-only, at the production settings, to get the clusters the run
    # actually produced. Same code path as the product, so the ids below are the
    # ids on the video.
    lines = []
    remap = R.reconcile_tracklets(_ReadOnlyStore(store), run_id=run_id,
                                 log=lines.append, **kw)
    if not remap:
        raise SystemExit("[explain] reconcile produced no assignments -- nothing "
                         "to explain.")
    print(f"  reconciled: {len(remap)} tracklet(s) -> "
          f"{len(set(remap.values()))} identities")

    # ---- which two clusters -------------------------------------------------
    by_gid = defaultdict(set)
    for key, gid in remap.items():
        by_gid[gid].add(key)

    tracklet_spec = arg("--tracklets")
    if tracklet_spec:
        wanted = []
        for part in tracklet_spec.split(","):
            cam, _, tid = part.strip().partition(":")
            wanted.append((cam, int(tid)))
        if len(wanted) != 2:
            raise SystemExit("[explain] --tracklets takes exactly two, "
                             "e.g. cam_219:7,cam_219:46")
        missing = [w for w in wanted if w not in remap]
        if missing:
            raise SystemExit(f"[explain] {missing} not in this run's remap")
        gid_a, gid_b = remap[wanted[0]], remap[wanted[1]]
        print(f"  {_fmt_key(wanted[0])} -> reid {gid_a};  "
              f"{_fmt_key(wanted[1])} -> reid {gid_b}")
        if gid_a == gid_b:
            # Not an error -- this is how a candidate fix is validated. Report WHICH
            # merge united them and whether that merge was order-dependent.
            report_already_merged(wanted[0], wanted[1], gid_a, tracklets, kw, lines)
            return
    else:
        if len(sys.argv) < 4:
            raise SystemExit("[explain] give two reid ids, e.g. "
                             f"... {run_id} 1 11   (or --tracklets cam:tid,cam:tid)")
        gid_a, gid_b = int(sys.argv[2]), int(sys.argv[3])
        for g in (gid_a, gid_b):
            if g not in by_gid:
                raise SystemExit(
                    f"[explain] reid {g} does not exist at these settings. "
                    f"Present: {sorted(by_gid)}")

    set_a, set_b = by_gid[gid_a], by_gid[gid_b]

    for gid, members in ((gid_a, set_a), (gid_b, set_b)):
        print(f"\n  reid {gid}: {len(members)} tracklet(s)")
        for k in sorted(members):
            t = tracklets[k]
            print(f"    {_fmt_key(k):<16} obs={len(t['vectors']):<4} "
                  f"frames={t['span'][0]}..{t['span'][1]:<6} "
                  f"wall={_fmt_span(t['span_ts'])}")

    # ---- the bar that applies ----------------------------------------------
    cams_a = {k[0] for k in set_a}
    cams_b = {k[0] for k in set_b}
    shared = cams_a & cams_b
    per_cam = kw["same_camera_thresholds"] or {}
    global_bar = kw["same_camera_threshold"]
    if shared:
        bar = R.strictest_same_camera_bar(shared, per_cam, global_bar)
        bars = ", ".join(f"{c}={per_cam.get(c, global_bar):.2f}"
                         for c in sorted(shared))
        why = (f"the two clusters SHARE {sorted(shared)}, so this is judged as a "
               f"SAME-camera claim at the STRICTEST of their bars ({bars}) "
               f"-- NOT the cross-camera {kw['threshold']:.2f}")
    else:
        bar = kw["threshold"]
        why = "the clusters are camera-disjoint, so the cross-camera bar applies"

    header("THE BAR")
    print(f"  bar = {bar:.2f}   ({why})")

    # ---- the score, under every mode ---------------------------------------
    protos = {k: R._prototype(tracklets[k]["vectors"]) for k in set_a | set_b}
    rows = {k: R._unit_rows(tracklets[k]["vectors"]) for k in set_a | set_b}
    proto_a = R._cluster_prototype(sorted(set_a), protos)
    proto_b = R._cluster_prototype(sorted(set_b), protos)
    cap = kw["max_observations_per_side"]

    def cluster_rows(members):
        """Every observation in the cluster, capped exactly as reconcile caps it."""
        mats = [rows[k] for k in sorted(members)]
        stacked = mats[0] if len(mats) == 1 else np.concatenate(mats, axis=0)
        return R._subsample_rows(stacked, cap)

    rows_a, rows_b = cluster_rows(set_a), cluster_rows(set_b)

    header("THE SCORE (cluster vs cluster)")
    for mode in R.SCORING_MODES:
        s = R.score_observation_sets(rows_a, rows_b, proto_a, proto_b, mode=mode,
                                     top_frac=kw["consensus_top_frac"], cap=cap)
        mark = "  <-- in force" if mode == kw["scoring"] else ""
        verdict = "PASSES" if s >= bar else f"FAILS by {bar - s:.3f}"
        print(f"  {mode:<14} {s:.3f}  vs bar {bar:.2f}  {verdict}{mark}")

    # Per shared camera, the comparison the same-camera claim is actually ABOUT
    # (Part M.3): the best fragment-to-fragment score inside that camera. This is
    # the number the bar was calibrated on, as opposed to a mean of cluster means.
    if shared:
        header("WHAT THE SAME-CAMERA CLAIM IS ACTUALLY ABOUT (M.3)")
        print("  Per shared camera, the best fragment-vs-fragment score -- the "
              "comparison\n  the per-camera bar was calibrated on, as opposed to a "
              "mean of cluster means.\n  Under ALL THREE modes, because a mode "
              "change is only worth making if it moves\n  THIS number in the right "
              "direction (the cluster-level table above can move the\n  other way "
              "-- two multi-person clusters have few matching view pairs).\n")
        # EVERY pair, not just the best one. The best pair says whether the merge
        # COULD happen; the WEAKEST says whether it survives the all-member-pairs
        # rule that Phase 1 rounds apply (M.9.11) -- one disagreeing fragment is
        # enough to refuse a merge, so a table showing only the best hides the
        # number that actually decides.
        print(f"    {'camera':<10}{'fragment pair':<32}"
              + "".join(f"{m:>15}" for m in R.SCORING_MODES) + f"{'bar':>7}")
        for cam in sorted(shared):
            cbar = per_cam.get(cam, kw["same_camera_threshold"])
            rated = []
            for x in sorted(k for k in set_a if k[0] == cam):
                for y in sorted(k for k in set_b if k[0] == cam):
                    scores = {
                        m: R.score_observation_sets(
                            R._subsample_rows(rows[x], cap),
                            R._subsample_rows(rows[y], cap),
                            protos[x], protos[y], mode=m,
                            top_frac=kw["consensus_top_frac"], cap=cap)
                        for m in R.SCORING_MODES}
                    rated.append((scores[kw["scoring"]], x, y, scores))
            # Ranked by the mode IN FORCE, so the top row is the pair reconcile
            # would have picked and the bottom row is the one that can veto it.
            rated.sort(reverse=True)
            for i, (_, x, y, scores) in enumerate(rated):
                cells = "".join(
                    f"{scores[m]:>10.3f} {'PASS' if scores[m] >= cbar else 'fail':<4}"
                    for m in R.SCORING_MODES)
                tag = ("  <- best" if i == 0 and len(rated) > 1 else
                       "  <- WEAKEST (this one can veto)"
                       if i == len(rated) - 1 and len(rated) > 1 else "")
                print(f"    {cam if i == 0 else '':<10}"
                      f"{_fmt_key(x) + ' vs ' + _fmt_key(y):<32}"
                      f"{cells}{cbar:>7.2f}{tag}")
        print("\n  A fragment pair that FAILS its own camera's bar is one person's "
              "two appearance\n  modes scoring below a bar set for two different "
              "people. That is the defect at\n  its root -- everything downstream "
              "is the consequence.")
        print("  If the BEST pair passes but the WEAKEST fails, the merge is "
              "blocked by the\n  all-member-pairs rule, not by the bar -- a "
              "different fix (see M.9.11).")

    # ---- the conflict grid -------------------------------------------------
    covis_enabled, covis_pairs = kw["covisibility"]
    # Rebuild the geometric envelopes the same way reconcile does -- from THIS run's
    # recorded positions, never from a calibration file (geometry/__init__.py
    # invariant 1). Empty when the veto is off or the run carries no positions, which
    # makes the grid's geometry column silent rather than misleading.
    geo_cfg = kw.get("geometry") or {}
    geo_envelopes = {}
    if geo_cfg.get("enabled"):
        geo_envelopes = R.build_speed_envelope(
            tracklets,
            clock_error_sec=float(geo_cfg.get("clock_error_sec", 0.5)),
            safety_factor=float(geo_cfg.get("safety_factor", 1.5)),
            log=lambda m: print(f"  {m}"))
    # Tolerance for "the same sampled instant". Observations are taken every
    # reid.interval_sec, so two boxes that really were on screen together land
    # within about one cadence of each other; 1.5x leaves room for the jitter in
    # `ts` (receive time, #15) without accepting a whole gap as co-presence.
    sample_tol = float((load_config().get("reid") or {}).get("interval_sec")
                       or 0.4) * 1.5

    header("THE CONFLICT GRID (hard vetoes, checked BEFORE any score)")
    print(f"  {len(set_a)} x {len(set_b)} = {len(set_a) * len(set_b)} member "
          f"pair(s). ONE failure removes this merge from the candidate")
    print("  set entirely -- no score is ever taken, so no threshold or scoring "
          "mode can reach it.\n")
    blockers = []
    for x in sorted(set_a):
        for y in sorted(set_b):
            gate, detail = describe_pair(x, y, tracklets, covis_enabled,
                                         covis_pairs, sample_tol, geo_envelopes)
            tag = "BLOCK" if gate else "  ok "
            if gate:
                blockers.append((x, y, gate, detail))
            print(f"  {tag} {_fmt_key(x):<16} {_fmt_key(y):<16} {detail}")

    header("VERDICT")
    if blockers:
        print(f"  BLOCKED. {len(blockers)} of {len(set_a) * len(set_b)} member "
              f"pair(s) carry a hard veto, so reid {gid_a} and reid {gid_b}")
        print("  were never scored against each other at all:")
        for x, y, gate, detail in blockers:
            print(f"    {gate}  {_fmt_key(x)} <-> {_fmt_key(y)}")
            print(f"      {detail}")
        phantom = [b for b in blockers if "PHANTOM" in b[3]]
        if phantom:
            print(f"\n  {len(phantom)} of those are PHANTOM envelope overlaps: the "
                  f"two tracklets never share\n  a sampled instant, so the veto is "
                  f"asserting a co-presence that did not happen.\n  -> Part M.2 "
                  f"(occupancy intervals instead of min/max envelopes) is the fix.")
        else:
            print("\n  All blocking overlaps are REAL co-presence, so the blocking "
                  "members genuinely\n  are different people -- which means a WRONG "
                  "member was absorbed into one of\n  these clusters and is now "
                  "vetoing on its behalf.\n  -> Part M.4 (stop weak-edge capture) "
                  "is the fix.")
    else:
        s = R.score_observation_sets(rows_a, rows_b, proto_a, proto_b,
                                     mode=kw["scoring"],
                                     top_frac=kw["consensus_top_frac"], cap=cap)
        print(f"  NOT blocked by any veto. The pair was scored at {s:.3f} against "
              f"a bar of {bar:.2f}")
        print(f"  -> {'it should have merged; look at reciprocal-best' if s >= bar else 'it failed the bar'}.")
        if shared:
            print("  Because the clusters share a camera the bar is the strictest "
                  "SAME-camera bar\n  over every shared camera, applied to a mean "
                  "of cluster means. -> Part M.3.")

    print("\n  Merge log from this reconcile (for context):")
    for line in lines:
        if "merge" in line or "SKIP" in line:
            print(f"   {line.strip()}")


if __name__ == "__main__":
    main()
