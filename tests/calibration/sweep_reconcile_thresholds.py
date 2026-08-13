"""
Re-run reconcile over a COMPLETED run's stored observations, at different
thresholds, WITHOUT re-running the cameras.

    python tests/calibration/sweep_reconcile_thresholds.py <run_id>
    python tests/calibration/sweep_reconcile_thresholds.py <run_id> --cross 0.63,0.70,0.75,0.80
    python tests/calibration/sweep_reconcile_thresholds.py <run_id> --same cam_219=0.80,cam_213=0.80

WHY THIS EXISTS. Every threshold question so far has cost a live run: four
cameras, an operator walking around, five minutes, and a finalization that must
not be interrupted. That is why four rounds of tuning were reverted without
learning anything -- the feedback loop was too expensive to close.

But reconcile is a PURE FUNCTION of the stored observations. They are still in
Qdrant under their run_id long after the run ends, so the whole threshold space
can be explored in seconds. This is REMEDIATION_PLAN.md #23 (offline
replayability) for the part that matters most: the merge decisions.

READ-ONLY, BY CONSTRUCTION. reconcile_tracklets calls set_global_id and
clear_global_id as it goes, which would rewrite the gallery's ids on every
combination swept. _ReadOnlyStore intercepts both, so a sweep can never corrupt
the run it is measuring -- and never disagrees with what the live run produced,
because it reads the same vectors through the same code path.

WHAT IT CANNOT TELL YOU. Cluster composition, not correctness. It reports how
many identities each setting produces, how big the largest cluster gets, and how
many merges land below the measured different-person ceiling -- but only the
operator watching the videos knows whether GID 6 is one person or four. Use this
to find the settings worth rendering, then render those.
"""

import sys
from collections import defaultdict

from _common import arg, bootstrap, header, parse_same, reconcile_settings

bootstrap()

from database.store import PersonVectorStore                      # noqa: E402
from identity.reconcile import (SCORING_MODES,                     # noqa: E402
                                describe_reconcile_kwargs,
                                reconcile_tracklets)

# Measured different-person prototype ceiling, same-camera, n=12 (Part H.4).
# Cross-camera stranger scores run LOWER than same-camera, so a cross-camera
# merge below this is suggestive of a false merge, not proof of one.
STRANGER_CEILING = 0.661


class _ReadOnlyStore:
    """Real observations in, no writes out. Everything reconcile reads is the
    genuine article; everything it writes is discarded."""

    def __init__(self, store):
        self._store = store
        self.client = store.client
        self.collection = store.collection
        self.writes = 0

    def set_global_id(self, point_ids, global_id):
        self.writes += len(point_ids)

    def clear_global_id(self, point_ids):
        self.writes += len(point_ids)


def _parse_same_variants(text):
    """Several same-camera settings at once, separated by ';':

        "cam_213=0.80,cam_224=0.80 ; cam_213=0.87,cam_224=0.87"

    Both axes matter and they fail differently: the cross-camera bar governs
    whether one person follows themselves BETWEEN cameras, the same-camera bar
    whether two people get fused WITHIN one. A cluster welded together inside a
    camera cannot be separated by any cross-camera bar, so sweeping only the
    cross axis can miss the defect entirely.
    """
    variants = []
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parsed = parse_same(chunk)
        label = ", ".join(f"{c}={v:.2f}" for c, v in sorted(parsed.items()))
        variants.append((label or "global only", parsed))
    return variants or [("global only", {})]


def run_once(store, run_id, base_kw, cross, per_camera, scoring=None):
    """One reconcile at these settings -> (remap, merge log lines).

    `base_kw` is the PRODUCTION setting set (reconcile_settings()); only the axes
    being swept are overridden. Before this, run_once built its kwargs by hand and
    passed neither `covisibility` nor `same_camera_reciprocal_best`, so every row
    of every sweep described a clustering that does not ship -- with the
    cross-camera veto OFF, the guard that dominates the real candidate space.
    See REMEDIATION_PLAN.md Part M.4.
    """
    lines = []
    kw = dict(base_kw)
    kw["threshold"] = cross
    kw["same_camera_thresholds"] = per_camera
    # scoring=None means "whatever config says", so the baseline row is the
    # shipped behaviour rather than a mode this script chose.
    if scoring is not None:
        kw["scoring"] = scoring
    remap = reconcile_tracklets(_ReadOnlyStore(store), run_id=run_id,
                               log=lines.append, **kw)
    return remap, lines


def summarize(remap, lines):
    """Cluster shape + how many accepted merges sit below the stranger ceiling."""
    clusters = defaultdict(list)
    for key, gid in remap.items():
        clusters[gid].append(key)
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    multi = sum(1 for v in clusters.values() if len({k[0] for k in v}) > 1)

    weak, cross_scores = 0, []
    for line in lines:
        if "cross-camera cluster merge" not in line:
            continue
        try:
            score = float(line.split("cosine ")[1].split()[0])
        except (IndexError, ValueError):
            continue
        cross_scores.append(score)
        if score < STRANGER_CEILING:
            weak += 1
    return {
        "identities": len(clusters),
        "sizes": sizes,
        "max": sizes[0] if sizes else 0,
        "multi_camera": multi,
        "cross_merges": len(cross_scores),
        "below_ceiling": weak,
        "min_cross": min(cross_scores) if cross_scores else None,
        "clusters": clusters,
    }


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        raise SystemExit(__doc__.strip().split("\n\n")[1])
    run_id = sys.argv[1]

    # Base = exactly what production would run (config.yaml through reconcile's own
    # resolver), then the sweep overrides only the axes it is sweeping. The FIRST
    # row of a default sweep must therefore reproduce the live run's identity count
    # -- that equality is what makes every other row meaningful.
    base_kw = reconcile_settings(owns=("--cross", "--same", "--scoring"),
                                 extra_flags=("--url", "--path"))
    crosses = [float(x) for x in
               (arg("--cross") or f"{base_kw['threshold']:.2f},0.70,0.75,0.80"
                ).split(",")]
    default_same = ", ".join(f"{c}={v}" for c, v in
                             sorted((base_kw["same_camera_thresholds"] or {}).items()))
    variants = _parse_same_variants(arg("--same", default_same))
    # --scoring here is a LIST (the sweep compares modes); reconcile_settings has
    # already applied a single --scoring to the base, so drop it from the base to
    # avoid describing a mode twice.
    scorings = (arg("--scoring", "") or "").split(",")
    scorings = [m.strip() for m in scorings if m.strip()] or [None]
    for m in scorings:
        if m is not None and m not in SCORING_MODES:
            raise SystemExit(f"[sweep] unknown --scoring {m!r}; "
                             f"expected any of {list(SCORING_MODES)}")

    url = arg("--url", "http://localhost:6333") or None
    store = PersonVectorStore(path=arg("--path", "qdrant_data"), url=url)
    print(f"[sweep] store: {url or arg('--path', 'qdrant_data')} "
          f"-- {store.count()} point(s) total")
    # The settings the rows are relative to, named in full. A sweep whose switches
    # differ from production is not a measurement of production.
    print(f"[sweep] base settings: {describe_reconcile_kwargs(base_kw)}")

    # Refuse to sweep nothing. Without this a wrong run_id reports "0 identities"
    # at every setting, which reads like a devastating result rather than a typo.
    runs = defaultdict(int)
    offset = None
    while True:
        pts, offset = store.client.scroll(store.collection, limit=1000,
                                          offset=offset, with_payload=True,
                                          with_vectors=False)
        for p in pts:
            runs[(p.payload or {}).get("run_id")] += 1
        if offset is None:
            break
    if not runs.get(run_id):
        raise SystemExit(
            f"[sweep] run_id {run_id!r} has NO observations in this store.\n"
            f"        available: "
            + ", ".join(f"{r} ({n})" for r, n in sorted(runs.items(),
                                                        key=lambda kv: str(kv[0]))))
    print(f"[sweep] run_id={run_id}: {runs[run_id]} observation(s)")

    header("SWEEP -- identities and cluster shape per setting")
    print("  'below ceiling' counts accepted CROSS-camera merges under "
          f"{STRANGER_CEILING} (the measured")
    print("  different-person prototype ceiling). Those are the false-merge "
          "candidates.\n")
    print(f"  {'same-camera bars':<38}{'cross':>7}{'ids':>6}{'max':>6}"
          f"{'multi':>7}{'xmerges':>9}{'<ceil':>7}{'min x':>8}")
    print("  " + "-" * 88)

    results = {}
    variants = [(f"{lab}" if m is None else f"{lab} [{m}]", pc, m)
                for m in scorings for lab, pc in variants]
    for label, per_camera, mode in variants:
        for cross in crosses:
            remap, lines = run_once(store, run_id, base_kw, cross, per_camera,
                                    scoring=mode)
            s = summarize(remap, lines)
            results[(label, cross)] = s
            # "shipped" is now derived from config, not from a hardcoded 0.63 that
            # goes stale the moment the value moves -- which it has, twice.
            shipped = (abs(cross - base_kw["threshold"]) < 1e-9
                       and per_camera == (base_kw["same_camera_thresholds"] or {})
                       and (mode is None or mode == base_kw["scoring"]))
            mark = "  <-- shipped" if shipped else ""
            print(f"  {label:<38}{cross:>7.2f}{s['identities']:>6}{s['max']:>6}"
                  f"{s['multi_camera']:>7}{s['cross_merges']:>9}"
                  f"{s['below_ceiling']:>7}"
                  f"{(s['min_cross'] if s['min_cross'] else 0):>8.3f}{mark}")

    header("CLUSTER COMPOSITION for every setting swept")
    print("  Check these against what you SAW. A cluster holding several people")
    print("  is a false merge; one person appearing in several clusters is a split.")
    print("  One block per combination -- keep the grid small when reading these.\n")
    for label, _, _mode in variants:
        for cross in crosses:
            s = results[(label, cross)]
            print(f"  --- same[{label}]  cross={cross:.2f}: "
                  f"{s['identities']} identities, sizes {s['sizes']}")
            for gid, keys in sorted(s["clusters"].items()):
                if len(keys) < 2:
                    continue
                by_cam = defaultdict(list)
                for cam, tid in sorted(keys):
                    by_cam[cam].append(tid)
                desc = "  ".join(f"{c}({','.join(str(t) for t in sorted(v))})"
                                 for c, v in sorted(by_cam.items()))
                print(f"      GID {gid:>3}: {desc}")
            print()

    header("HOW TO READ THIS")
    print("""  Raising the cross-camera bar can only SPLIT clusters, never merge more, so
  'ids' rises and 'max' falls as you go right. The question is where the split
  stops separating strangers and starts separating one person from themselves.

  Two signals bound the answer:
    * 'below ceiling' should reach 0 -- accepting a cross-camera merge under the
      measured different-person ceiling is very likely fusing two people
    * 'multi' (identities spanning >1 camera) should NOT collapse -- if it falls
      sharply, the bar has stopped real people from following themselves between
      cameras, which is the failure the low bar was protecting against

  Pick the setting where 'below ceiling' hits 0 while 'multi' is still healthy,
  then RE-RUN THE PIPELINE at that setting and watch the videos. This tool ranks
  candidates; only the videos confirm.""")


if __name__ == "__main__":
    main()
