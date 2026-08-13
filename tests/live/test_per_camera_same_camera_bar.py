"""
Plan item #40 -- `same_camera_threshold` is PER-CAMERA.

WHY THIS EXISTS. On real field data (run 20260730_093723, 286 decisions) a single
global 0.90 gave cam_206 9.0 eligible same-camera partners per subject while
cam_213 got ZERO across all 11 of its subjects and cam_224 managed it for 5 of 17.
A camera with no same-camera merging cannot join one person's front-view and
back-view fragments, so each is absorbed cross-camera into a DIFFERENT cluster --
the operator-visible "reid 2 by his front, reid 7 by his back". The analyser also
showed no global value can work: the per-subject boundaries overlap (p95 of the
top "different" score 0.816 > p5 of the worst "same" score 0.719).

WHAT IS PINNED HERE:
  1. the rule that resolves a bar when two clusters share SEVERAL cameras (max)
  2. config resolution, including every malformed-input path (fail-soft AND loud:
     a silently-ignored `80` where `0.80` was meant would disable same-camera
     merging for that camera entirely, which is the very failure #40 fixes)
  3. NO OVERRIDES => byte-identical behaviour. #40 must be a no-op for anyone
     who does not opt a camera in
  4. the bar applies in BOTH phases -- phase 1 (tracklet pairs) and phase 2
     (cluster pairs, the `pair_threshold` lookup)
  5. an override on a camera that is present but NOT shared by the two clusters
     does not lower the bar
  6. the hard guards are untouched: a time-overlapping same-camera pair never
     merges however low the bar, and the cross-camera lane keeps its own threshold
  7. the applied bar reaches the decision log, which is what lets
     tests/calibration/analyze_decision_log.py report per-camera bars and makes
     the change measurable in the next run

Deterministic and synthetic: exact orthonormal axes, so every cosine below is the
number written down, not an approximation.
"""

import sys

import numpy as np
from types import SimpleNamespace

from _synth import Check, DIM
from identity import decision_log as dlog
from identity.decision_log import DecisionLog
from identity.reconcile import (reconcile_tracklets,
                                resolve_same_camera_thresholds,
                                strictest_same_camera_bar)

THRESHOLD = 0.63              # identity.reconcile.threshold (cross-camera)
GLOBAL_SAME = 0.90            # identity.reconcile.same_camera_threshold
LOWERED = 0.80                # what #40 sets for cam_213 / cam_224


def _unit(*coeffs):
    """Unit vector with the given leading coefficients (rest zero) -- the cosine
    between two such vectors is exactly controllable."""
    v = np.zeros(DIM, np.float32)
    for i, c in enumerate(coeffs):
        v[i] = c
    return v / np.linalg.norm(v)


def _at_cosine(c):
    """A unit vector whose cosine with _unit(1.0) is exactly `c`."""
    return _unit(c, float(np.sqrt(1.0 - c * c)))


class _FakeStore:
    """Minimal stand-in for PersonVectorStore -- reconcile only needs
    `client.scroll`, `collection`, `set_global_id` and `clear_global_id`."""

    collection = "persons"

    def __init__(self, observations):
        self.assigned = {}
        self.cleared = []
        self.client = SimpleNamespace(
            scroll=lambda col, limit, offset, with_payload, with_vectors: (
                observations, None))

    def set_global_id(self, point_ids, global_id):
        for p in point_ids:
            self.assigned[p] = global_id

    def clear_global_id(self, point_ids):
        self.cleared.extend(point_ids)


def _observations(specs, obs_per_tracklet=6):
    """specs: [(camera, track_id, vector, first_frame, original_gid)] -> points."""
    out = []
    for cam, tid, vec, f0, gid in specs:
        for f in range(f0, f0 + obs_per_tracklet):
            out.append(SimpleNamespace(
                id=f"{cam}-{tid}-{f}",
                vector=vec.tolist(),
                payload={"camera": cam, "track_id": tid, "frame": f,
                         "run_id": "r1", "reid_id": gid},
            ))
    return out


def _run(specs, per_camera=None, decision_log=None, lines=None):
    """Reconcile `specs`; return (n_identities, remap)."""
    store = _FakeStore(_observations(specs))
    remap = reconcile_tracklets(
        store, threshold=THRESHOLD, run_id="r1",
        same_camera_threshold=GLOBAL_SAME,
        same_camera_thresholds=per_camera,
        require_reciprocal_best=True, min_tracklet_observations=1,
        decision_log=decision_log,
        log=(lines.append if lines is not None else (lambda *_a, **_k: None)))
    return len(set(remap.values())), remap


def main():
    c = Check("#40: per-camera same_camera_threshold")

    # ---------------------------------------------------------------- 1. rule
    print("\n1. THE RULE -- strictest bar over the cameras a claim covers")

    per = {"cam_213": 0.80, "cam_224": 0.80}
    c.eq(strictest_same_camera_bar(["cam_213"], per, GLOBAL_SAME), 0.80,
         "one overridden camera uses its own bar")
    c.eq(strictest_same_camera_bar(["cam_206"], per, GLOBAL_SAME), GLOBAL_SAME,
         "a camera with no override keeps the global bar")
    c.eq(strictest_same_camera_bar(["cam_213", "cam_206"], per, GLOBAL_SAME),
         GLOBAL_SAME,
         "clusters sharing TWO cameras must clear the STRICTER bar (0.90, not 0.80)")
    c.eq(strictest_same_camera_bar(["cam_213", "cam_224"], per, GLOBAL_SAME), 0.80,
         "two loosened cameras: strictest of them is still 0.80")
    c.eq(strictest_same_camera_bar([], per, GLOBAL_SAME), GLOBAL_SAME,
         "no shared camera degrades to the global bar (no same-camera claim)")

    # -------------------------------------------------------------- 2. config
    print("\n2. CONFIG RESOLUTION -- fail-soft and LOUD on every bad input")

    c.eq(resolve_same_camera_thresholds(
        {"per_camera": {"cam_213": {"same_camera_threshold": 0.8},
                        "cam_224": {"same_camera_threshold": 0.8}}}),
        {"cam_213": 0.8, "cam_224": 0.8}, "the shipped shape resolves")
    c.eq(resolve_same_camera_thresholds({}), {}, "no per_camera block -> no overrides")
    c.eq(resolve_same_camera_thresholds(None), {}, "no reconcile config at all -> {}")
    c.eq(resolve_same_camera_thresholds({"per_camera": None}), {},
         "an empty per_camera block -> no overrides")

    for name, cfg in (
            ("a non-numeric bar", {"cam_a": {"same_camera_threshold": "high"}}),
            ("a bar outside [0,1] (80 for 0.80)",
             {"cam_a": {"same_camera_threshold": 80}}),
            ("a negative bar", {"cam_a": {"same_camera_threshold": -0.5}}),
            ("overrides that are not a mapping", {"cam_a": 0.8}),
    ):
        said = []
        got = resolve_same_camera_thresholds({"per_camera": cfg},
                                             log=lambda m, s=said: s.append(m))
        c.ok(got == {} and said, f"{name} is ignored AND reported "
                                f"(got={got}, said={len(said)} line(s))")

    said = []
    got = resolve_same_camera_thresholds(
        {"per_camera": {"cam_a": {"same_camera_threshold": 0.8,
                                  "min_tracklet_observations": 5}}},
        log=lambda m, s=said: s.append(m))
    c.ok(got == {"cam_a": 0.8} and any("min_tracklet_observations" in m for m in said),
         "an unhonoured key is reported but does not discard the valid one")

    said = []
    resolve_same_camera_thresholds({"per_camera": ["cam_213"]},
                                   log=lambda m, s=said: s.append(m))
    c.ok(said, "a per_camera block of the wrong TYPE is reported, not crashed on")

    # --------------------------------------------------- 3. no override = no-op
    print("\n3. NO OVERRIDES => BEHAVIOUR IS UNCHANGED (#40 is opt-in)")

    P = _unit(1.0)
    scene = [("cam_A", 1, P, 0, 1),
             ("cam_A", 3, _at_cosine(0.95), 100, 3),      # genuine fragment
             ("cam_B", 2, P, 0, 2),
             ("cam_B", 4, _at_cosine(0.70), 0, 4)]        # look-alike
    base_n, base_remap = _run(scene, per_camera=None)
    for label, per_camera in (("{}", {}),
                              ("a camera not in the run", {"cam_ZZZ": 0.50})):
        n, remap = _run(scene, per_camera=per_camera)
        c.ok((n, remap) == (base_n, base_remap),
             f"identical remap with per_camera={label} "
             f"({base_n} identity(ies), {len(base_remap)} tracklets)")

    # ------------------------------------------------------------- 4. phase 1
    print("\n4. PHASE 1 -- a tracklet pair is judged by ITS camera's bar")

    # One person's front and back view in cam_213, time-disjoint, cosine 0.85:
    # exactly the case that produced zero merges in the field at 0.90.
    frag = [("cam_213", 1, P, 0, 1), ("cam_213", 2, _at_cosine(0.85), 100, 2)]
    n, _ = _run(frag, per_camera=None)
    c.eq(n, 2, "at the global 0.90 a 0.85 same-camera fragment stays ORPHANED "
               "(the defect #40 fixes)")
    n, _ = _run(frag, per_camera={"cam_213": LOWERED})
    c.eq(n, 1, "at cam_213=0.80 the same 0.85 fragment MERGES into one identity")

    # The lowered bar is still a bar, not an unconditional merge.
    n, _ = _run([("cam_213", 1, P, 0, 1), ("cam_213", 2, _at_cosine(0.75), 100, 2)],
                per_camera={"cam_213": LOWERED})
    c.eq(n, 2, "a 0.75 pair is still rejected at 0.80 (the bar is real)")

    # An override must not leak into a camera it was not configured for.
    n, _ = _run([("cam_206", 1, P, 0, 1), ("cam_206", 2, _at_cosine(0.85), 100, 2)],
                per_camera={"cam_213": LOWERED})
    c.eq(n, 2, "lowering cam_213 does NOT lower cam_206 (0.85 still rejected there)")

    # ------------------------------------------------------------- 5. phase 2
    print("\n5. PHASE 2 -- the pair_threshold lookup on CLUSTER pairs")

    # Constructed so phase 1 CANNOT do the merge in either configuration: the
    # tracklet-to-tracklet score inside cam_213 is 0.75, below both bars. Only the
    # cluster prototype (pulled toward the fragment by the cam_206 member) reaches
    # the 0.80-0.90 band, so whatever merges here went through phase 2.
    A = _unit(1.0)                                     # cam_213, first sighting
    F = _unit(0.75, float(np.sqrt(1.0 - 0.75 ** 2)))   # cam_213, later fragment
    B = _unit(0.90, 0.30, float(np.sqrt(1.0 - 0.81 - 0.09)))   # cam_206, same person
    span = [("cam_213", 1, A, 0, 1), ("cam_213", 2, F, 100, 2), ("cam_206", 9, B, 0, 9)]

    # Assert the fixture really does exercise the intended path, rather than
    # trusting arithmetic done while writing the test.
    x_proto = (A + B) / np.linalg.norm(A + B)
    y_proto = (B + F) / np.linalg.norm(B + F)
    c.ok(float(A @ F) < LOWERED,
         f"fixture: the cam_213 tracklet pair scores {float(A @ F):.3f} < 0.80, "
         f"so phase 1 cannot merge it under EITHER config")
    c.ok(LOWERED <= float(x_proto @ F) < GLOBAL_SAME
         and LOWERED <= float(y_proto @ A) < GLOBAL_SAME,
         f"fixture: the phase-2 cluster score is in [0.80, 0.90) "
         f"({float(x_proto @ F):.3f} / {float(y_proto @ A):.3f})")

    log2 = DecisionLog(run_id="r1")
    n, _ = _run(span, per_camera={"cam_213": LOWERED}, decision_log=log2)
    c.eq(n, 1, "a cluster spanning cam_213+cam_206 absorbs the cam_213 fragment "
               "at cam_213's 0.80 bar")
    p2_accepts = [r for r in log2.records if r.accepted_partner
                  and r.phase == "cross_camera" and r.context == "same_camera"]
    c.ok(p2_accepts, "the merge is recorded as a PHASE 2 same-camera-context "
                     "decision (so pair_threshold is what applied the bar)")
    if p2_accepts:
        c.eq(p2_accepts[0].gates[dlog.ABSOLUTE_THRESHOLD].threshold, LOWERED,
             "and the record carries cam_213's bar, not the global one")

    n, _ = _run(span, per_camera=None)
    c.eq(n, 2, "at the global 0.90 that same cluster leaves the fragment behind")

    # The bar comes from the SHARED camera. cam_206 is in the pair but not shared
    # (the fragment is cam_213-only), so loosening cam_206 must change nothing.
    n, _ = _run(span, per_camera={"cam_206": LOWERED})
    c.eq(n, 2, "an override on a PRESENT but NOT SHARED camera does not lower the bar")

    # ---------------------------------------------------------- 6. hard guards
    print("\n6. THE HARD GUARDS ARE UNTOUCHED")

    n, _ = _run([("cam_213", 1, P, 0, 1), ("cam_213", 2, P, 0, 2)],
                per_camera={"cam_213": 0.10})
    c.eq(n, 2, "time-overlapping same-camera tracklets never merge, even at 0.10")

    n, _ = _run([("cam_213", 1, P, 0, 1), ("cam_206", 2, _at_cosine(0.70), 0, 2)],
                per_camera={"cam_213": LOWERED})
    c.eq(n, 1, "camera-disjoint clusters still use the cross-camera threshold (0.70 "
               "clears 0.63)")

    n, _ = _run([("cam_213", 1, P, 0, 1), ("cam_206", 2, _at_cosine(0.60), 0, 2)],
                per_camera={"cam_213": 0.10})
    c.eq(n, 2, "and lowering a SAME-camera bar cannot admit a sub-0.63 cross-camera "
               "match")

    # ------------------------------------------------------- 7. decision log
    print("\n7. THE APPLIED BAR IS VISIBLE TO THE ANALYSER")

    log7 = DecisionLog(run_id="r1")
    _run([("cam_213", 1, P, 0, 1), ("cam_213", 2, _at_cosine(0.85), 100, 2),
          ("cam_206", 3, P, 0, 3), ("cam_206", 4, _at_cosine(0.85), 100, 4)],
         per_camera={"cam_213": LOWERED}, decision_log=log7)
    p1 = [r for r in log7.records if r.phase == "same_camera"]
    bars = {tuple(r.cameras)[0]: r.gates[dlog.ABSOLUTE_THRESHOLD].threshold
            for r in p1}
    c.eq(bars.get("cam_213"), LOWERED,
         "cam_213 phase-1 records log the 0.80 bar")
    c.eq(bars.get("cam_206"), GLOBAL_SAME,
         "cam_206 phase-1 records log the 0.90 bar")

    # Same run, both bars present: this is what makes the analyser's section 1
    # report `configured same_camera_threshold: [0.8, 0.9]` and section 5's
    # per-camera eligible-set sizes interpretable after the change.
    c.eq(sorted({b for b in bars.values() if b is not None}), [LOWERED, GLOBAL_SAME],
         "one run carries BOTH bars, per camera")

    # -------------------------------------------------- 8. absent-camera check
    print("\n8. A CONFIGURED CAMERA THAT IS NOT IN THE RUN IS REPORTED")

    lines = []
    _run([("cam_213", 1, P, 0, 1), ("cam_213", 2, _at_cosine(0.85), 100, 2)],
         per_camera={"cam_213": LOWERED, "cam_XYZ": 0.70}, lines=lines)
    c.ok(any("cam_XYZ" in m for m in lines),
         "a typo'd / retired camera name is named in the log (D1: never silent)")
    c.ok(any("cam_213=0.80" in m for m in lines),
         "the effective bars are logged, so a run's own log records what applied")

    c.done()


if __name__ == "__main__":
    main()
