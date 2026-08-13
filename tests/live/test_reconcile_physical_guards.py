"""
Two guards that cut false merges WITHOUT moving any threshold.

Both exist because the same-camera bar was being asked to do a job it cannot do.
Lower it and strangers fuse; raise it and one person shatters. These remove the
trade instead of repositioning it.

1. RECIPROCAL BEST IN PHASE 1. Phase 2 has always required two clusters to be each
   other's best partner. Phase 1 never did -- it merged EVERY above-bar pair in
   score order. The fixture here is taken from the real failure: in run
   20260730_111232, cam_224's tracklet 58 had its best partner at 0.937 and still
   merged with 112 at 0.823 and 120 at 0.872, fusing two people into one reid.
   Mutual-best refuses both, at an unchanged bar.

2. CROSS-CAMERA SIMULTANEITY VETO (#38). `conflict()` skipped every cross-camera
   pair, so two tracklets overlapping IN TIME in cameras that cannot both see one
   person merged on appearance alone. That is physically impossible regardless of
   cosine, and no threshold reaches it. Requires wall-clock `ts` (#26), because
   frame indices are per-camera and meaningless across 15 vs 25 fps.

Both are OFF by default, and this file pins that too: they must change nothing
until switched on deliberately.
"""

import sys

import numpy as np
from types import SimpleNamespace

from _synth import Check, DIM
from identity.reconcile import (reconcile_tracklets, resolve_covisibility,
                                temporal_overlap_sec)

THRESHOLD = 0.63
SAME_BAR = 0.80


def _unit(*coeffs):
    v = np.zeros(DIM, np.float32)
    for i, c in enumerate(coeffs):
        v[i] = c
    return v / np.linalg.norm(v)


def _at(c):
    return _unit(c, float(np.sqrt(1.0 - c * c)))


class _FakeStore:
    collection = "persons"

    def __init__(self, observations):
        self.assigned, self.cleared = {}, []
        self.client = SimpleNamespace(
            scroll=lambda col, limit, offset, with_payload, with_vectors: (
                observations, None))

    def set_global_id(self, point_ids, global_id):
        for p in point_ids:
            self.assigned[p] = global_id

    def clear_global_id(self, point_ids):
        self.cleared.extend(point_ids)


def _observations(specs, per=6, ts_step=1.0):
    """specs: [(cam, track_id, vec, first_frame, gid, ts0)] -- ts in seconds.

    `ts_step` of 1.0s gives each tracklet a 5-SECOND wall-clock span. That matters:
    with a 0.1s step every span was 0.5s long, so a 1.0s veto tolerance could never
    be exceeded -- the test "passed" the veto while never exercising it. Real
    tracklets last seconds to minutes.
    """
    out = []
    for cam, tid, vec, f0, gid, ts0 in specs:
        for i in range(per):
            out.append(SimpleNamespace(
                id=f"{cam}-{tid}-{f0 + i}",
                vector=vec.tolist(),
                payload={"camera": cam, "track_id": tid, "frame": f0 + i,
                         "run_id": "r1", "reid_id": gid,
                         "ts": float(ts0) + i * ts_step},
            ))
    return out


def _run(specs, **kw):
    store = _FakeStore(_observations(specs))
    kw.setdefault("same_camera_threshold", SAME_BAR)
    kw.setdefault("min_tracklet_observations", 1)
    lines = []
    remap = reconcile_tracklets(store, threshold=THRESHOLD, run_id="r1",
                               require_reciprocal_best=True,
                               log=lines.append, **kw)
    return len(set(remap.values())), remap, lines


def main():
    c = Check("physical guards: Phase 1 mutual-best + cross-camera simultaneity")

    # ---------------------------------------------------------------- 1
    print("\n1. PHASE 1 MUTUAL-BEST -- the cam_224 reid-6 failure, reproduced")

    P = _unit(1.0)                          # person A, first sighting
    A2 = _unit(0.94, 0.341)                 # A again: 0.94 to P -> their mutual best
    # Person B on a THIRD axis: 0.82 against P (above the bar, so today it merges)
    # but only 0.77 against A2 (below it). Built in a separate dimension on purpose
    # -- an earlier version put A2 and B in the same plane, which made them 0.966
    # similar and therefore each OTHER's best partner, so the fixture tested
    # something else entirely and the check failed for the wrong reason.
    B = _unit(0.82, 0.0, 0.572)
    # Time-disjoint so nothing but appearance decides -- exactly the real case.
    scene = [("cam_224", 16, P, 0, 16, 0.0),
             ("cam_224", 58, A2, 100, 58, 10.0),
             ("cam_224", 112, B, 200, 112, 20.0)]

    n, remap, _ = _run(scene)
    c.eq(n, 1, "WITHOUT mutual-best: the 0.82 stranger is absorbed too "
               "(one reid on two people)")

    n, remap, lines = _run(scene, same_camera_reciprocal_best=True)
    c.eq(n, 2, "WITH mutual-best: only the genuine 0.94 pair merges, at the SAME "
               "bar")
    c.ok(remap[("cam_224", 16)] == remap[("cam_224", 58)],
         "and the real pair is still merged (not just merging less)")
    c.ok(any("refused for not being each other's best" in m for m in lines),
         "the refusal is reported, so it is visible in a run log")

    # It must not block a clean two-tracklet fragment, which IS mutual by
    # construction -- otherwise it would trade over-merging for under-merging.
    n, _, _ = _run([("cam_213", 1, P, 0, 1, 0.0),
                    ("cam_213", 2, _at(0.95), 100, 2, 10.0)],
                   same_camera_reciprocal_best=True)
    c.eq(n, 1, "a plain two-piece fragment still merges (it is mutual by "
               "construction)")

    # ---------------------------------------------------------------- 2
    print("\n2. CROSS-CAMERA SIMULTANEITY -- impossible merges, any cosine")

    # IDENTICAL appearance, so only physics can separate them. Overlapping in
    # wall-clock time, in two cameras that cannot both see one person.
    twins = [("cam_213", 1, P, 0, 1, 100.0),
             ("cam_219", 2, P, 0, 2, 100.0)]     # both span 100-105s: 5s overlap
    n, _, _ = _run(twins)
    c.eq(n, 1, "veto OFF: two simultaneous people in non-co-visible cameras merge "
               "at cosine 1.0 (the defect)")

    covis = resolve_covisibility({"covisibility": {
        "enabled": True,
        "pairs": [["cam_224", "cam_219", "covisible"], ["cam_213", "cam_219", 1.0]],
    }})
    n, _, lines = _run(twins, covisibility=covis)
    c.eq(n, 2, "veto ON: they stay separate, appearance notwithstanding")
    c.ok(any("simultaneity veto ON" in m for m in lines), "and it announces itself")

    # Co-visible pair: simultaneity proves NOTHING there, so it must still merge.
    both_in_room = [("cam_224", 1, P, 0, 1, 100.0),
                    ("cam_219", 2, P, 0, 2, 100.0)]
    n, _, _ = _run(both_in_room, covisibility=covis)
    c.eq(n, 1, "a CO-VISIBLE pair still merges while overlapping (same room)")

    # Sequential, not simultaneous: the normal cross-camera walk must survive.
    walk = [("cam_213", 1, P, 0, 1, 100.0),
            ("cam_219", 2, P, 0, 2, 130.0)]      # 30s later
    n, _, _ = _run(walk, covisibility=covis)
    c.eq(n, 1, "walking 213 -> 219 thirty seconds later still merges")

    # Overlap under the tolerance is timing noise, not two people.
    graze = [("cam_213", 1, P, 0, 1, 100.0),
             ("cam_219", 2, P, 0, 2, 104.5)]     # 0.5s overlap, tol 1.0s
    n, _, _ = _run(graze, covisibility=covis)
    c.eq(n, 1, "a 0.5s overlap is within the 1.0s tolerance -> no veto "
               "(ts is receive time, so small overlaps are jitter)")

    # An UNLISTED pair is unconstrained by design (D1) and says so.
    n, _, lines = _run([("cam_213", 1, P, 0, 1, 100.0),
                        ("cam_206", 2, P, 0, 2, 100.0)],
                       covisibility=covis)
    c.eq(n, 1, "an unlisted pair is NOT vetoed (fail-open: silence must not split "
               "a real person)")
    c.ok(any("NO covisibility entry" in m for m in lines),
         "but every unlisted pair is named at startup, so it is not silent")

    # ---------------------------------------------------------------- 3
    print("\n3. NO WALL CLOCK -> NO VETO (cannot judge, so do not)")

    store = _FakeStore([SimpleNamespace(
        id=f"{cam}-{tid}-{i}", vector=P.tolist(),
        payload={"camera": cam, "track_id": tid, "frame": i, "run_id": "r1",
                 "reid_id": tid})                      # no "ts" at all
        for cam, tid in (("cam_213", 1), ("cam_219", 2)) for i in range(6)])
    remap = reconcile_tracklets(store, threshold=THRESHOLD, run_id="r1",
                               same_camera_threshold=SAME_BAR,
                               min_tracklet_observations=1,
                               covisibility=covis, log=lambda *_a, **_k: None)
    c.eq(len(set(remap.values())), 1,
         "tracklets with no ts are never vetoed (old runs stay reproducible)")

    c.eq(temporal_overlap_sec(None, (1.0, 2.0)), None,
         "temporal_overlap_sec(None, ...) is None, not 0.0 -- 'unknown' and "
         "'no overlap' must not be confused")
    c.eq(temporal_overlap_sec((0.0, 5.0), (4.0, 9.0)), 1.0, "overlap arithmetic")
    c.ok(temporal_overlap_sec((0.0, 1.0), (5.0, 6.0)) < 0,
         "disjoint spans give a negative overlap (never > tolerance)")

    # ---------------------------------------------------------------- 4
    print("\n4. BOTH GUARDS ARE OFF BY DEFAULT")

    base = _run(scene)
    c.ok(base == _run(scene, covisibility=None,
                      same_camera_reciprocal_best=False),
         "explicitly-off matches not-passed-at-all")
    c.ok(_run(twins)[0] == 1,
         "and with no covisibility config the cross-camera veto never fires")
    c.eq(resolve_covisibility({}), (False, {}),
         "no covisibility block -> disabled, no pairs")

    c.done()


if __name__ == "__main__":
    main()
