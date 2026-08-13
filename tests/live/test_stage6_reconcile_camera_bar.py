"""
Stage 6 -- offline reconcile: the SAME-CAMERA bar must survive Phase 2.

Regression guard for a real defect. `reconcile_tracklets` runs two phases:

  * Phase 1 repairs same-camera track fragmentation at the STRICT
    `same_camera_threshold` (0.90). "These two tracks in one camera are the same
    person" is an easy claim to get wrong -- same lighting, same pose
    distribution, so unrelated people score unusually high.
  * Phase 2 consolidates CLUSTERS at the LOW cross-camera `threshold` (0.63).

The bug: Phase 2's gate (`mergeable_cross`) only asked whether *some* member pair
across the two clusters was cross-camera. So the moment a cluster had absorbed a
second camera, that pre-existing member satisfied the gate, and a SAME-camera
fragment could then join at the cross-camera bar -- throwing away Phase 1's strict
threshold entirely. Two strangers seen in one camera at different times merged at
cosine ~0.69 on real footage.

Fix: `pair_threshold()` -- clusters that SHARE a camera must clear
`same_camera_threshold`; only camera-disjoint clusters get the low bar.

Note the pre-existing test_stage6_offline_reconcile.py passes both BEFORE and
AFTER the fix -- it never exercised this path. Hence this file.
"""

import sys

import numpy as np
from types import SimpleNamespace

from _synth import Check, DIM
from identity.reconcile import reconcile_tracklets

THRESHOLD = 0.63              # cross-camera bar (identity.reconcile.threshold)
SAME_CAM_THRESHOLD = 0.90     # strict same-camera bar


def _unit(*coeffs):
    """A unit vector with the given leading coefficients (rest zero), so the
    cosine between two such vectors is exactly controllable."""
    v = np.zeros(DIM, np.float32)
    for i, c in enumerate(coeffs):
        v[i] = c
    return v / np.linalg.norm(v)


def _at_cosine(c):
    """A unit vector whose cosine with _unit(1.0) is exactly `c`."""
    return _unit(c, float(np.sqrt(1.0 - c * c)))


class _FakeStore:
    """Minimal stand-in for PersonVectorStore: reconcile only needs `client.scroll`,
    `collection`, `set_global_id` and `clear_global_id`."""

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
    """specs: [(camera, track_id, vector, first_frame, original_gid)] -> point list."""
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


def _identities(specs):
    """Run reconcile over `specs`; return (n_identities, remap)."""
    store = _FakeStore(_observations(specs))
    remap = reconcile_tracklets(
        store, threshold=THRESHOLD, run_id="r1",
        same_camera_threshold=SAME_CAM_THRESHOLD,
        require_reciprocal_best=True, min_tracklet_observations=1,
        log=lambda *_a, **_k: None)      # quiet: the harness prints the verdict
    return len(set(remap.values())), remap


def main():
    c = Check("Stage6 reconcile: same-camera bar survives Phase 2 cluster merging")

    P = _unit(1.0)                 # person P
    look_alike = _at_cosine(0.70)  # a STRANGER at cosine 0.70 -- below 0.90, above 0.63
    same_person = _at_cosine(0.95) # P again in another pose -- clears 0.90

    # 1) THE BUG. cam_A/1 and cam_B/2 are one person (they link cross-camera, which
    #    makes the cluster span two cameras). cam_A/3 is a DIFFERENT person who
    #    merely looks similar (0.70), time-disjoint from cam_A/1 so the hard
    #    conflict guard does not protect us. It must NOT be absorbed: judged against
    #    cam_A it is a same-camera claim and 0.70 < 0.90.
    n, _ = _identities([("cam_A", 1, P, 0, 1),
                        ("cam_A", 3, look_alike, 100, 3),
                        ("cam_B", 2, P, 0, 2)])
    c.eq(n, 2, "same-camera look-alike at 0.70 is NOT absorbed by a cross-camera cluster")

    # 2) The fix must not over-block: a GENUINE same-camera fragment at 0.95 still
    #    consolidates into a cluster that already spans cameras. Without this, the
    #    fix would trade a false-merge bug for a fragmentation bug.
    n, _ = _identities([("cam_A", 1, P, 0, 1),
                        ("cam_A", 3, same_person, 100, 3),
                        ("cam_B", 2, P, 0, 2)])
    c.eq(n, 1, "genuine same-camera fragment at 0.95 STILL merges (no over-blocking)")

    # 3) Camera-disjoint clusters keep the LOW cross-camera bar. 0.70 clears 0.63,
    #    and nothing here makes a same-camera claim, so this must still link --
    #    otherwise cross-camera reconciliation would have been broken outright.
    n, _ = _identities([("cam_A", 1, P, 0, 1),
                        ("cam_B", 2, look_alike, 0, 2)])
    c.eq(n, 1, "camera-disjoint clusters still link at the cross-camera bar (0.70)")

    # 4) The hard same-camera time-overlap guard is untouched by any of this: two
    #    OVERLAPPING tracklets in one camera can never merge, whatever they score.
    n, _ = _identities([("cam_A", 1, P, 0, 1),
                        ("cam_A", 2, P, 0, 2)])       # identical AND overlapping
    c.eq(n, 2, "time-overlapping same-camera tracklets never merge (hard guard intact)")

    c.done()


if __name__ == "__main__":
    main()
