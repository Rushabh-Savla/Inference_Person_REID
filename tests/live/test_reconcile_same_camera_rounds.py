"""
Phase 1 in ROUNDS -- the third fragment that mutual-best orphans forever.

THE DEFECT (REMEDIATION_PLAN.md M.9.11). Phase 1 required mutual best-match and ran
exactly ONE pass. A person fragmented into three pieces in one camera has only one
mutual-best pair, so the third piece is refused -- even when its edge clears the
camera's bar comfortably. Phase 1's own comment claimed Phase 2 would consolidate it
later; it cannot. Phase 2's `mergeable_cross` requires a CROSS-camera member pair, so
two clusters that both sit in one camera are excluded outright
(dlog.NOT_MERGEABLE_CROSS) and the promised second chance never comes.

An orphaned fragment is then free to be captured cross-camera into somebody ELSE's
identity, which is what the operator saw in run 20260731_060425:

    206/26 <-> 206/43 = 0.916   merged (each other's best)
    206/12 <-> 206/43 = 0.906   ABOVE cam_206's 0.90 bar, REFUSED
    -> 206/12 orphaned, then absorbed cross-camera at 0.771 into another person

No threshold reaches that: the edge already clears the bar, and lowering cam_206's
bar measurably makes cam_206 worse (25 -> 20 identities). The fix is to let the
fragment compete against the CLUSTER its sibling formed, which is what Phase 2
already does for cross-camera pairs.

WHAT THIS FILE PINS
  1. the orphaning, so the defect cannot silently return
  2. rounds recover the third fragment
  3. rounds relax NOTHING -- a stranger above the bar is still refused, so the
     guard that mutual-best exists for is intact
  4. the flag OFF is byte-identical to the old single pass

FIXTURE CAVEAT, stated because Part L.3 warns about fixtures that do not test what
they claim: this fixture has all three pairwise scores above the bar, so the merged
cluster's prototype stays within reach of the third fragment. That is NOT guaranteed
in general -- `_cluster_prototype` averages tracklet prototypes UNWEIGHTED, so a
4-observation fragment pulls the cluster centre as hard as a 149-observation one, and
it can pull the centre away from the orphan. Rounds are necessary for the real
cam_206 case; whether they are sufficient there is a measurement, not an assumption.
"""

import sys

import numpy as np
from types import SimpleNamespace

from _synth import Check, DIM
from identity.reconcile import reconcile_tracklets

THRESHOLD = 0.63        # cross-camera bar; irrelevant here (one camera only)
SAME_BAR = 0.90         # the bar cam_206 actually runs at


def _at(degrees, plane=(0, 1)):
    """Unit vector at `degrees` from e0, inside one 2-D plane.

    Angles make the pairwise cosines readable: the cosine of two of these is the
    cosine of their angle DIFFERENCE, so a chain is easy to lay out on purpose
    instead of discovered by trial and error.
    """
    v = np.zeros(DIM, np.float32)
    rad = np.deg2rad(degrees)
    v[plane[0]] = np.cos(rad)
    v[plane[1]] = np.sin(rad)
    return v


class _FakeStore:
    collection = "persons"

    def __init__(self, observations):
        self.assigned, self.cleared = {}, []
        self.client = SimpleNamespace(scroll=lambda *a, **k: (observations, None))

    def set_global_id(self, point_ids, global_id):
        for p in point_ids:
            self.assigned[p] = global_id

    def clear_global_id(self, point_ids):
        self.cleared.extend(point_ids)


def _observations(specs, per=6):
    """specs: [(cam, track_id, vec, first_frame, gid, ts0)] -> store points.

    Spans are made disjoint by the caller's frame/ts offsets; a 6-observation
    tracklet at 1.0s steps spans 5 seconds, long enough that a co-presence veto
    would actually fire if the spans overlapped (the trap in L.3).
    """
    out = []
    for cam, tid, vec, f0, gid, ts0 in specs:
        for i in range(per):
            out.append(SimpleNamespace(
                id=f"{cam}-{tid}-{f0 + i}",
                vector=vec.tolist(),
                payload={"camera": cam, "track_id": tid, "frame": f0 + i,
                         "run_id": "r1", "reid_id": gid,
                         "ts": float(ts0) + i * 1.0}))
    return out


def _run(specs, **kw):
    store = _FakeStore(_observations(specs))
    kw.setdefault("same_camera_threshold", SAME_BAR)
    kw.setdefault("min_tracklet_observations", 1)
    kw.setdefault("same_camera_reciprocal_best", True)
    lines = []
    remap = reconcile_tracklets(store, threshold=THRESHOLD, run_id="r1",
                               require_reciprocal_best=True,
                               log=lines.append, **kw)
    return remap, lines


def main():
    c = Check("Phase 1 rounds: the orphaned third fragment (M.9.11)")

    # ---- the chain: one person, three fragments, every edge above the 0.90 bar.
    #   A(0 deg) <-> B(12 deg) = cos 12 = 0.978
    #   B(12)    <-> C(25)     = cos 13 = 0.974
    #   A(0)     <-> C(25)     = cos 25 = 0.906   <- above the bar, and refused
    # B's best partner is A, A's best is B, so A+B merge and C is left out even
    # though C's own best (B, 0.974) is far above the bar.
    A, B, C = _at(0.0), _at(12.0), _at(25.0)
    chain = [("cam_206", 43, A, 0, 43, 0.0),
             ("cam_206", 26, B, 100, 26, 20.0),
             ("cam_206", 12, C, 200, 12, 40.0)]

    # Assert the fixture's own preconditions before trusting either result (L.3).
    c.ok(abs(float(A @ B) - 0.978) < 0.002, f"fixture: A.B = {float(A @ B):.3f}")
    c.ok(abs(float(B @ C) - 0.974) < 0.002, f"fixture: B.C = {float(B @ C):.3f}")
    c.ok(abs(float(A @ C) - 0.906) < 0.002, f"fixture: A.C = {float(A @ C):.3f}")
    c.ok(min(float(A @ B), float(B @ C), float(A @ C)) > SAME_BAR,
         "fixture: EVERY edge clears the same-camera bar, so refusing any of them "
         "is a defect and not the bar doing its job")

    # ---------------------------------------------------------------- 1
    print("\n1. THE DEFECT: one pass orphans the third fragment")
    remap, lines = _run(chain, same_camera_rounds=False)
    ids = set(remap.values())
    c.eq(len(ids), 2, "single pass leaves TWO identities for one person")
    c.ok(remap[("cam_206", 12)] != remap[("cam_206", 43)],
         "and 206/12 is the orphan -- filed apart from its own siblings")
    c.eq(sum(1 for ln in lines if "same-camera merge" in ln), 1,
         "exactly one merge happened")
    c.ok(any("refused for not being each other's best" in ln for ln in lines),
         "the log names mutual-best as the reason")

    # ---------------------------------------------------------------- 2
    print("\n2. THE FIX: rounds recover it")
    remap_r, lines_r = _run(chain, same_camera_rounds=True)
    c.eq(len(set(remap_r.values())), 1,
         "with rounds, all three fragments are ONE identity")
    c.eq(len(set(remap_r)), 3, "and no tracklet was dropped to get there")
    c.ok(any("[round 2]" in ln for ln in lines_r),
         "the second round is visible in the log, so this is auditable")

    # ---------------------------------------------------------------- 3
    print("\n3. ROUNDS RELAX NOTHING: a stranger above the bar is still refused")
    # The physical-guards fixture's shape: a third person who scores above the bar
    # against ONE fragment but is nobody's mutual best. Rounds must not let them in,
    # or this change trades one failure for the worse one.
    #   A(0) <-> B(12) = 0.978 (mutual best, merge)
    #   S: 0.91 against A, and far from B -- placed on a THIRD axis so it cannot
    #      accidentally become B's best partner (the mistake L.3 records).
    stranger = np.zeros(DIM, np.float32)
    stranger[0], stranger[2] = 0.91, float(np.sqrt(1.0 - 0.91 ** 2))
    scene = [("cam_206", 43, A, 0, 43, 0.0),
             ("cam_206", 26, B, 100, 26, 20.0),
             ("cam_206", 99, stranger, 200, 99, 40.0)]
    c.ok(float(stranger @ A) > SAME_BAR,
         f"fixture: the stranger IS above the bar against A "
         f"({float(stranger @ A):.3f}), so this tests the guard and not the bar")
    c.ok(float(stranger @ B) < SAME_BAR,
         f"fixture: and below it against B ({float(stranger @ B):.3f})")
    remap_s, _ = _run(scene, same_camera_rounds=True)
    c.eq(len(set(remap_s.values())), 2,
         "the stranger stays OUT with rounds on (guard intact)")
    c.ok(remap_s[("cam_206", 99)] != remap_s[("cam_206", 43)],
         "and specifically is not filed with the person it half-matched")

    # ---------------------------------------------------------------- 4
    print("\n4. FLAG OFF is the old behaviour, on a scene rounds would change")
    off, _ = _run(chain, same_camera_rounds=False)
    c.eq(off, remap, "same result as the first run -- deterministic")
    c.ok(off != remap_r,
         "and genuinely different from rounds ON, so the flag is doing something "
         "(a test where both sides agree would prove nothing)")

    c.done()


if __name__ == "__main__":
    main()
