"""
Offline reconcile CONSUMES recorded geometry. It never recomputes it.

This file exists because that rule is easy to break by accident and impossible to
notice afterwards. The tempting change is one line: reconcile already has each
observation's `bbox`, so why not load the calibration and derive the position on the
spot? Two reasons, both of which produce silent wrongness rather than an error:

  1. THE LIVE FEED IS NEVER RECORDED. A position not written during the run is gone.
     Deriving it later only appears to work while the calibration happens to be the
     one that was in force.

  2. TWO RECONCILES OF ONE RUN WOULD DISAGREE. Re-fit a calibration -- which the
     self-fitting tool makes cheap and therefore likely -- and re-reconciling a
     finished run silently returns different identities, with nothing in either
     output saying why. The whole value of the offline pass is that it is the
     authority on final ids; an authority that changes its mind for invisible
     reasons is not one.

So the rule is mechanical, not aspirational: reconcile may import
`geometry.reachability` (pure arithmetic over positions it was handed) and must not
be able to reach `geometry.floor` or `geometry.calibration`, which are the modules
that could turn a box into a position. `geometry/__init__.py` deliberately imports
nothing, so importing the arithmetic does not drag the machinery in with it.

Checks 1-2 are the invariant. Checks 3-6 are the guard's behaviour on synthetic
tracklets, including the two ways it must NOT fire.
"""

import sys

import numpy as np
from types import SimpleNamespace

from _synth import Check, DIM

THRESHOLD = 0.63
SAME_BAR = 0.80


def _unit(*coeffs):
    v = np.zeros(DIM, np.float32)
    for i, c in enumerate(coeffs):
        v[i] = c
    return v / np.linalg.norm(v)


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


# Per-observation position uncertainty. In production this is the calibration's own
# held-out reprojection p95 -- i.e. how far apart the SAME person's two co-visible
# positions actually land. That is load-bearing: if the error radius were smaller
# than the real cross-camera disagreement, the same-instant rule would veto every
# true co-visible match. tools/fit_floor_frame.py derives it from exactly that
# measurement, so the two cannot drift apart.
POS_ERROR = 0.25

# Observations per tracklet. Generous on purpose: the speed ceiling is measured from
# consecutive same-track pairs and needs at least 30 of them per floor group, so a
# skimpy fixture would leave the envelope unmeasurable and every "does not fire"
# check below would pass for the wrong reason.
N_OBS = 40


def _obs(cam, tid, vec, gid, ts0, positions, ts_step=0.5, group="room",
         error=POS_ERROR):
    """One tracklet. `positions` is [(x, y)] -- one per observation, or None.

    None models an observation with NO recorded floor position, which is what every
    run captured before geometry existed looks like.
    """
    out = []
    for i, xy in enumerate(positions):
        payload = {"camera": cam, "track_id": tid, "frame": i, "run_id": "r1",
                   "reid_id": gid, "ts": float(ts0) + i * ts_step}
        if xy is not None:
            payload["floor"] = {"x": float(xy[0]), "y": float(xy[1]),
                                "error": error, "group": group,
                                "source": "box_bottom_centre", "clipped": False,
                                "calib_version": "test"}
        out.append(SimpleNamespace(id=f"{cam}-{tid}-{i}", vector=vec.tolist(),
                                   payload=payload))
    return out


def _run(observations, geometry=None, **kw):
    from identity.reconcile import reconcile_tracklets
    store = _FakeStore(observations)
    kw.setdefault("same_camera_threshold", SAME_BAR)
    kw.setdefault("min_tracklet_observations", 1)
    lines = []
    remap = reconcile_tracklets(store, threshold=THRESHOLD, run_id="r1",
                                require_reciprocal_best=True, geometry=geometry,
                                log=lines.append, **kw)
    return len(set(remap.values())), remap, lines


def walk(n, x0, y0, dx, dy):
    """A person walking in a straight line -- n positions, dx/dy per observation."""
    return [(x0 + dx * i, y0 + dy * i) for i in range(n)]


def main():
    c = Check("geometry: consumed by reconcile, never recomputed")

    # ------------------------------------------------------------------ 1
    print("\n1. THE INVARIANT -- reconcile cannot reach the position machinery")

    for mod in ("geometry", "geometry.floor", "geometry.calibration",
                "identity.reconcile"):
        sys.modules.pop(mod, None)
    import identity.reconcile                                       # noqa: F401

    c.ok("geometry.reachability" in sys.modules,
         "importing reconcile DOES pull in geometry.reachability (the arithmetic)")
    c.ok("geometry.floor" not in sys.modules,
         "importing reconcile does NOT pull in geometry.floor -- the bbox-to-position "
         "code is out of reach")
    c.ok("geometry.calibration" not in sys.modules,
         "importing reconcile does NOT pull in geometry.calibration -- no homography, "
         "no calibration file")

    # Check the IMPORTS, via the AST -- not the source text. The text mentions the
    # forbidden modules in the comments that explain why they are forbidden, so a
    # substring search would fail on its own documentation.
    import ast
    tree = ast.parse(open(identity.reconcile.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for banned in ("geometry.floor", "geometry.calibration", "cv2", "geometry"):
        c.ok(banned not in imported,
             f"reconcile.py does not import {banned!r}")
    c.ok("geometry.reachability" in imported,
         "reconcile.py imports geometry.reachability and nothing else from geometry")

    # ------------------------------------------------------------------ 2
    print("\n2. RECORDED POSITIONS ARE READ FROM THE PAYLOAD, NOT DERIVED")

    from geometry.reachability import RecordedPosition
    p = RecordedPosition.from_payload({
        "ts": 100.0,
        "floor": {"x": 1.5, "y": -2.0, "error": 0.1, "group": "room"}})
    c.ok(p is not None and p.x == 1.5 and p.y == -2.0 and p.ts == 100.0,
         "a payload's floor block reads back exactly")
    c.ok(RecordedPosition.from_payload({"ts": 1.0, "bbox": [0, 0, 10, 20]}) is None,
         "a payload with a BBOX but no floor block gives None -- there is no path "
         "from a box to a position here")
    c.ok(RecordedPosition.from_payload({"floor": {"x": 1, "y": 2, "group": "r"}})
         is not None, "a floor block with no ts still parses (ts None fails open "
                      "downstream)")

    # ------------------------------------------------------------------ 3
    print("\n3. THE VETO FIRES -- two people the appearance model wants to merge")

    # Two tracklets in the co-visible pair that look nearly identical (cosine 0.99,
    # far above every bar) but stand 20 units apart throughout their overlap. This
    # is the uniformed-look-alike case that defeats appearance entirely, and the one
    # no threshold reaches: the cosine is not wrong, it is irrelevant.
    look_alike_a = _unit(1.0)
    look_alike_b = _unit(0.99, 0.141)
    scene = (_obs("cam_219", 1, look_alike_a, 1, 100.0,
                  walk(N_OBS, 0, 0, 0.2, 0)) +
             _obs("cam_224", 2, look_alike_b, 2, 100.0,
                  walk(N_OBS, 20, 0, 0.2, 0)))

    n_off, remap_off, _ = _run(scene, geometry={"enabled": False})
    c.eq(n_off, 1, "geometry OFF: the look-alikes merge into one reid (today's "
                   "behaviour)")

    n_on, remap_on, lines = _run(
        scene, geometry={"enabled": True, "clock_error_sec": 0.5,
                         "safety_factor": 1.5})
    c.eq(n_on, 2, "geometry ON: the merge is refused -- 20 units apart at the same "
                  "instant is not one person")
    c.ok(any("vetoed as physically unreachable" in m for m in lines),
         "and the veto is reported in the run log")
    c.ok(any("units/s" in m and "group room" in m for m in lines),
         "with the measured ceiling named, so the envelope can be judged")

    # ------------------------------------------------------------------ 4
    print("\n4. IT DOES NOT FIRE ON A REAL PERSON -- the topology failure, checked")

    # The disabled live.topology veto pruned the TRUE cross-camera match because it
    # assumed 2-3 seconds of transit between cameras that are actually adjacent.
    # Here one person is genuinely seen by both co-visible cameras at the same
    # instant, standing in the same place (0.3 units apart, inside the error budget).
    # Merging must still happen.
    same_a = _unit(1.0)
    same_b = _unit(0.97, 0.243)              # 0.97 -- a real cross-camera match
    one_person = (_obs("cam_219", 3, same_a, 3, 200.0,
                       walk(N_OBS, 5, 5, 0.15, 0)) +
                  _obs("cam_224", 4, same_b, 4, 200.0,
                       walk(N_OBS, 5.3, 5, 0.15, 0)))
    n, remap, lines = _run(one_person, geometry={"enabled": True,
                                                "clock_error_sec": 0.5,
                                                "safety_factor": 1.5})
    c.eq(n, 1, "one person seen by both co-visible cameras at once still merges -- "
               "overlapping cameras give a distance near zero, so the check is silent")

    # And a person who walks out of one camera and into the other a second later,
    # covering ground at a normal pace, must not be vetoed either.
    hand_a = _unit(1.0)
    hand_b = _unit(0.96, 0.28)
    # A walks at 0.25 units per 0.5 s = 0.5 units/s, ending at x=9.75, t=319.5.
    # Three seconds later they appear in cam_224 1.5 units further on -- exactly
    # their own observed walking pace, so this must pass.
    handover = (_obs("cam_219", 5, hand_a, 5, 300.0,
                     walk(N_OBS, 0, 0, 0.25, 0)) +
                _obs("cam_224", 6, hand_b, 6, 322.5,
                     walk(N_OBS, 11.25, 0, 0.25, 0)))
    n, _remap, _lines = _run(handover, geometry={"enabled": True,
                                                "clock_error_sec": 0.5,
                                                "safety_factor": 1.5})
    c.eq(n, 1, "a normal-paced walk from one camera to the other is NOT vetoed")

    # ------------------------------------------------------------------ 5
    print("\n5. NO RECORDED GEOMETRY -> NO OPINION, AND IT SAYS SO")

    bare = (_obs("cam_219", 7, look_alike_a, 7, 400.0, [None] * N_OBS) +
            _obs("cam_224", 8, look_alike_b, 8, 400.0, [None] * N_OBS))
    n, _remap, lines = _run(bare, geometry={"enabled": True,
                                            "clock_error_sec": 0.5,
                                            "safety_factor": 1.5})
    c.eq(n, 1, "a run captured WITHOUT geometry reconciles exactly as before")
    c.ok(any("NOT ONE observation" in m for m in lines),
         "and reconcile says loudly that the enabled veto can never fire")

    # Half a run positioned: the unpositioned tracklets must simply be exempt.
    mixed = (_obs("cam_219", 9, look_alike_a, 9, 500.0,
                  walk(N_OBS, 0, 0, 0.2, 0)) +
             _obs("cam_224", 10, look_alike_b, 10, 500.0, [None] * N_OBS))
    n, _remap, lines = _run(mixed, geometry={"enabled": True,
                                             "clock_error_sec": 0.5,
                                             "safety_factor": 1.5})
    c.eq(n, 1, "a pair where only ONE side has positions is never vetoed")

    # ------------------------------------------------------------------ 6
    print("\n6. DIFFERENT FLOOR GROUPS ARE UNKNOWN, NOT FAR APART")

    # cam_213 shares no floor frame with the co-visible room, so its coordinates are
    # not comparable. A naive implementation would read the raw numbers, find them
    # 500 units apart, and veto every cross-room merge in the deployment.
    far_a = _unit(1.0)
    far_b = _unit(0.97, 0.243)
    cross_group = (
        _obs("cam_219", 11, far_a, 11, 600.0, walk(N_OBS, 0, 0, 0.2, 0),
             group="room") +
        _obs("cam_213", 12, far_b, 12, 600.0, walk(N_OBS, 500, 500, 0.2, 0),
             group="corridor"))
    n, _remap, _lines = _run(cross_group, geometry={"enabled": True,
                                                   "clock_error_sec": 0.5,
                                                   "safety_factor": 1.5})
    c.eq(n, 1, "coordinates from two different floor groups are NEVER compared -- "
               "500 units apart in incomparable frames is not evidence")

    # ------------------------------------------------------------------ 7
    print("\n7. THE POLICY DEFAULT IS OFF, AND OFF CHANGES NOTHING")

    n_none, remap_none, _ = _run(scene, geometry=None)
    c.eq((n_none, remap_none), (n_off, remap_off),
         "geometry=None is byte-identical to geometry={'enabled': False}")

    from identity.reconcile import resolve_geometry_policy
    pol = resolve_geometry_policy({})
    c.ok(pol["enabled"] is False,
         "resolve_geometry_policy defaults the veto OFF on an empty config")
    pol = resolve_geometry_policy({"geometry": {"enabled": True}})
    c.ok(pol["enabled"] is False,
         "geometry.enabled (RECORDING) does not enable the veto -- "
         "geometry.reconcile.enabled does, and only that")

    c.done()


if __name__ == "__main__":
    sys.exit(main() or 0)
