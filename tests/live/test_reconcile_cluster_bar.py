"""
test_reconcile_cluster_bar.py  --  PHASE 2's same-camera bar, separated from PHASE 1's.

WHAT THIS PINS, and why it earns its own file.

`identity.reconcile.same_camera_threshold` used to serve BOTH phases, and the two are
different claims on different evidence: Phase 1 compares two SINGLE tracklets, Phase 2
compares two CLUSTERS that are each already assembled, denoised over more
observations, and past their own Phase-1 and reciprocal-best checks. Conflating them
made the operator's reported split unfixable by any single number -- measured on run
20260804_064551, the two clusters holding one person score 0.840 against cam_219's
0.90 bar, and lowering that bar to 0.80 lets the join through but simultaneously
admits three unverified Phase-1 pairs in cam_219. The re-render confirmed exactly
those three Phase-1 merges and no cluster join.

Three properties, all load-bearing:

  1. DEFAULT IS UNCHANGED. cluster_same_camera_threshold=None must reproduce the
     shipped result. This is what makes the change safe to ship OFF, and it is the
     first thing that breaks if someone later "simplifies" cluster_cam_bar.
  2. IT OPENS THE PHASE-2 JOIN that Phase 1's bar was blocking.
  3. IT LEAVES PHASE 1 ALONE -- otherwise this is an ordinary threshold cut with
     extra steps, which six reverted changes already established does not work.

Plus the cap: the Phase-2 bar is never allowed ABOVE Phase 1's, because a claim with
more evidence behind it must not face a stricter standard.

Synthetic vectors, no GPU, no video, no Qdrant.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from identity.reconcile import (_gather_tracklets, _prototype,  # noqa: E402
                                reconcile_tracklets,
                                resolve_reconcile_kwargs)
from _synth import Check  # noqa: E402

DIM = 32
CROSS = 0.30              # low, so the cross-camera lane links freely
PHASE1 = 0.90             # strict, as cam_219 ships


class _FakeStore:
    """Minimal stand-in: reconcile needs client.scroll, collection, set_global_id
    and clear_global_id. Same shape as test_stage6_reconcile_camera_bar.py's."""

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


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _basis(seed):
    rng = np.random.default_rng(seed)
    return _unit(rng.normal(size=DIM))


def _at_cosine(a, target, seed):
    """A unit vector at approximately `target` cosine from `a`."""
    b = _basis(seed)
    b = _unit(b - (b @ a) * a)                 # component orthogonal to a
    return _unit(target * a + np.sqrt(max(0.0, 1 - target ** 2)) * b)


def _observations(specs, n=6):
    """specs: [(camera, track_id, vector, first_frame)] -> flat point list."""
    out = []
    for cam, tid, vec, f0 in specs:
        for f in range(f0, f0 + n):
            out.append(SimpleNamespace(
                id=f"{cam}-{tid}-{f}",
                vector=np.asarray(vec, dtype=np.float32).tolist(),
                payload={"camera": cam, "track_id": tid, "frame": f,
                         "run_id": "r1", "ts": float(f) * 0.04}))
    return out


def _fixture():
    """One person split across two clusters that SHARE cam_A.

    cam_A/1 and cam_A/2 are disjoint-in-time fragments of the person at a cosine
    deliberately BELOW Phase 1's 0.90. Each links cross-camera to its own cam_B
    tracklet, so both clusters end up containing cam_A -- which is the geometry that
    makes strictest_same_camera_bar govern the join, exactly as on the real run.
    """
    p = _basis(1)
    f1 = p
    f2 = _at_cosine(p, 0.84, seed=7)           # the measured 0.840, near enough
    return [("cam_A", 1, f1, 0), ("cam_A", 2, f2, 200),
            ("cam_B", 10, f1, 0), ("cam_B", 11, f2, 200)]


def _run(cluster_bar):
    specs = _fixture()
    store = _FakeStore(_observations(specs))
    lines = []
    remap = reconcile_tracklets(
        store, threshold=CROSS, run_id="r1",
        same_camera_threshold=PHASE1,
        require_reciprocal_best=True, same_camera_reciprocal_best=True,
        min_tracklet_observations=1,
        cluster_same_camera_threshold=cluster_bar,
        log=lines.append)
    return remap, lines


def main():
    c = Check("reconcile: Phase 2's same-camera bar is separable from Phase 1's")

    # ---- fixture sanity: the fragment pair must sit below Phase 1's bar
    store = _FakeStore(_observations(_fixture()))
    tk = _gather_tracklets(store, "r1")
    protos = {k: _prototype(v["vectors"]) for k, v in tk.items()}
    frag = float(protos[("cam_A", 1)] @ protos[("cam_A", 2)])
    c.ok(frag < PHASE1,
         f"fixture: cam_A fragment pair {frag:.3f} is BELOW Phase 1's {PHASE1}")
    c.ok(frag > CROSS,
         f"fixture: cam_A fragment pair {frag:.3f} is above the cross bar {CROSS}")

    # ---- 1. default reproduces the split
    remap_off, lines_off = _run(None)
    off_ids = len(set(remap_off.values()))
    c.ok(remap_off[("cam_A", 1)] != remap_off[("cam_A", 2)],
         f"cluster bar OFF (None): the person stays SPLIT ({off_ids} identities)")
    c.ok(not any("PHASE 2 same-camera bar separated" in l for l in lines_off),
         "cluster bar OFF: nothing logged about phase separation")

    # ---- 2. a cluster bar below the fragment score opens the join
    bar = round(frag - 0.04, 2)
    remap_on, lines_on = _run(bar)
    c.ok(remap_on[("cam_A", 1)] == remap_on[("cam_A", 2)],
         f"cluster bar {bar:.2f}: the two clusters MERGE "
         f"({len(set(remap_on.values()))} identities)")
    c.ok(any("PHASE 2 same-camera bar separated" in l for l in lines_on),
         "cluster bar ON: the separation is logged")
    c.ok(any("cluster merge" in l for l in lines_on),
         "the merge happened in PHASE 2 (a cluster merge line exists)")

    # ---- 3. Phase 1 untouched
    c.ok(not any("same-camera merge" in l for l in lines_on),
         "PHASE 1 admitted NO same-camera tracklet merge at the strict bar")

    # ---- 4. the cap
    remap_hi, _ = _run(0.99)
    c.ok(len(set(remap_hi.values())) == off_ids,
         f"a cluster bar ABOVE Phase 1's cannot make merging stricter "
         f"({len(set(remap_hi.values()))} vs {off_ids} identities)")

    # ---- 5. the resolver
    kw_none = resolve_reconcile_kwargs({"identity": {"reconcile": {}}})
    kw_set = resolve_reconcile_kwargs(
        {"identity": {"reconcile": {"cluster_same_camera_threshold": 0.8}}})
    c.ok(kw_none["cluster_same_camera_threshold"] is None,
         "resolver: absent key -> None")
    c.ok(kw_set["cluster_same_camera_threshold"] == 0.8,
         "resolver: present key -> float 0.8")

    c.done()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
