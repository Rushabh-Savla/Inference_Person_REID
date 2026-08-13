#!/usr/bin/env python3
"""
test_same_camera_chain.py -- synthetic, deterministic, seconds, CPU.
No store, no model, no footage, no Qdrant, no labels.

    PYTHONPATH=src python tests/live/test_same_camera_chain.py

WHAT IT SETTLES

reconcile.py documents two same-camera failures and one flag that trades between
them, and the trade has never been exhibited:

  * cam_224: greedy above-bar merging fused two people
    (58+112 at 0.823, 58+120 at 0.872 while 58's best was 16 at 0.937).
    -> `same_camera_reciprocal_best` was added to stop this.

  * cam_206: mutual-best REFUSED an above-bar edge and stranded a fragment
    (12<->43 = 0.906, above cam_206's 0.90 bar, refused because 43's best was 26).
    -> `same_camera_rounds` was added to stop this.

In config.yaml as of 2026-08-04, `same_camera_reciprocal_best` is **ON** and
`same_camera_rounds` is **OFF** -- so the cam_224 greedy-fuse path is closed and
the cam_206 stranding path is open. The recip=False rows below therefore describe
a configuration that no longer ships; they are kept because they exhibit the other
half of the trade.

!! THIS FIXTURE DOES NOT REPRODUCE THE OPERATOR'S cam_219 SPLIT. !!
That pair fails on SCORE -- explain_merge_failure.py on run 20260804_064551
measured cam_219 tracklets 0020 vs 0008 at prototype 0.574 against cam_219's 0.90
bar -- not on mutual-best and not on linkage. No flag in this grid can touch a
score failure. The chain geometry below is the cam_206 case (an ABOVE-bar edge
refused for not being mutual), which is a different, real, still-open bug.

THE FIXTURE

One camera. Three time-disjoint fragments of ONE person, placed on a great circle
so that adjacent fragments clear a 0.90 bar and the ends do not -- the front/back
chain the scoring-modes comment says the design exists for:

    F1 at   0 deg      F1.F2 = cos 22 = 0.927   (adjacent, clears)
    F2 at  22 deg      F2.F3 = cos 18 = 0.951   (adjacent, clears)
    F3 at  40 deg      F1.F3 = cos 40 = 0.766   (ends, FAILS)

Plus one STRANGER, placed in an orthogonal direction at a fixed angle from F2, so
it clears the bar against F2 only:

    S.F2 = 0.920   S.F1 = 0.853   S.F3 = 0.875

Correct answer: {F1,F2,F3} = one identity, S = a second. Two identities.

WHAT TO READ

  chain=YES stranger=SEPARATE   -> both documented failures avoided
  chain=NO                      -> a real person is split (the cam_206 failure)
  stranger=FUSED                -> two people are one reid (the cam_224 failure)

The grid is expected to show that with `scoring=prototype` and
`member_quorum=1.0` NO setting achieves both -- reciprocal-best off fuses the
stranger, on strands the chain -- because round 2's member-pair tests are a
SUBSET of round 1's, so rounds alone can never admit an edge round 1 rejected on
score. That is a claim this file either demonstrates or refutes. Run it.

`member_quorum` and `consensus_best` only appear if Part C of RECONCILE_PATCHES.md
has been applied; the grid skips whatever your build does not support.
"""
import os
import sys
from collections import defaultdict

import numpy as np

# Resolve src/ from THIS FILE, never from the cwd. tests/run_all.py deliberately
# runs each test with cwd = the test's own directory, and discover() globs
# tests/**/test_*.py, so a cwd-relative path passes standalone from the repo root
# and raises ModuleNotFoundError inside the suite -- taking a green run to red on
# every fresh clone. Same three lines tests/live/_synth.py uses. See CLAUDE.md §7.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from identity import reconcile as R

BAR = 0.90
DIM = 16

# Intra-tracklet spread, in degrees. 0.0 means every observation of a tracklet is
# the SAME vector -- what shipped, and what the first recorded grid was measured on.
#
# !! WITH SPREAD 0 THE SCORING COLUMN OF THE GRID IS UNINFORMATIVE, and that is a
# limitation of this fixture, not a result. `max_exemplar` is
# max(prototype, best single observation pair) and `consensus` is a mean over
# observation pairs, so with zero intra-tracklet variance all three modes return the
# SAME number by construction. A grid reading "identical for max_exemplar and
# consensus" therefore says nothing whatever about those modes -- do not carry
# "scoring mode does not matter" out of it. The mutual-best and linkage conclusions
# are unaffected; those are what this fixture actually tests.
#
# Set this to e.g. 8.0 to give each tracklet a real spread of observations around
# its angle. `max_exemplar` can then find a matching view pair that the means miss,
# which is the entire reason those modes exist, and the scoring column becomes a
# real axis. Expect different numbers throughout; re-read the grid from scratch
# rather than comparing it to a spread=0 one.
SPREAD_DEG = 0.0


# ------------------------------------------------------------------ fake store
class _Point:
    __slots__ = ("id", "vector", "payload")

    def __init__(self, pid, vector, payload):
        self.id = pid
        self.vector = vector
        self.payload = payload


class FakeStore:
    """The minimum surface reconcile_tracklets touches: client.scroll,
    collection, set_global_id, clear_global_id. Deliberately NOT ordered by
    frame -- points come back in id order, exactly as Qdrant returns them, so
    this also exercises the ordering assumption in _gather_tracklets."""

    collection = "fixture"

    def __init__(self, points):
        self._points = list(points)
        self.client = self
        self.assigned = {}
        self.cleared = set()

    def scroll(self, collection, limit=1000, offset=None, scroll_filter=None,
               with_payload=True, with_vectors=True):
        start = int(offset or 0)
        chunk = self._points[start:start + limit]
        nxt = start + limit
        return chunk, (nxt if nxt < len(self._points) else None)

    def set_global_id(self, point_ids, gid):
        for pid in point_ids:
            self.assigned[pid] = gid

    def clear_global_id(self, point_ids):
        for pid in point_ids:
            self.cleared.add(pid)
            self.assigned.pop(pid, None)


# ------------------------------------------------------------------- geometry
def _basis():
    e = np.zeros((3, DIM), dtype=np.float32)
    e[0, 0] = 1.0
    e[1, 1] = 1.0
    e[2, 2] = 1.0
    return e[0], e[1], e[2]


def build_points(spread_deg=SPREAD_DEG):
    e1, e2, e3 = _basis()

    def on_circle(deg):
        r = np.deg2rad(deg)
        v = np.cos(r) * e1 + np.sin(r) * e2
        return (v / np.linalg.norm(v)).astype(np.float32)

    # Stranger direction: fixed cosine to the chain at 22 deg, pushed off the
    # chain's plane so its scores against F1 and F3 fall away by exactly the cosine
    # of their angle to 22 deg.
    c = 0.92
    off = float(np.sqrt(max(1.0 - c * c, 0.0)))

    def stranger_at(deg):
        v = (c * on_circle(deg) + off * e3).astype(np.float32)
        return (v / np.linalg.norm(v)).astype(np.float32)

    n_obs = 4                                   # > min_tracklet_observations=3
    # Symmetric about the base angle, so spread_deg=0 gives every observation the
    # identical vector -- byte-for-byte the original fixture.
    offsets = [(j - (n_obs - 1) / 2.0) * float(spread_deg) for j in range(n_obs)]

    tracks = [
        ("F1", 1,  0.0, 100, on_circle),
        ("F2", 2, 22.0, 200, on_circle),
        ("F3", 3, 40.0, 300, on_circle),
        ("S",  9, 22.0, 400, stranger_at),
    ]

    points, pid = [], 0
    for _name, track_id, base_deg, frame0, maker in tracks:
        for j, d in enumerate(offsets):
            points.append(_Point(pid, maker(base_deg + d), {
                "run_id": "fixture",
                "camera": "cam_fix",
                "track_id": track_id,
                "frame": frame0 + j * 10,
                "ts": 1000.0 + frame0 + j * 10,
                "reid_id": None,
            }))
            pid += 1
    label = {1: "F1", 2: "F2", 3: "F3", 9: "S"}
    return points, label


def report_geometry(points, label):
    per = defaultdict(list)
    for p in points:
        per[p.payload["track_id"]].append(p.vector)
    protos = {t: R._prototype(v) for t, v in per.items()}
    order = [1, 2, 3, 9]
    print("pairwise cosine (bar %.2f):" % BAR)
    print("        " + "".join(f"{label[t]:>9}" for t in order))
    for a in order:
        row = "".join(
            ("    --   " if a == b else f"{float(protos[a] @ protos[b]):9.4f}")
            for b in order)
        print(f"   {label[a]:>4} " + row)
    print()


# ------------------------------------------------------------------- the grid
def run_once(points, label, **kwargs):
    store = FakeStore(points)
    silenced = []
    remap = R.reconcile_tracklets(
        store, threshold=BAR, run_id="fixture",
        same_camera_threshold=BAR,
        min_tracklet_observations=3,
        log=silenced.append,
        **kwargs)
    groups = defaultdict(set)
    for (_cam, track_id), gid in remap.items():
        groups[gid].add(label[track_id])
    return groups, silenced


def supported(name):
    import inspect
    return name in inspect.signature(R.reconcile_tracklets).parameters


def main():
    points, label = build_points()
    print(__doc__.split("THE FIXTURE")[0].strip())
    print()
    report_geometry(points, label)

    modes = [R.PROTOTYPE, R.MAX_EXEMPLAR, R.CONSENSUS]
    if hasattr(R, "CONSENSUS_BEST"):
        modes.append(R.CONSENSUS_BEST)
    quorums = [1.0]
    has_quorum = supported("same_camera_member_quorum")
    if has_quorum:
        quorums.append(0.6)
    else:
        print("NOTE  same_camera_member_quorum not in this build "
              "(RECONCILE_PATCHES.md Part C1 not applied) -- quorum column fixed "
              "at 1.0, i.e. complete linkage.\n")

    print("NOTE  a bar of %.2f is held fixed across scoring modes ONLY so the grid "
          "is\n      readable. Modes have different scales; reconcile.py says so, and "
          "a real\n      bar must be re-derived per mode with the sweep. Do not read "
          "these as\n      tuned results.\n" % BAR)

    header = (f"{'scoring':<16}{'recip':<7}{'rounds':<8}{'quorum':<8}"
              f"{'ids':<5}{'chain':<7}{'stranger':<10}both")
    print(header)
    print("-" * len(header))

    wins = []
    for mode in modes:
        for recip in (False, True):
            for rounds in (False, True):
                for quorum in quorums:
                    kw = dict(scoring=mode,
                              same_camera_reciprocal_best=recip,
                              same_camera_rounds=rounds,
                              require_reciprocal_best=True)
                    if has_quorum:
                        kw["same_camera_member_quorum"] = quorum
                    elif quorum != 1.0:
                        continue
                    groups, _ = run_once(points, label, **kw)
                    chain_together = any({"F1", "F2", "F3"} <= g
                                         for g in groups.values())
                    stranger_alone = any(g == {"S"} for g in groups.values())
                    both = chain_together and stranger_alone
                    if both:
                        wins.append((mode, recip, rounds, quorum))
                    print(f"{mode:<16}{str(recip):<7}{str(rounds):<8}{quorum:<8.2f}"
                          f"{len(groups):<5}"
                          f"{'YES' if chain_together else 'NO':<7}"
                          f"{'SEPARATE' if stranger_alone else 'FUSED':<10}"
                          f"{'  <== both' if both else ''}")

    print()
    print("clusters, for the default configuration (scoring=prototype, both same-"
          "camera flags off):")
    groups, log_lines = run_once(
        points, label, scoring=R.PROTOTYPE,
        same_camera_reciprocal_best=False, same_camera_rounds=False,
        require_reciprocal_best=True)
    for gid, g in sorted(groups.items()):
        print(f"   reid {gid}: {sorted(g)}")
    print("\n   reconcile log for that run:")
    for line in log_lines:
        print("     " + str(line))

    print()
    if not wins:
        print("RESULT  NO setting in this grid achieves both. The trade is real: "
              "reciprocal-best\n        off fuses the stranger, on strands the "
              "chain.")
        if not has_quorum:
            print("\n        member_quorum is absent, so complete linkage is fixed "
                  "at 1.0 and the\n        chain question is open BY CONSTRUCTION. "
                  "RECONCILE_PATCHES.md C1 adds it\n        with default 1.0, which "
                  "is bit-identical to today -- applying that CODE\n        changes "
                  "no decision and is sanctioned. What is NOT sanctioned is "
                  "setting\n        the quorum below 1.0 in config.yaml: that voids "
                  "the bars, needs a sweep\n        and a re-render, and targets the "
                  "cam_206 stranding rather than the\n        operator's reported "
                  "split. Apply, re-run this, record the answer, stop.")
    else:
        print("RESULT  settings that closed the chain AND kept the stranger out:")
        for mode, recip, rounds, quorum in wins:
            print(f"          scoring={mode} recip={recip} rounds={rounds} "
                  f"quorum={quorum}")
        print("\n        This is a 4-tracklet synthetic fixture, not evidence about "
              "your footage.\n        It shows a setting CAN do both; whether it "
              "does on cam_206 is a sweep\n        plus rerender_from_clips.py, "
              "and watching it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
