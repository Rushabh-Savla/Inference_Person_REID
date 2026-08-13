"""
propose_merges.py  --  cross-camera positives from TIMING, not appearance.

    python tests/calibration/propose_merges.py 20260806_095141 > merges.txt
    python tests/calibration/propose_merges.py 20260806_095141 20260806_081045 \
        --max-transit 8 --verbose

WHY THIS EXISTS. Fine-tuning needs cross-camera positives -- "this cam_213
tracklet and that cam_224 tracklet are one person" -- and those are exactly what
the encoder gets wrong, so they cannot be derived from the encoder without
circularity. Asking an operator to write them by hand is the usual answer and it
does not scale.

But this deployment has a fact that replaces the operator: cam_213 is a
CHOKEPOINT. Nobody reaches cam_219 or cam_224 without crossing it. So a tracklet
appearing in the room a few seconds after exactly one person left the corridor is
that person, by elimination -- an argument from topology and wall-clock, using no
pixels at all.

THE ONLY LINKS IT EMITS ARE UNAMBIGUOUS ONES. A candidate is kept only when the
match is one-to-one in BOTH directions inside the transit window: this exit has
exactly one possible arrival, and that arrival has exactly one possible exit. Two
people crossing together produces two exits and two arrivals, so every pairing is
ambiguous and ALL of them are discarded. Quiet moments give clean labels; busy
moments give none. That is the correct trade for training data, where one wrong
positive teaches the model that two people are the same person.

AND CO-PRESENCE OVERRIDES EVERYTHING. If a proposed pair is ever visible
simultaneously in one camera, they are provably two people and the pair is
dropped no matter how clean the timing looked.

WHAT IT CANNOT DO. It links across the chokepoint. It does not link two room
cameras to each other directly -- but it does not need to: if cam_219:7 and
cam_224:3 both link to cam_213:19, they land in one group transitively.

Output is merges.txt for export_reid_dataset.py. EYEBALL IT before training: a
wrong positive is worse than a missing one, and the reasoning is printed beside
every line with --verbose.
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from compare_backbones_on_run import (  # noqa: E402
    fetch_observations, provable_strangers)


class Union:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def spans(obs):
    return {k: (v[0][0], v[-1][0]) for k, v in obs.items() if v}


def propose(obs, anchor, max_transit, min_gap, min_obs, forbidden):
    """-> [(exit_key, arrival_key, gap, reason)] for unambiguous transits only.

    A candidate arrival starts between min_gap and max_transit seconds after an
    anchor tracklet ends. min_gap can be NEGATIVE: the corridor and the room may
    overlap slightly in view, so a person can appear in the room a moment before
    the corridor loses them.
    """
    sp = spans(obs)
    counts = {k: len(v) for k, v in obs.items()}
    exits = [k for k in sp if k[0] == anchor and counts[k] >= min_obs]
    arrivals = [k for k in sp if k[0] != anchor and counts[k] >= min_obs]

    fwd, back = defaultdict(list), defaultdict(list)
    for e in exits:
        for a in arrivals:
            if (e, a) in forbidden or (a, e) in forbidden:
                continue
            gap = sp[a][0] - sp[e][1]
            if min_gap <= gap <= max_transit:
                fwd[e].append((gap, a))
                back[a].append((gap, e))

    out = []
    for e, cands in fwd.items():
        if len(cands) != 1:
            continue
        gap, a = cands[0]
        # One-to-one in BOTH directions. Without the reverse check, two people
        # leaving together and one arriving would produce a confident-looking
        # link to whichever exit happened to have a single candidate.
        if len(back[a]) != 1:
            continue
        out.append((e, a, gap,
                    f"only arrival {gap:+.1f}s after {e[0]}:{e[1]:04d} exits, "
                    f"and its only candidate exit"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--anchor", default="cam_213",
                    help="the chokepoint camera everyone must cross")
    ap.add_argument("--max-transit", type=float, default=6.0,
                    help="seconds allowed between leaving the anchor and arriving")
    ap.add_argument("--min-gap", type=float, default=-1.5,
                    help="negative allows a small view overlap at the handover")
    ap.add_argument("--min-obs", type=int, default=5,
                    help="ignore tracklets thinner than this; their spans are "
                         "unreliable and a bad span makes a bad link")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--url", default=os.environ.get("QDRANT_URL",
                                                    "http://localhost:6333"))
    ap.add_argument("--collection", default="persons")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = (lambda m: print(m, file=sys.stderr))
    total = 0

    for run_id in args.runs:
        obs = fetch_observations(args.url, args.collection, run_id)
        if not obs:
            log(f"# {run_id}: no observations (store wiped?) -- skipped")
            continue

        forbidden = set()
        for a, b, _ in provable_strangers(obs, args.min_overlap):
            forbidden.add((a, b))
            forbidden.add((b, a))

        links = propose(obs, args.anchor, args.max_transit, args.min_gap,
                        args.min_obs, forbidden)

        uf = Union()
        for e, a, _, _ in links:
            uf.union(e, a)

        groups = defaultdict(list)
        for e, a, _, _ in links:
            groups[uf.find(e)].append(e)
            groups[uf.find(a)].append(a)

        emitted = 0
        for root, members in sorted(groups.items()):
            members = sorted(set(members))
            if len(members) < 2:
                continue
            # A group whose own members are co-present contradicts itself. Drop
            # it rather than emit a contradiction into the training set.
            bad = [(x, y) for i, x in enumerate(members)
                   for y in members[i + 1:] if (x, y) in forbidden]
            if bad:
                log(f"# {run_id}: DROPPED group {members} -- members co-present "
                    f"{bad}, so provably not one person")
                continue
            print(" ".join(f"{c}:{t}" for c, t in members))
            emitted += 1
            total += 1

        log(f"# {run_id}: {len(obs)} tracklet(s), {len(links)} unambiguous "
            f"transit(s), {emitted} group(s)")
        if args.verbose:
            for e, a, gap, why in sorted(links, key=lambda x: x[2]):
                log(f"#   {e[0]}:{e[1]:04d} -> {a[0]}:{a[1]:04d}  {why}")

    log(f"""#
# {total} group(s) total.
#
# EYEBALL THESE BEFORE TRAINING. One wrong positive teaches the model that two
# people are one, which is the failure being fixed. Check a few against the
# contact sheets:
#     python tests/calibration/contact_sheet_halves.py <run>
# then compare sheets/halves_<cam>_<track>.png for the members of a line.
#
# TOO FEW GROUPS? The window is probably too tight, or the room was too busy for
# any transit to be unambiguous. Try --max-transit 10 --verbose and read which
# candidates were rejected. Ambiguity is not a bug here -- two people crossing
# together genuinely cannot be told apart by timing, and a guess would be worse
# than a gap.
#
# TOO MANY, or they look wrong? Tighten --max-transit, or raise --min-obs so
# thin tracklets with unreliable spans stop participating.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())