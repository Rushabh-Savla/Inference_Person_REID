"""
sweep_against_labels.py  --  sweep the same-camera bars against GROUND TRUTH.

    python tests/calibration/sweep_against_labels.py <run_id>
    python tests/calibration/sweep_against_labels.py <run_id> --cam-a cam_219 --cam-b cam_213

WHY THIS IS DIFFERENT FROM sweep_reconcile_thresholds.py, which already exists.

That tool ranks settings by IDENTITY COUNT. A count cannot tell you whether a cluster
is one person or three (CLAUDE.md section 4), and every threshold change this project
has reverted was chosen that way. This one scores each setting on two things that are
either LABELLED or PROVABLE:

  * labelled same-person pairs, from calibration/tracklet_pairs.jsonl (or the three
    operator-confirmed pairs in CLAUDE.md 3b as a fallback) -- did they end up in
    ONE cluster?
  * PROVABLE false merges: pairs co-present in the SAME camera for more than
    --min-overlap seconds are two different people (one body cannot be two
    simultaneous detections). Did any of them end up in one cluster?

Cross-camera co-presence proves nothing here because the cameras overlap
(CLAUDE.md 6.1), so it is never used.

It calls the REAL reconcile_tracklets over the run's REAL stored vectors, so there is
no reimplementation to drift. Deterministic, CPU-only, no model: it reads embeddings
Qdrant already holds and does arithmetic, so it gives the same answer on the dev box
and on the A6000.

WHAT "ZERO FALSE MERGES" DOES AND DOES NOT MEAN. Co-presence can only convict pairs
that appear together in one camera. Two people who never share a frame can still be
merged wrongly and this reference cannot see it. So a clean column is necessary, not
sufficient -- the winner still has to be WATCHED (CLAUDE.md section 4), and a plateau
of adjacent clean cells should be preferred over an isolated one, because an isolated
cell is an overfit to three labels.
"""

import argparse
import collections
import itertools
import json
import os
from types import SimpleNamespace

import numpy as np

from _common import bootstrap, header

ROOT = bootstrap()

from identity.reconcile import (describe_reconcile_kwargs,  # noqa: E402
                                reconcile_tracklets,
                                resolve_reconcile_kwargs)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "persons")
LABEL_PATH = "calibration/tracklet_pairs.jsonl"

# CLAUDE.md 3b. NOTE the doc writes these as "0020 vs 0008", which reads like
# reconcile's U-handles and is NOT -- they are TRACK IDs. Verified by reproducing all
# three published prototype cosines exactly (0.574 / 0.630 / 0.907).
FALLBACK = [("20260804_064551", ("cam_219", 20), ("cam_219", 8)),
            ("20260804_064551", ("cam_213", 31), ("cam_213", 35)),
            ("20260804_064551", ("cam_224", 1), ("cam_224", 30))]


class _ReadOnlyStore:
    """reconcile needs client.scroll, collection, set_global_id, clear_global_id.
    The writes are NO-OPS on purpose: a sweep must never restamp the gallery, or
    every later comparison is against a store the previous cell mutated."""

    collection = COLLECTION

    def __init__(self, points):
        self.client = SimpleNamespace(
            scroll=lambda col, limit, offset, with_payload, with_vectors: (
                points, None))

    def set_global_id(self, point_ids, global_id):
        pass

    def clear_global_id(self, point_ids):
        pass


def fetch(run_id):
    import requests
    points, offset = [], None
    while True:
        body = {"limit": 500, "with_vector": True,
                "with_payload": ["camera", "track_id", "frame", "ts"],
                "filter": {"must": [{"key": "run_id", "match": {"value": run_id}}]}}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                          json=body, timeout=300)
        r.raise_for_status()
        res = r.json()["result"]
        for p in res["points"]:
            pl = p["payload"]
            points.append(SimpleNamespace(
                id=p["id"], vector=p["vector"],
                payload={"camera": pl.get("camera"),
                         "track_id": int(pl.get("track_id")),
                         "frame": int(pl.get("frame")), "run_id": run_id,
                         "ts": (None if pl.get("ts") is None else float(pl["ts"]))}))
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return points


def labels_for(run_id):
    if os.path.exists(LABEL_PATH):
        out = []
        with open(LABEL_PATH) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("run_id") != run_id or not rec.get("same"):
                    continue
                try:
                    out.append(((str(rec["a"][0]), int(rec["a"][1])),
                                (str(rec["b"][0]), int(rec["b"][1]))))
                except (KeyError, IndexError, TypeError, ValueError):
                    print(f"  {LABEL_PATH}:{ln}: malformed -- skipped")
        return out, False
    return [(a, b) for r, a, b in FALLBACK if r == run_id], True


def provable_different(points, min_obs, min_overlap):
    """Same camera, co-present for > min_overlap seconds => two people."""
    idx = collections.defaultdict(list)
    for p in points:
        idx[(p.payload["camera"], p.payload["track_id"])].append(p)
    span = {}
    for k, v in idx.items():
        ts = [p.payload["ts"] for p in v if p.payload["ts"] is not None]
        if ts:
            span[k] = (min(ts), max(ts))
    keys = sorted(k for k, v in idx.items() if len(v) >= min_obs and k in span)
    return [(a, b) for a, b in itertools.combinations(keys, 2)
            if a[0] == b[0]
            and min(span[a][1], span[b][1]) - max(span[a][0], span[b][0]) > min_overlap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--cam-a", default="cam_219")
    ap.add_argument("--cam-b", default="cam_213")
    ap.add_argument("--bars", default="0.90,0.80,0.70,0.60,0.55,0.50")
    ap.add_argument("--cross", type=float, default=0.63)
    ap.add_argument("--hold", default="cam_224=0.80",
                    help="cam=bar[,cam=bar] held fixed")
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument("--min-overlap", type=float, default=0.5)
    args = ap.parse_args()

    points = fetch(args.run_id)
    if not points:
        print(f"[calib] run {args.run_id}: no stored observations at {QDRANT_URL}")
        return 1
    labels, fallback = labels_for(args.run_id)
    diff = provable_different(points, args.min_obs, args.min_overlap)
    cams = sorted({p.payload["camera"] for p in points})
    held = {}
    for kv in (args.hold or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            held[k.strip()] = float(v)

    print(f"[calib] run {args.run_id}: {len(points)} obs, cameras {cams}")
    print(f"[calib] {len(labels)} labelled same-person pair(s)"
          + ("  [FALLBACK: CLAUDE.md 3b, n=3 -- a hypothesis, not a calibration]"
             if fallback else f"  from {LABEL_PATH}"))
    print(f"[calib] {len(diff)} PROVABLE different-person pair(s) "
          f"(same camera, co-present > {args.min_overlap}s)")
    if not labels:
        print("[calib] no labels for this run -> nothing to score against. A sweep "
              "without labels ranks by identity count, which is what this tool "
              "exists to replace. Label a run first.")
        return 1

    bars = [float(x) for x in args.bars.split(",")]

    # EVERY setting except the two swept bars comes from config.yaml through the ONE
    # resolver, so this sweep cannot measure a clustering that does not ship. The
    # first version of this file hardcoded same_camera_rounds=True while production
    # runs False -- exactly the defect resolve_reconcile_kwargs' docstring was written
    # about ("the offline sweep and the offline re-render were measuring a different
    # clustering algorithm than the one that ships"). Verified: the recommendation is
    # unchanged at rounds=False, but a tool must not need that luck.
    import yaml
    with open("config.yaml") as f:
        prod_kw = resolve_reconcile_kwargs(yaml.safe_load(f) or {},
                                           log=lambda *a, **k: None)
    prod_kw["threshold"] = args.cross
    print("[calib] production settings held fixed: "
          + describe_reconcile_kwargs(prod_kw))

    def evaluate(bar_a, bar_b):
        per_cam = dict(held)
        per_cam[args.cam_a] = bar_a
        per_cam[args.cam_b] = bar_b
        kw = dict(prod_kw)
        kw["same_camera_thresholds"] = per_cam
        kw["min_tracklet_observations"] = args.min_obs
        remap = reconcile_tracklets(
            _ReadOnlyStore(points), run_id=args.run_id,
            log=lambda *a, **k: None, **kw)
        got = remap.get
        merged = [1 if (got(a) is not None and got(a) == got(b)) else 0
                  for a, b in labels]
        false_pairs = [(a, b) for a, b in diff
                       if got(a) is not None and got(a) == got(b)]
        return sum(merged), merged, false_pairs, len(set(remap.values()))

    header(f"labels_merged / PROVABLE_false_merges / identities      "
           f"cross={args.cross:.2f}  held {held}")
    print(f"  {args.cam_b + ' ->':>14}" + "".join(f"{b:>14.2f}" for b in bars))
    grid = {}
    for ba in bars:
        row = f"  {args.cam_a} {ba:.2f}"
        for bb in bars:
            n, per, fp, ids = evaluate(ba, bb)
            grid[(ba, bb)] = (n, per, fp, ids)
            row += f"{f'{n}/{len(labels)} . {len(fp)} . {ids}':>14}"
        print(row)

    clean = [(ba, bb) for (ba, bb), (n, _p, fp, _i) in grid.items()
             if n == len(labels) and not fp]
    header("VERDICT")
    ship = grid.get((max(bars), max(bars)))
    if ship:
        print(f"  at the STRICTEST cell ({max(bars):.2f}/{max(bars):.2f}): "
              f"{ship[0]}/{len(labels)} labels, {len(ship[2])} provable false "
              f"merges, {ship[3]} identities")
    if not clean:
        best = max(grid.items(), key=lambda kv: (kv[1][0], -len(kv[1][2])))
        print(f"  NO cell satisfies every label with zero provable false merges.")
        print(f"  best is {best[1][0]}/{len(labels)} at {args.cam_a}="
              f"{best[0][0]:.2f} {args.cam_b}={best[0][1]:.2f}. If the best cell is "
              f"short by a label, the bar is NOT the lever for that pair -- check its "
              f"RANKING (measure_rank_and_bias.py): if the true partner is not rank 1, "
              f"no bar can merge it under reciprocal-best.")
        return 0
    print(f"  {len(clean)} cell(s) satisfy ALL {len(labels)} labels with ZERO "
          f"provable false merges.")
    # Prefer the strictest cell inside the largest contiguous clean region: an
    # isolated clean cell is an overfit to a handful of labels.
    cs = set(clean)
    def neighbours(c):
        i, j = bars.index(c[0]), bars.index(c[1])
        return sum(1 for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if 0 <= i + di < len(bars) and 0 <= j + dj < len(bars)
                   and (bars[i + di], bars[j + dj]) in cs)
    ranked = sorted(clean, key=lambda c: (-neighbours(c), -c[0], -c[1]))
    pick = ranked[0]
    print(f"  RECOMMEND {args.cam_a}={pick[0]:.2f} {args.cam_b}={pick[1]:.2f} "
          f"-- strictest cell with the most clean neighbours "
          f"({neighbours(pick)} of 4), {grid[pick][3]} identities.")
    print("  A PLATEAU of adjacent clean cells is the point: it survives a +/- step")
    print("  in either bar. An isolated cell would be fitted to the labels.")
    print("\n  STILL REQUIRED BEFORE SHIPPING: watch it. Co-presence can only convict")
    print("  pairs that share a frame, so 'zero false merges' is necessary and NOT")
    print("  sufficient -- two people who never co-occur can still be fused and this")
    print("  reference cannot see it. See CLAUDE.md section 4.")
    print(f"    python tests/calibration/rerender_from_clips.py {args.run_id} \\")
    print(f"        --same \"{args.cam_a}={pick[0]:.2f},{args.cam_b}={pick[1]:.2f}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
