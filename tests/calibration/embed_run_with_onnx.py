"""
embed_run_with_onnx.py  --  put a different model's vectors in a SIDE collection.

    python tests/calibration/embed_run_with_onnx.py 20260806_095141 \
        --onnx resnet50_market1501_aicity156.onnx \
        --onnx-size 256x128 --onnx-std 0.226,0.226,0.226 \
        --collection-out persons_reidnet

Then render it (needs the one-line --collection flag, see the bottom of this
file):

    python tests/calibration/rerender_from_clips.py 20260806_095141 \
        --collection persons_reidnet \
        --cross <printed> --same "cam_213=<printed>,cam_219=<printed>,cam_224=<printed>" \
        --out "reidnet_{name}.mp4"

WHY A SIDE COLLECTION. A different backbone is a different feature space: the
width changes, so `persons` cannot hold both, and every threshold in config.yaml
is void because it was derived against the shipping model's score distribution
(extractor.py says this outright). Writing beside the real collection means the
production one is never touched, the comparison can be repeated, and reverting
is a delete rather than a re-capture.

WHAT IS HELD FIXED. Point ids, payloads, boxes, track ids and run_id are copied
verbatim from `persons`. ONLY the vector changes. So the tracklets reconcile
sees are identical and any difference in the output is the model, not a
renumbering.

THE THRESHOLDS IT PRINTS ARE A STARTING POINT, NOT A CALIBRATION. They are
percentile-matched: whatever fraction of provable strangers the shipping bar
sits above, the suggested bar sits above the same fraction in the new model's
distribution. That is the only defensible way to carry a threshold across
feature spaces without new labels -- but it assumes the two distributions have
the same SHAPE, and for ReIdentificationNet they measurably do not (its stranger
p50 is 0.683 against FastReID's 0.360, and its spread is half as wide). Treat
the printed values as the middle of a sweep, and watch the video.

Uses raw Qdrant REST rather than PersonVectorStore, deliberately: the store
guards its collection against a dimension mismatch, which is correct for
production and exactly wrong for a side-by-side.
"""

import argparse
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))

from compare_backbones_on_run import (  # noqa: E402
    embed_onnx, fetch_observations, load_crops, prototypes,
    provable_strangers, pct)


def fetch_points(url, collection, run_id):
    """Every point for the run, payload included, vectors NOT.

    Point ids come back so the new collection can reuse them: same id, same
    payload, different vector. Nothing downstream can then tell the two
    collections apart except by what the model saw.
    """
    pts = []
    offset = None
    while True:
        body = {"limit": 500, "with_vector": False, "with_payload": True,
                "filter": {"must": [{"key": "run_id",
                                     "match": {"value": run_id}}]}}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{url}/collections/{collection}/points/scroll",
                          json=body, timeout=300)
        r.raise_for_status()
        res = r.json()["result"]
        pts.extend(res["points"])
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return pts


def recreate_collection(url, name, dim):
    """Drop and rebuild at the new width. Destructive to `name` ONLY."""
    requests.delete(f"{url}/collections/{name}", timeout=60)
    r = requests.put(
        f"{url}/collections/{name}",
        json={"vectors": {"size": int(dim), "distance": "Cosine"}},
        timeout=60)
    r.raise_for_status()
    print(f"[embed] collection '{name}' created at dim {dim}, cosine")


def upsert(url, name, points, batch=256):
    for i in range(0, len(points), batch):
        chunk = points[i:i + batch]
        r = requests.put(f"{url}/collections/{name}/points",
                         json={"points": chunk},
                         params={"wait": "true"}, timeout=300)
        r.raise_for_status()
    print(f"[embed] upserted {len(points)} point(s)")


def suggest_bars(old_diff, new_diff, bars):
    """Carry a threshold across feature spaces by matching its PERCENTILE.

    A bar is only meaningful relative to the score distribution it was tuned
    against. 0.55 means "above 88% of provable strangers" in one space and
    something else entirely in another -- so the transferable quantity is the
    percentile, not the number.

    ASSUMES THE DISTRIBUTIONS HAVE THE SAME SHAPE, which is a real assumption and
    frequently false. If the new model's spread is much narrower, matching a high
    percentile lands on a bar with almost no margin either side, and small
    differences between people stop being separable at any value. Sweep around
    what this prints; do not trust it alone.
    """
    old = np.asarray(old_diff)
    new = np.asarray(new_diff)
    out = {}
    for name, v in bars.items():
        q = 100.0 * float((old < v).mean())
        out[name] = (v, q, pct(new, min(99.9, q)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--url", default=os.environ.get("QDRANT_URL",
                                                    "http://localhost:6333"))
    ap.add_argument("--collection", default="persons")
    ap.add_argument("--collection-out", default="persons_onnx")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--onnx-size", default="256x128")
    ap.add_argument("--onnx-mean", default="0.485,0.456,0.406")
    ap.add_argument("--onnx-std", default="0.229,0.224,0.225")
    ap.add_argument("--onnx-bgr", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cross", type=float, default=0.55,
                    help="the SHIPPING cross bar, for percentile matching")
    ap.add_argument("--same", type=float, default=0.60,
                    help="the SHIPPING same-camera bar, for percentile matching")
    args = ap.parse_args()

    print(f"[embed] run {args.run_id}")
    obs = fetch_observations(args.url, args.collection, args.run_id)
    if not obs:
        raise SystemExit(f"[embed] no observations for {args.run_id}")
    points = fetch_points(args.url, args.collection, args.run_id)
    print(f"[embed] {len(obs)} tracklet(s), {len(points)} point(s)")

    # EVERY observation, not a sample: these vectors go into a collection that
    # reconcile will read, so a capped subset would silently change what a
    # prototype is built from and the comparison would not be like for like.
    print("[embed] cutting crops...")
    crops, owners = load_crops(obs, args.dir, cap_per_tracklet=0)
    if not crops:
        raise SystemExit("[embed] no crops -- check --dir")
    print(f"[embed] {len(crops)} crop(s)")

    if len(crops) != len(points):
        # Not fatal, but it means some stored observation had no matching frame
        # in the clip -- usually a clip from a different run. Say so rather than
        # writing a collection that is quietly missing people.
        print(f"[embed] WARNING: {len(crops)} crops vs {len(points)} points. "
              f"The join dropped some observations; check --dir names this run.")

    H, W = (int(x) for x in args.onnx_size.lower().split("x"))
    feats = embed_onnx(crops, os.path.expanduser(args.onnx), (H, W),
                       [float(x) for x in args.onnx_mean.split(",")],
                       [float(x) for x in args.onnx_std.split(",")],
                       not args.onnx_bgr, args.batch)
    feats = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True),
                            1e-12, None)
    dim = feats.shape[1]
    print(f"[embed] {feats.shape[0]} vector(s), dim {dim}")

    # Crops were cut in (camera, ascending frame) order and `points` came back in
    # scroll order, so they cannot be zipped positionally. Re-key both by
    # (camera, track_id) and pair within each tracklet, in time order.
    by_key_rows = {}
    for i, k in enumerate(owners):
        by_key_rows.setdefault(k, []).append(i)
    by_key_pts = {}
    for p in points:
        pl = p["payload"]
        by_key_pts.setdefault((pl["camera"], int(pl["track_id"])), []).append(p)
    for k in by_key_pts:
        by_key_pts[k].sort(key=lambda p: p["payload"].get("ts") or 0.0)

    out_points, skipped = [], 0
    for k, rows in by_key_rows.items():
        pts = by_key_pts.get(k, [])
        n = min(len(rows), len(pts))
        skipped += max(len(rows), len(pts)) - n
        for r, p in zip(rows[:n], pts[:n]):
            out_points.append({"id": p["id"],
                               "vector": feats[r].astype(float).tolist(),
                               "payload": p["payload"]})
    if skipped:
        print(f"[embed] {skipped} observation(s) unpaired and dropped")

    recreate_collection(args.url, args.collection_out, dim)
    upsert(args.url, args.collection_out, out_points)

    # ---- suggested bars, from each model's own stranger distribution --------
    strangers = provable_strangers(obs, 0.5)
    new_protos = prototypes(feats, owners)
    new_diff = [float(new_protos[a] @ new_protos[b])
                for a, b, _ in strangers if a in new_protos and b in new_protos]

    print("\n[embed] re-embedding the SHIPPING model to percentile-match bars...")
    from compare_backbones_on_run import embed_fastreid
    old_protos = prototypes(embed_fastreid(crops, args.batch), owners)
    old_diff = [float(old_protos[a] @ old_protos[b])
                for a, b, _ in strangers if a in old_protos and b in old_protos]

    if old_diff and new_diff:
        sug = suggest_bars(old_diff, new_diff,
                           {"cross": args.cross, "same": args.same})
        print(f"\n  provable strangers, n={len(new_diff)}")
        print(f"    shipping model : p50 {pct(np.asarray(old_diff),50):.3f}   "
              f"p95 {pct(np.asarray(old_diff),95):.3f}")
        print(f"    new model      : p50 {pct(np.asarray(new_diff),50):.3f}   "
              f"p95 {pct(np.asarray(new_diff),95):.3f}")
        print("\n  SUGGESTED BARS (percentile-matched, a starting point only)")
        for name, (old_v, q, new_v) in sug.items():
            print(f"    {name:6s} {old_v:.2f} sits above {q:.1f}% of strangers "
                  f"-> try {new_v:.3f}")
        c = sug["cross"][2]
        s = sug["same"][2]
        print(f"""
  Render it:

    python tests/calibration/rerender_from_clips.py {args.run_id} \\
        --collection {args.collection_out} \\
        --cross {c:.2f} \\
        --same "cam_213={s:.2f},cam_219={s:.2f},cam_224={s:.2f}" \\
        --out "reidnet_{{name}}.mp4"

  And sweep either side of those -- a percentile match is not a calibration,
  and if the new model's spread is narrower the right bar may not exist at all.""")

    print(f"""
{'=' * 78}
rerender_from_clips.py NEEDS ONE LINE FOR --collection
{'=' * 78}
  In its argument list, add "--collection" to extra_flags:

      extra_flags=("--url", "--path", "--out", "--fps", "--dir",
                   "--min-visible-kp", "--collection")

  and where it builds the store:

      store = PersonVectorStore(path=arg("--path", "qdrant_data"),
                                url=arg("--url", "http://localhost:6333") or None)
      coll = arg("--collection", "")
      if coll:
          store.collection = coll          # side collection, production untouched

  Without it the render reads `persons` and you will be watching the SHIPPING
  model's identities while believing they are the new model's.

  To revert everything: curl -X DELETE {args.url}/collections/{args.collection_out}
  The production collection is never written by this script.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())