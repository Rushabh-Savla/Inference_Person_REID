"""
backfill_visible_kp.py  --  add visible_kp to a FINISHED run, from its clips.

    python tests/calibration/backfill_visible_kp.py 20260806_061201            # dry run
    python tests/calibration/backfill_visible_kp.py 20260806_061201 --write
    python tests/calibration/backfill_visible_kp.py 20260806_061201 --write --clips clips_20260806_061201

WHY THIS EXISTS. A visibility floor can only be judged on video, and
rerender_from_clips.py --min-visible-kp filters observations already in Qdrant --
so it needs a run whose payloads carry the metric. A run captured before the pose
patch has none, and re-capturing costs people walking the room.

But nothing is actually missing. The clip holds the pixels, the sidecar holds a
wall-clock stamp per clip frame, and every stored observation holds its own `ts`
and `bbox`. So the keypoints can be recomputed and joined back:

    payload.ts  --nearest-->  sidecar.frame_ts[i]  -->  clip frame i
    pose model on frame i     -->  boxes + keypoints
    best IoU against payload.bbox  -->  that observation's keypoints
    -->  visible_kp, written back onto the point

THE JOIN IS ON WALL CLOCK, NOT FRAME INDEX, and that is the whole reason it
works. The clip holds PROCESSED frames -- 2069 of 2138 read on cam_213 in run
20260806_061201 -- so clip index N is not payload `frame` N, and matching
positionally would silently pair one moment's box with another moment's crop.
`frame_ts` is the only thing that relates the two, which is the same rule
reconcile follows for cross-camera timing (#26): frame indices are per-source
and mean nothing outside their own stream.

WHAT IT CHANGES. It adds ONE top-level payload key, `visible_kp`. Vectors,
point ids, global_id and reid_id are never touched, so a reconcile of this run
produces exactly what it produced before -- until a floor is applied. Additive
and reversible.

WHAT IT CANNOT DO. It recomputes detection, so the pose model's boxes will not
be pixel-identical to the ones the run recorded. Every observation is matched by
IoU and anything below --min-iou is left WITHOUT the metric rather than guessed
at, and the match rate is reported. A low rate means the clip and the store
disagree about something and nothing here should be trusted.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import requests

NUM_KEYPOINTS = 17


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def fetch_run(url, collection, run_id):
    """-> {camera: [(point_id, ts, bbox), ...]}. Vectors are never requested."""
    out = defaultdict(list)
    offset = None
    while True:
        body = {"limit": 500, "with_vector": False,
                "with_payload": ["camera", "ts", "bbox", "frame"],
                "filter": {"must": [{"key": "run_id",
                                     "match": {"value": run_id}}]}}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{url}/collections/{collection}/points/scroll",
                          json=body, timeout=300)
        r.raise_for_status()
        res = r.json()["result"]
        for p in res["points"]:
            pl = p["payload"]
            ts, bbox = pl.get("ts"), pl.get("bbox")
            if ts is None or not bbox:
                continue
            out[pl.get("camera")].append(
                (p["id"], float(ts), [float(v) for v in bbox]))
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return out


def keypoint_rows(result):
    """(N, K, 3) aligned with result.boxes, or None."""
    kp = getattr(result, "keypoints", None)
    if kp is None or getattr(kp, "data", None) is None:
        return None
    try:
        arr = kp.data.cpu().numpy()
    except AttributeError:
        arr = np.asarray(kp.data)
    if arr.ndim != 3 or arr.shape[0] != len(result.boxes):
        return None
    return arr


def process_camera(cam, clip, sidecar, observations, model_path, imgsz, conf,
                   kp_conf, min_iou):
    """-> ({point_id: visible_kp}, stats)."""
    import cv2
    from ultralytics import YOLO

    with open(sidecar) as f:
        meta = json.load(f)
    frame_ts = meta.get("frame_ts")
    if not frame_ts:
        print(f"  {cam}: sidecar has no frame_ts -- cannot join. Skipped.")
        return {}, {}
    frame_ts = np.asarray(frame_ts, dtype=np.float64)

    # Every observation lands on its nearest clip frame. Several observations of
    # DIFFERENT people share a frame, which is why the IoU match below is per
    # observation rather than per frame.
    by_frame = defaultdict(list)
    dt_worst = 0.0
    for pid, ts, bbox in observations:
        i = int(np.abs(frame_ts - ts).argmin())
        dt_worst = max(dt_worst, abs(float(frame_ts[i]) - ts))
        by_frame[i].append((pid, bbox))

    model = YOLO(model_path)
    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        print(f"  {cam}: cannot open {clip}")
        return {}, {}

    kwargs = dict(classes=[0], conf=conf, verbose=False)
    if imgsz:
        kwargs["imgsz"] = imgsz

    found, unmatched, no_kp = {}, 0, 0
    idx = -1
    wanted = set(by_frame)
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx not in by_frame:
            continue
        wanted.discard(idx)
        # detect(), not track(): track ids are irrelevant here and persist=True
        # would make the result depend on which frames were visited.
        results = model(frame, **kwargs)
        boxes, kps = [], None
        for r in results:
            if r.boxes is None:
                continue
            kps = keypoint_rows(r)
            boxes = [tuple(b.xyxy[0].tolist()) for b in r.boxes]
        if kps is None:
            no_kp += len(by_frame[idx])
            continue
        for pid, bbox in by_frame[idx]:
            best_i, best_v = -1, 0.0
            for j, pb in enumerate(boxes):
                v = iou(bbox, pb)
                if v > best_v:
                    best_i, best_v = j, v
            if best_i < 0 or best_v < min_iou:
                unmatched += 1
                continue
            row = kps[best_i]
            if row.shape[1] < 3:
                no_kp += 1
                continue
            found[pid] = round(
                float((row[:, 2] >= kp_conf).sum()) / NUM_KEYPOINTS, 4)
    cap.release()

    total = len(observations)
    stats = {"total": total, "matched": len(found), "unmatched": unmatched,
             "no_kp": no_kp, "dt_worst": dt_worst}
    return found, stats


def write_payload(url, collection, values, batch=256):
    """One request per DISTINCT rounded value -- Qdrant sets one payload across
    many points, so grouping is far cheaper than one call each."""
    groups = defaultdict(list)
    for pid, v in values.items():
        groups[round(float(v), 2)].append(pid)
    sent = 0
    for value, ids in sorted(groups.items()):
        for i in range(0, len(ids), batch):
            r = requests.post(
                f"{url}/collections/{collection}/points/payload",
                json={"payload": {"visible_kp": value},
                      "points": ids[i:i + batch]},
                params={"wait": "true"}, timeout=120)
            r.raise_for_status()
            sent += len(ids[i:i + batch])
    return sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--clips", default=".")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--url", default=os.environ.get("QDRANT_URL",
                                                    "http://localhost:6333"))
    ap.add_argument("--collection", default="persons")
    ap.add_argument("--imgsz", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--kp-conf", type=float, default=0.5)
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    print(f"[backfill] run {args.run_id} @ {args.url}/{args.collection}")
    per_cam = fetch_run(args.url, args.collection, args.run_id)
    if not per_cam:
        raise SystemExit("[backfill] no observations for that run_id")
    for cam, obs in sorted(per_cam.items()):
        print(f"  {cam}: {len(obs)} observation(s)")

    all_values, all_stats = {}, {}
    for cam, obs in sorted(per_cam.items()):
        clip = os.path.join(args.clips, f"._live_src_{cam}.mp4")
        side = os.path.join(args.clips, f"._live_src_{cam}.annotations.json")
        if not (os.path.exists(clip) and os.path.exists(side)):
            print(f"  {cam}: clip or sidecar missing in {args.clips} -- skipped")
            continue
        with open(side) as f:
            meta = json.load(f)
        if meta.get("run_id") not in (None, args.run_id):
            # The same refusal rerender_from_clips makes, for the same reason:
            # clip filenames carry no run_id and every run overwrites them, so a
            # stale clip would silently pair one run's pixels with another's
            # observations and every number below would be fiction.
            print(f"  {cam}: sidecar says run {meta.get('run_id')}, "
                  f"NOT {args.run_id} -- REFUSED")
            continue
        print(f"  {cam}: detecting on {len(obs)} sampled frame(s)...")
        vals, stats = process_camera(cam, clip, side, obs, args.model,
                                     args.imgsz, args.conf, args.kp_conf,
                                     args.min_iou)
        all_values.update(vals)
        all_stats[cam] = stats
        if stats:
            pct = 100.0 * stats["matched"] / max(1, stats["total"])
            print(f"    matched {stats['matched']}/{stats['total']} ({pct:.1f}%)"
                  f"  unmatched {stats['unmatched']}  no-keypoints {stats['no_kp']}"
                  f"  worst ts gap {stats['dt_worst'] * 1000:.0f} ms")
            if pct < 80:
                print(f"    !! LOW MATCH RATE. The clip and the store disagree "
                      f"about geometry -- check the clip belongs to this run and "
                      f"that resize_width was 0. Do not trust these values.")

    if not all_values:
        raise SystemExit("\n[backfill] nothing matched; nothing to write")

    v = np.asarray(sorted(all_values.values()))
    print(f"\n[backfill] visible_kp over {len(v)} observation(s) "
          f"(joint conf >= {args.kp_conf})")
    print(f"  p05 {np.percentile(v, 5):.2f}  p25 {np.percentile(v, 25):.2f}  "
          f"p50 {np.percentile(v, 50):.2f}  p75 {np.percentile(v, 75):.2f}")
    for floor in (0.25, 0.4, 0.5, 0.6):
        print(f"  a floor at {floor:.2f} would drop "
              f"{100.0 * float((v < floor).mean()):.1f}% of matched observations")

    if not args.write:
        print("\n[backfill] DRY RUN -- nothing written. Re-run with --write.")
        return 0

    sent = write_payload(args.url, args.collection, all_values)
    print(f"\n[backfill] wrote visible_kp onto {sent} point(s).")
    print(f"""
Now the floor can be WATCHED, on this run's own footage and ids:

    R={args.run_id}
    python tests/calibration/rerender_from_clips.py $R --out "vk00_{{name}}.mp4"
    python tests/calibration/rerender_from_clips.py $R --min-visible-kp 0.25 --out "vk25_{{name}}.mp4"
    python tests/calibration/rerender_from_clips.py $R --min-visible-kp 0.50 --out "vk50_{{name}}.mp4"
    python tests/calibration/rerender_from_clips.py $R --min-visible-kp 0.60 --out "vk60_{{name}}.mp4"

Observations left WITHOUT the metric are kept by the filter, not dropped -- so
an unmatched crop can never be mistaken for an invisible one.

To undo: the key is additive and touches nothing else, so re-running reconcile
without a floor reproduces this run exactly as it was.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())