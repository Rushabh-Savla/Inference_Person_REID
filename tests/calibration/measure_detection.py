"""
Detection + tracking measurement: NMS duplicates and fragmentation.

Produces REMEDIATION_PLAN.md Part H.6 / H.7. This script exists mainly to STOP
changes: it is what showed that `iou: 0.60` already suppresses duplicate boxes,
and that raising `imgsz` or lowering `conf` buys nothing measurable. Re-run it
before proposing any detector-side change.

    python tests/calibration/measure_detection.py [video] [frames]

Uses CONSECUTIVE frames, not a strided sample -- ByteTrack needs real continuity,
so a strided sample measures a harder problem than the pipeline faces.
"""

import sys
import time
from collections import defaultdict

import numpy as np

from _common import (bootstrap, pick_video, consecutive_frames, DETECT_WEIGHTS,
                     iou, containment, header)

bootstrap()

from ultralytics import YOLO

VIDEO = pick_video(sys.argv[1] if len(sys.argv) > 1 else None)
NFRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 50

frames = consecutive_frames(VIDEO, NFRAMES)
print(f"[calib] {VIDEO}: {len(frames)} CONSECUTIVE frames, "
      f"{frames[0].shape[1]}x{frames[0].shape[0]}")


header("A. NMS -- does lowering `iou` remove duplicate boxes?   (Part H.7)")
print("  Direction reminder: `iou` is the overlap ABOVE WHICH a lower-confidence")
print("  box is SUPPRESSED. LOWER = suppress more = fewer duplicates.\n")
print(f"  {'iou':>6}{'boxes':>8}{'pairs IoU>=.5':>15}{'pairs IoU>=.7':>15}"
      f"{'pairs contain>=.8':>19}")
model = YOLO(DETECT_WEIGHTS)
per_iou = {}
for nms_iou in (0.70, 0.60, 0.50, 0.45, 0.35):
    total = hi5 = hi7 = cont8 = 0
    counts = []
    for f in frames:
        r = model(f, classes=[0], conf=0.4, iou=nms_iou, verbose=False)[0]
        boxes = [] if r.boxes is None else [b.xyxy[0].tolist() for b in r.boxes]
        counts.append(len(boxes))
        total += len(boxes)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                v, c = iou(boxes[i], boxes[j]), containment(boxes[i], boxes[j])
                hi5 += v >= 0.5
                hi7 += v >= 0.7
                cont8 += c >= 0.8
    per_iou[nms_iou] = counts
    mark = "  <-- configured" if abs(nms_iou - 0.60) < 1e-9 else ""
    print(f"  {nms_iou:>6.2f}{total:>8}{hi5:>15}{hi7:>15}{cont8:>19}{mark}")

base = per_iou[0.60]
for other in (0.45, 0.35):
    delta = [b - o for b, o in zip(base, per_iou[other])]
    changed = sum(1 for d in delta if d != 0)
    print(f"\n  0.60 -> {other}: {sum(delta)} fewer boxes overall, "
          f"{changed}/{len(frames)} frames changed")
print("\n  If the 0.60 row already shows ~0 high-overlap pairs, duplicate boxes are")
print("  NOT an NMS problem on this footage and lowering `iou` will do nothing.")


header("B. FRAGMENTATION vs imgsz / conf / frame dropping   (Part H.6)")
print("  `imgsz` is never passed by the pipeline, so Ultralytics' default 640")
print("  applies -- a 4x downscale on 2560x1440. `conf: 0.4` also empties")
print("  ByteTrack's second-association pool (which spans 0.1-0.25), disabling its")
print("  occlusion-recovery stage. Both are measured here rather than assumed.\n")
print(f"  {'config':<46}{'proc':>5}{'ids':>5}{'dets':>6}{'meanlen':>9}"
      f"{'>=5f':>6}{'<3f':>5}{'ms/f':>7}")
print("  " + "-" * 87)

rows = []
for stride, sname in ((1, "no drop"), (3, "2of3 dropped")):
    for imgsz in (640, 1280):
        for conf, post, cname in ((0.40, None, "conf .40 [shipped]"),
                                  (0.10, 0.40, "conf .10 +post .40")):
            m = YOLO(DETECT_WEIGHTS)      # fresh model => fresh ByteTrack state
            seen, per_track, ndet, nproc = set(), defaultdict(int), 0, 0
            t0 = time.time()
            for i, f in enumerate(frames):
                if i % stride:
                    continue
                nproc += 1
                r = m.track(f, classes=[0], conf=conf, iou=0.60, imgsz=imgsz,
                            tracker="bytetrack.yaml", persist=True, verbose=False)[0]
                if r.boxes is None or r.boxes.id is None:
                    continue
                for tid, c in zip([int(v) for v in r.boxes.id],
                                  [float(v) for v in r.boxes.conf]):
                    if post is not None and c < post:
                        continue
                    seen.add(tid)
                    per_track[tid] += 1
                    ndet += 1
            dt = time.time() - t0
            lens = sorted(per_track.values(), reverse=True)
            tag = f"imgsz {imgsz:<4} {cname:<19} {sname}"
            rows.append((tag, lens))
            print(f"  {tag:<46}{nproc:>5}{len(seen):>5}{ndet:>6}"
                  f"{(np.mean(lens) if lens else 0):>9.1f}"
                  f"{sum(1 for L in lens if L >= 5):>6}"
                  f"{sum(1 for L in lens if L < 3):>5}"
                  f"{1000 * dt / max(1, nproc):>7.0f}")

print("\n  track lengths (frames per track id), longest first:")
for tag, lens in rows:
    print(f"    {tag:<46} {lens[:12]}")

header("HOW TO READ THIS")
print("""  For the SAME people, FEWER distinct ids and LONGER tracks = less
  fragmentation = fewer tracklets for reconcile to re-merge.

  If every row shows the same id count, then on this footage NONE of these knobs
  affects fragmentation, and changing them is wasted effort. That was the result
  on register_file.avi -- including under 2-of-3 frame dropping, which produced no
  extra fragmentation at all.

  That does NOT mean fragmentation is not real in production: a prior 3-camera
  RTSP run reconciled 35 tracklets into 7 identities. It means this clip does not
  reproduce it, so a fix cannot be validated here. Capture footage that fragments
  (REMEDIATION_PLAN.md Part E) before tuning the detector.

  ms/f is CPU-only -- treat it as RELATIVE cost between configs, never as a
  throughput figure for the GPU box.""")
