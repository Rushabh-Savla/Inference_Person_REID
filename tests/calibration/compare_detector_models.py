"""
Does a BIGGER detector find people the small one misses?

    python tests/calibration/compare_detector_models.py [video] [frames] [models...]
    python tests/calibration/compare_detector_models.py register_file.avi 60 \
        yolo11n.pt yolo11s.pt yolo11m.pt

Exists because of one operator observation (REMEDIATION_PLAN.md J.8): in cam_206,
5 people were present and only 1 was detected at first, then it "randomly starts
detecting the rest". That is DETECTION RECALL. Nothing downstream can invent a box
YOLO never produced -- no threshold, no merge rule, no reconcile fix reaches it --
so it is the one failure a bigger model can actually address.

Also the one it can HIDE: a model 10x the cost can drop enough frames on four live
streams to fragment tracks worse than the extra recall repairs. That trade cannot
be measured here (CPU, one file, no RTSP); this script answers only the recall half
and reports RELATIVE cost. The throughput half is the per-camera `dropped%` in the
live run's final metrics block.

WHAT TO BELIEVE. Same frames, same conf/iou/imgsz, fresh ByteTrack per model, so
the only difference is capacity. Trust:
  * per-frame detection counts -- does the bigger model see more people at once?
  * how early it reaches full count -- the J.8 symptom is a SLOW START
  * distinct track ids and track lengths -- more boxes are only useful if they
    become sustained tracks rather than one-frame noise
Do not trust ms/frame as anything but a ratio: CPU here, GPU there.
"""

import sys
import time
from collections import defaultdict

import numpy as np

from _common import bootstrap, pick_video, consecutive_frames, header

bootstrap()

from ultralytics import YOLO

VIDEO = pick_video(sys.argv[1] if len(sys.argv) > 1 else None)
NFRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 50
MODELS = sys.argv[3:] or ["yolo11n.pt", "yolo11m.pt"]

# Mirror the shipped detector config exactly, so this measures the MODEL and
# nothing else. imgsz is Ultralytics' default because the pipeline never passes it.
CONF, IOU, IMGSZ = 0.40, 0.60, 640

frames = consecutive_frames(VIDEO, NFRAMES)
print(f"[calib] {VIDEO}: {len(frames)} CONSECUTIVE frames, "
      f"{frames[0].shape[1]}x{frames[0].shape[0]}")
print(f"[calib] conf={CONF} iou={IOU} imgsz={IMGSZ} (as shipped)   "
      f"models: {', '.join(MODELS)}")

results = {}
for name in MODELS:
    model = YOLO(name)                     # fresh model => fresh ByteTrack state
    per_frame, per_track = [], defaultdict(int)
    t0 = time.time()
    for f in frames:
        r = model.track(f, classes=[0], conf=CONF, iou=IOU, imgsz=IMGSZ,
                        tracker="bytetrack.yaml", persist=True, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            per_frame.append(0)
            continue
        tids = [int(v) for v in r.boxes.id]
        per_frame.append(len(tids))
        for t in tids:
            per_track[t] += 1
    dt = time.time() - t0
    results[name] = {
        "per_frame": per_frame,
        "per_track": dict(per_track),
        "ms": 1000.0 * dt / max(1, len(frames)),
    }


header("A. PEOPLE FOUND PER FRAME  (the J.8 recall question)")
print(f"  {'model':<16}{'dets':>7}{'mean':>7}{'median':>8}{'max':>5}"
      f"{'frames@max':>12}{'empty':>7}{'ms/f':>8}")
print("  " + "-" * 70)
overall_max = max(max(r["per_frame"]) for r in results.values())
for name, r in results.items():
    a = np.array(r["per_frame"])
    print(f"  {name:<16}{a.sum():>7}{a.mean():>7.2f}{np.median(a):>8.1f}"
          f"{a.max():>5}{int((a == overall_max).sum()):>12}"
          f"{int((a == 0).sum()):>7}{r['ms']:>8.0f}")
print(f"\n  'frames@max' counts frames reaching {overall_max}, the most any model")
print("  found -- i.e. how often each model sees EVERYONE the best one ever saw.")

base = MODELS[0]
for name in MODELS[1:]:
    a, b = np.array(results[base]["per_frame"]), np.array(results[name]["per_frame"])
    better, worse = int((b > a).sum()), int((b < a).sum())
    print(f"\n  {name} vs {base}: more boxes on {better}/{len(a)} frames, "
          f"fewer on {worse}, equal on {len(a) - better - worse}")
    print(f"    total {b.sum() - a.sum():+d} detections   "
          f"cost {results[name]['ms'] / max(1e-9, results[base]['ms']):.1f}x (CPU ratio)")


header("B. SLOW START -- how many frames until the count settles?")
print("  The J.8 symptom is 1 of 5 people found AT FIRST. So the first frames")
print("  matter more than the average.\n")
for name, r in results.items():
    a = np.array(r["per_frame"])
    peak = a.max()
    first_peak = int(np.argmax(a == peak)) if peak else -1
    print(f"  {name:<16} first 10 frames: {a[:10].tolist()}")
    print(f"  {'':<16} reaches its own peak ({peak}) at frame {first_peak}")


header("C. DO THE EXTRA BOXES BECOME REAL TRACKS?")
print("  More detections are only progress if they turn into SUSTAINED tracks.")
print("  A model that adds one-frame boxes adds fragments for reconcile to clean")
print("  up, which is the opposite of help.\n")
print(f"  {'model':<16}{'ids':>5}{'mean len':>10}{'>=5 frames':>12}{'<3 frames':>11}")
print("  " + "-" * 54)
for name, r in results.items():
    lens = sorted(r["per_track"].values(), reverse=True)
    print(f"  {name:<16}{len(lens):>5}{(np.mean(lens) if lens else 0):>10.1f}"
          f"{sum(1 for L in lens if L >= 5):>12}{sum(1 for L in lens if L < 3):>11}")
for name, r in results.items():
    lens = sorted(r["per_track"].values(), reverse=True)
    print(f"    {name:<16} track lengths: {lens[:14]}")


header("HOW TO READ THIS")
print(f"""  KEEP the bigger model if it finds more people per frame AND those boxes
  become tracks of >=5 frames. REVERT it if the extra ids are all short: that is
  noise, and it costs {results[MODELS[-1]]['ms'] / max(1e-9, results[MODELS[0]]['ms']):.1f}x the compute to produce.

  If every model reports the SAME counts, capacity is not what loses people on
  this clip -- and this clip is not cam_206. The people cam_206 misses are in a
  crowded room with a table, which nothing local reproduces. In that case the
  answer is the frozen cam_206 replay clip, not a bigger model.

  Whatever this says, the live run decides: check per-camera `dropped%` and
  `infer_q dropped` in the final metrics block. A 10x detector that drops frames
  fragments tracks, which reconcile then has to re-merge -- trading a recall
  problem for an identity problem.""")
