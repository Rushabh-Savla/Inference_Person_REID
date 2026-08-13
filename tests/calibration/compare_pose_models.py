"""
compare_pose_models.py  --  three detectors, one clip, no camera time.

    python tests/calibration/compare_pose_models.py
    python tests/calibration/compare_pose_models.py --frames 900 --imgsz 640,1280
    python tests/calibration/compare_pose_models.py --clips clips_20260805_115845

WHY THIS EXISTS. Swapping the detector changes what gets RECORDED, so it cannot
be judged by re-rendering a finished run -- and every capture costs people
walking the room. But the processed-frame clips a run leaves behind
(._live_src_<cam>.mp4) are the same pixels the pipeline saw, so three models can
be run over them back to back and compared on identical footage.

WHAT IT ANSWERS, in order of how much it should weigh on the decision:

  1. DOES RECALL DROP? A pose head is optimised for joints, and its box quality
     can differ from a detection head's. Nothing downstream can invent a box the
     detector never produced, so a model that finds fewer people is worse no
     matter what else it improves. This is the veto question.

  2. DOES cam_213 RECOVER? Measured on 300 identical frames, imgsz 1280 took
     cam_213 from 31 to 167 detections while cam_219 went 945 -> 936. cam_213's
     median crop height is 250 px against cam_219's 622, so its people sit below
     Ultralytics' default 640 usable scale. YOLO26's STAL label assignment
     targets exactly this, so it may recover the same ground at 640.

  3. WHERE WOULD A VISIBILITY GATE CUT? For the pose models, the distribution of
     visible-keypoint fraction, and the fraction of detections that a floor at
     0.3 / 0.4 / 0.5 / 0.6 would reject. COCO-17 is nose, 2 eyes, 2 ears,
     2 shoulders, 2 elbows, 2 wrists, 2 hips, 2 knees, 2 ankles -- so a SEATED
     person loses knees and ankles and reads 13/17 = 0.76, and standing behind a
     desk reads 15/17 = 0.88. In an office those two postures ARE the population,
     which is why a 0.6-0.7 floor cuts through the middle of it and 0.5 does not.

  4. DO NESTED BOXES GO AWAY? cam_219 measured 10 box pairs with containment
     >= 0.8 at EVERY iou from 0.70 down to 0.35 -- IoU-based NMS structurally
     cannot remove a box that sits inside another. YOLO26's end-to-end head is
     NMS-free with one-to-one assignment, so it should not emit them at all.
     Those duplicates are what put two track ids on one body, which is how a
     "provably different, co-present" stranger pair gets manufactured.

WHAT IT DOES NOT ANSWER. Whether the identities come out right. That needs a
capture and a watch (CLAUDE.md section 4: a count cannot tell you whether a
cluster is one person or three). This only decides which model is worth
capturing with.

READ-ONLY: no store, no writes, no config changes. Deterministic given a clip.
"""

import argparse
import glob
import os
import statistics
import time
from collections import defaultdict

import numpy as np

# COCO-17. Fixed by the pose head's output shape, not a setting.
NUM_KEYPOINTS = 17

# Reported as "what would a gate here cost". Not a recommendation -- 0.5 is the
# only one clear of the seated cluster, and even that needs the measured
# distribution before it ships.
GATE_FLOORS = (0.3, 0.4, 0.5, 0.6)

DEFAULT_MODELS = ("yolo11m.pt", "yolo11m-pose.pt", "yolo26m-pose.pt")


def containment(a, b):
    """Fraction of box `a`'s area lying inside box `b`. Same formula as
    detector.PersonDetector._containment, deliberately -- this measures the
    thing that gate would see."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area_a


def nested_pairs(boxes, floor=0.8):
    """Count box pairs where one sits >= `floor` inside another.

    NOT IoU. A small box 80% inside a large one has LOW IoU by construction, so
    NMS at any iou threshold leaves it -- which is why cam_219's count did not
    move between 0.70 and 0.35. Two boxes on one body become two track ids,
    co-present for their whole overlap, and reconcile then treats them as
    provably two people.
    """
    n = 0
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if containment(a, b) >= floor or containment(b, a) >= floor:
                n += 1
    return n


def keypoint_fractions(result):
    """Per-detection fraction of COCO-17 keypoints above 0.5 confidence.

    Empty list for a box-only checkpoint. Also empty when the head reports
    (x, y) with no confidence channel -- in that case there is nothing to
    threshold, and returning 1.0 for every crop would make the metric silently
    meaningless rather than absent.
    """
    kp = getattr(result, "keypoints", None)
    if kp is None or getattr(kp, "data", None) is None:
        return []
    try:
        arr = kp.data.cpu().numpy()
    except AttributeError:
        arr = np.asarray(kp.data)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return []
    return [float((row[:, 2] >= 0.5).sum()) / NUM_KEYPOINTS for row in arr]


def run_one(model_path, clip, frames, imgsz, conf, iou, tracker):
    """One model over one clip. Returns a stats dict, or None if it could not run."""
    from ultralytics import YOLO

    try:
        model = YOLO(model_path)
    except Exception as exc:                                    # noqa: BLE001
        print(f"    !! {model_path}: {type(exc).__name__}: {exc}")
        return None

    import cv2
    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        print(f"    !! cannot open {clip}")
        return None

    track_lengths = defaultdict(int)
    dets = 0
    nested = 0
    kp_fracs = []
    processed = 0
    t0 = time.time()

    kwargs = dict(classes=[0], conf=conf, tracker=tracker,
                  persist=True, verbose=False)
    # YOLO26's default path is NMS-free, so `iou` has nothing to act on. Passing
    # it anyway is harmless, but the run header says which models it can affect
    # so a reader never credits an NMS setting for an end-to-end result.
    if iou is not None:
        kwargs["iou"] = iou
    if imgsz:
        kwargs["imgsz"] = imgsz

    while processed < frames:
        ok, frame = cap.read()
        if not ok:
            break
        processed += 1
        results = model.track(frame, **kwargs)
        for r in results:
            if r.boxes is None:
                continue
            boxes = [tuple(b.xyxy[0].tolist()) for b in r.boxes]
            dets += len(boxes)
            nested += nested_pairs(boxes)
            for b in r.boxes:
                if b.id is not None:
                    track_lengths[int(b.id[0])] += 1
            kp_fracs.extend(keypoint_fractions(r))
    cap.release()

    lengths = sorted(track_lengths.values(), reverse=True)
    return {
        "model": model_path,
        "frames": processed,
        "dets": dets,
        "ids": len(lengths),
        "lengths": lengths,
        "meanlen": (statistics.mean(lengths) if lengths else 0.0),
        # A track must survive long enough to yield >= min_tracklet_observations
        # embeddings. At interval_sec 0.4 and ~20 fps that is one embedding per
        # ~8 frames, so 3 observations needs roughly 24 frames of continuous
        # tracking. Shorter tracks are DELETED by reconcile, not merely weak --
        # which is why this column matters more than the raw id count.
        "usable": sum(1 for L in lengths if L >= 24),
        "nested": nested,
        "kp": kp_fracs,
        "ms": (time.time() - t0) * 1000.0 / max(1, processed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--clips", default=".",
                    help="directory holding ._live_src_<cam>.mp4")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--imgsz", default="",
                    help="comma list, e.g. 640,1280. Empty = model default (640)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = ([int(s) for s in args.imgsz.split(",") if s.strip()]
             if args.imgsz else [None])
    clips = sorted(glob.glob(os.path.join(args.clips, "._live_src_*.mp4")))
    if not clips:
        raise SystemExit(
            f"[compare] no ._live_src_*.mp4 in {os.path.abspath(args.clips)}.\n"
            f"          These are written when live.reconcile.keep_frames is true.\n"
            f"          Point --clips at a saved run directory if you copied them aside.")

    print(f"[compare] {len(clips)} clip(s), {len(models)} model(s), "
          f"imgsz {sizes}, first {args.frames} frames each")
    print(f"[compare] conf={args.conf} iou={args.iou} tracker={args.tracker}")
    print("[compare] NOTE iou is inert on an NMS-free (YOLO26) default path -- do "
          "not read an end-to-end result as an NMS result.")

    for clip in clips:
        cam = os.path.basename(clip)[len("._live_src_"):-4]
        print(f"\n{'=' * 78}\n{cam}   ({clip})\n{'=' * 78}")
        print(f"  {'model':<20}{'imgsz':>7}{'dets':>8}{'ids':>6}"
              f"{'meanlen':>9}{'usable':>8}{'nested':>8}{'ms/f':>7}")
        print(f"  {'-' * 74}")

        rows = []
        for model_path in models:
            for size in sizes:
                st = run_one(model_path, clip, args.frames, size,
                             args.conf, args.iou, args.tracker)
                if st is None:
                    continue
                rows.append((size, st))
                print(f"  {os.path.basename(model_path):<20}"
                      f"{(size or 640):>7}{st['dets']:>8}{st['ids']:>6}"
                      f"{st['meanlen']:>9.1f}{st['usable']:>8}"
                      f"{st['nested']:>8}{st['ms']:>7.0f}")

        for size, st in rows:
            if st["lengths"]:
                print(f"    {os.path.basename(st['model'])}@{size or 640} "
                      f"track lengths: {st['lengths'][:8]}")

        # ---- visibility distribution, pose models only ----
        posed = [(s, st) for s, st in rows if st["kp"]]
        if posed:
            print(f"\n  VISIBLE-KEYPOINT FRACTION (COCO-17, joint conf >= 0.5)")
            print(f"  Reference: seated at a desk = 13/17 = 0.76; standing behind")
            print(f"  a desk = 15/17 = 0.88. A floor ABOVE ~0.55 cuts into the")
            print(f"  seated population, which in this deployment is the norm.")
            print(f"    {'model':<20}{'n':>7}{'p05':>7}{'p25':>7}"
                  f"{'p50':>7}{'p75':>7}" + "".join(f"{f'<{g}':>8}" for g in GATE_FLOORS))
            for size, st in posed:
                v = np.asarray(st["kp"])
                cuts = "".join(f"{100.0 * float((v < g).mean()):>7.1f}%"
                               for g in GATE_FLOORS)
                print(f"    {os.path.basename(st['model']):<20}{len(v):>7}"
                      f"{np.percentile(v, 5):>7.2f}{np.percentile(v, 25):>7.2f}"
                      f"{np.percentile(v, 50):>7.2f}{np.percentile(v, 75):>7.2f}"
                      + cuts)
            print(f"  The <X columns are the fraction of DETECTIONS a floor at X")
            print(f"  would reject. Judge it against observations-per-tracklet:")
            print(f"  reconcile DELETES a tracklet under min_tracklet_observations,")
            print(f"  so a gate that halves a short track does not weaken it, it")
            print(f"  removes the person.")

    print(f"\n{'=' * 78}\nHOW TO READ THIS\n{'=' * 78}")
    print("""  RECALL IS THE VETO. If a pose model finds fewer people than yolo11m on the
  same frames, stop -- nothing downstream can recover a box that was never
  produced, and no threshold or merge rule reaches it.

  `usable` MATTERS MORE THAN `ids`. A model that finds more people in
  two-frame flickers is worse than one that finds fewer people properly:
  reconcile suppresses anything under min_tracklet_observations, and the
  operator's cam_213 failure was a 6-observation tracklet that matched nothing
  at any bar. Compare `usable` and `meanlen` together, never the id count
  alone (CLAUDE.md section 4).

  `nested` IS THE ONE TO WATCH ON YOLO26. Those are boxes sitting inside other
  boxes -- invisible to IoU NMS at any setting, and the mechanism that puts two
  track ids on one body. If the NMS-free head drops this to ~0 while recall
  holds, that is a structural fix rather than a tuning one.

  VISIBILITY IS A DISTRIBUTION, NOT A THRESHOLD. Read where the mass sits
  before choosing a floor. If p25 is already 0.76, the population is seated and
  a 0.6 floor is cutting healthy crops.

  AND THIS DECIDES WHICH MODEL TO CAPTURE WITH, NOT WHETHER IDS IMPROVE. That
  still needs a run and a watch.""")


if __name__ == "__main__":
    raise SystemExit(main())