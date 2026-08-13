"""
render_visibility.py  --  SEE what a visibility floor throws away.

    python tests/calibration/render_visibility.py
    python tests/calibration/render_visibility.py --floors 0,0.25,0.5,0.6
    python tests/calibration/render_visibility.py --clips clips_20260805_115845 --kp-conf 0.25

Runs a *-pose detector over the WHOLE clip once, then writes one video per floor
with every box drawn GREEN (kept) or RED (rejected), its visible-keypoint
fraction printed on it, and the joints themselves marked so you can see WHY a
crop scored what it did.

WHY THIS EXISTS RATHER THAN rerender_from_clips.py --min-visible-kp. That flag
filters observations already in Qdrant, so it needs a run captured WITH the pose
detector -- `visible_kp` cannot be recovered for a run that never recorded
keypoints. This works on any clip, because it recomputes detection from pixels.

WHAT IT SHOWS AND WHAT IT DOES NOT. It shows which CROPS a floor would refuse to
embed. It does NOT show the resulting identities -- that needs a capture, a
reconcile and rerender_from_clips.py. Judge a floor here on "are the rejected
crops actually bad", then judge the identities there.

THE COST THIS PRINTS, and it is the number that decides the floor. Rejecting
crops does not weaken a tracklet gently: reconcile SUPPRESSES any tracklet under
min_tracklet_observations, so a gate that halves a short track deletes the
person. So the summary simulates the embedding throttle (reid.interval_sec) and
reports how many tracks would fall under the floor of 3 -- which is a different
and much harsher quantity than "percent of detections dropped".

READ-ONLY: no store, no config writes.
"""

import argparse
import glob
import os
from collections import defaultdict

import cv2
import numpy as np

NUM_KEYPOINTS = 17

GREEN = (80, 220, 80)
RED = (60, 60, 235)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def keypoint_fraction(row, kp_conf):
    """Fraction of COCO-17 joints on one detection scoring >= kp_conf.

    None when the head reported no confidence channel -- there is nothing to
    threshold then, and returning 1.0 would make every crop look perfect while
    measuring nothing.
    """
    if row is None or row.ndim != 2 or row.shape[1] < 3:
        return None
    return float((row[:, 2] >= kp_conf).sum()) / NUM_KEYPOINTS


def detect_clip(model_path, clip, imgsz, conf, iou, tracker, kp_conf, limit):
    """One pass of the detector over the clip. -> (per_frame, fps, size).

    per_frame[i] = [(box, track_id, visible_kp, keypoints), ...]

    Cached so the floors below can be drawn without re-running the model: the
    detector is the expensive part and its output does not depend on the floor.
    """
    from ultralytics import YOLO
    model = YOLO(model_path)

    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise SystemExit(f"[vis] cannot open {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    kwargs = dict(classes=[0], conf=conf, iou=iou, tracker=tracker,
                  persist=True, verbose=False)
    if imgsz:
        kwargs["imgsz"] = imgsz

    per_frame = []
    idx = 0
    while limit <= 0 or idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        rows = []
        for r in model.track(frame, **kwargs):
            if r.boxes is None:
                continue
            kp_all = None
            kp = getattr(r, "keypoints", None)
            if kp is not None and getattr(kp, "data", None) is not None:
                try:
                    kp_all = kp.data.cpu().numpy()
                except AttributeError:
                    kp_all = np.asarray(kp.data)
                if kp_all.shape[0] != len(r.boxes):
                    kp_all = None       # pairing untrustworthy -> no keypoints
            for i, b in enumerate(r.boxes):
                box = tuple(int(round(v)) for v in b.xyxy[0].tolist())
                tid = int(b.id[0]) if b.id is not None else None
                kpr = None if kp_all is None else kp_all[i]
                rows.append((box, tid, keypoint_fraction(kpr, kp_conf), kpr))
        per_frame.append(rows)
        if idx % 300 == 0:
            print(f"    ...{idx} frames")
    cap.release()
    return per_frame, fps, size


def draw(frame, rows, floor, kp_conf, header):
    for box, tid, vk, kpr in rows:
        x1, y1, x2, y2 = box
        rejected = (floor > 0.0 and vk is not None and vk < floor)
        colour = RED if rejected else GREEN
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # The joints themselves, so a low score is explicable rather than just
        # low: a filled dot passed kp_conf, a hollow one did not. Legs missing on
        # a seated person looks completely different from a whole body being
        # uncertain, and only one of those is what a visibility gate should
        # refuse.
        if kpr is not None and kpr.shape[1] >= 3:
            for kx, ky, kc in kpr:
                if kx <= 0 and ky <= 0:
                    continue
                ok = kc >= kp_conf
                cv2.circle(frame, (int(kx), int(ky)), 3,
                           GREEN if ok else GREY, -1 if ok else 1)

        label = f"{'' if tid is None else f'T{tid} '}" + (
            "kp n/a" if vk is None else f"{vk:.2f}")
        if rejected:
            label += " DROP"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 3, max(th, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), BLACK, -1)
    cv2.putText(frame, header, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    return frame


def cost_report(per_frame, floor, fps, interval_sec, min_obs):
    """What this floor costs in TRACKLETS, not in detections.

    Percent-of-detections-dropped is the wrong unit: reconcile deletes a tracklet
    that falls under min_tracklet_observations, so the same 40% cut is free on a
    long track and fatal on a short one. This simulates the embedding throttle --
    one attempt every interval_sec -- and counts how many tracks survive it with
    and without the floor.
    """
    step = max(1, int(round(interval_sec * fps)))
    before = defaultdict(int)
    after = defaultdict(int)
    kept = dropped = unmeasured = 0
    for i, rows in enumerate(per_frame):
        sampled = (i % step == 0)
        for _box, tid, vk, _kp in rows:
            if vk is None:
                unmeasured += 1
            elif floor > 0.0 and vk < floor:
                dropped += 1
            else:
                kept += 1
            if tid is None or not sampled:
                continue
            before[tid] += 1
            if vk is None or floor <= 0.0 or vk >= floor:
                after[tid] += 1
    survived_before = sum(1 for n in before.values() if n >= min_obs)
    survived_after = sum(1 for n in after.values() if n >= min_obs)
    total = kept + dropped
    return {
        "kept": kept, "dropped": dropped, "unmeasured": unmeasured,
        "pct": (100.0 * dropped / total if total else 0.0),
        "tracks": len(before),
        "survived_before": survived_before,
        "survived_after": survived_after,
        "lost": survived_before - survived_after,
        "step": step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--clips", default=".")
    ap.add_argument("--floors", default="0,0.25,0.5,0.6")
    ap.add_argument("--kp-conf", type=float, default=0.5,
                    help="joint confidence floor for counting a keypoint visible")
    ap.add_argument("--imgsz", type=int, default=0, help="0 = model default (640)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--frames", type=int, default=0, help="0 = the WHOLE clip")
    ap.add_argument("--interval-sec", type=float, default=0.4,
                    help="reid.interval_sec, for the tracklet-cost simulation")
    ap.add_argument("--min-obs", type=int, default=3,
                    help="reconcile.min_tracklet_observations")
    ap.add_argument("--out", default="vis_{cam}_f{floor}.mp4")
    args = ap.parse_args()

    floors = [float(f) for f in args.floors.split(",") if f.strip()]
    clips = sorted(glob.glob(os.path.join(args.clips, "._live_src_*.mp4")))
    if not clips:
        raise SystemExit(f"[vis] no ._live_src_*.mp4 in {os.path.abspath(args.clips)}")

    print(f"[vis] {args.model}  imgsz={args.imgsz or 640}  conf={args.conf}  "
          f"kp_conf={args.kp_conf}")
    print(f"[vis] floors {floors} over {len(clips)} clip(s), "
          f"{'whole clip' if not args.frames else str(args.frames) + ' frames'}")

    for clip in clips:
        cam = os.path.basename(clip)[len("._live_src_"):-4]
        print(f"\n=== {cam} ===")
        print("  detecting (one pass, reused for every floor)...")
        per_frame, fps, size = detect_clip(
            args.model, clip, args.imgsz, args.conf, args.iou,
            args.tracker, args.kp_conf, args.frames)
        print(f"  {len(per_frame)} frames @ {fps:.1f} fps")

        for floor in floors:
            tag = f"{int(round(floor * 100)):03d}"
            out_path = args.out.format(cam=cam, floor=tag)
            # mp4v, not h264: this OpenCV build has no usable h264 encoder and
            # falls back anyway, noisily. Skip the noise.
            writer = cv2.VideoWriter(out_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            cap = cv2.VideoCapture(clip)
            c = cost_report(per_frame, floor, fps, args.interval_sec, args.min_obs)
            header = (f"{cam}  floor={floor:.2f}  kp_conf={args.kp_conf:.2f}  "
                      f"dropped {c['pct']:.0f}% of crops  "
                      f"tracklets {c['survived_after']}/{c['survived_before']}")
            for rows in per_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(draw(frame, rows, floor, args.kp_conf, header))
            cap.release()
            writer.release()
            print(f"  floor {floor:.2f} -> {out_path}")
            print(f"      crops: kept {c['kept']}, dropped {c['dropped']} "
                  f"({c['pct']:.1f}%), unmeasured {c['unmeasured']}")
            print(f"      tracklets surviving min_obs={args.min_obs} at "
                  f"1 embedding / {c['step']} frames: "
                  f"{c['survived_after']} of {c['survived_before']}"
                  + (f"   <-- LOSES {c['lost']}" if c["lost"] else ""))

    print("""
==============================================================================
WHAT TO LOOK FOR
==============================================================================
  GREEN box = the crop would be embedded. RED = the floor refuses it.
  Filled dot = that joint cleared kp_conf. Hollow = it did not.

  1. ARE THE RED BOXES ACTUALLY BAD? That is the whole question. A person
     clipped at the frame edge, or half behind a desk, SHOULD be red. A person
     standing in clear view at the far end of the room should NOT be -- and if
     they are, the metric is reading distance rather than occlusion, and no
     value of this floor is the right fix.

  2. IS ANYONE RED FOR THEIR WHOLE WALK? A crop rejected here and there costs
     nothing. A person red end to end is a person the gallery never sees.

  3. THEN READ `tracklets surviving`, NOT the dropped percentage. Reconcile
     deletes a tracklet under min_tracklet_observations rather than weakening
     it, so a floor that drops 40% of crops evenly is free while one that drops
     15% concentrated on short tracks removes people.

  THIS DOES NOT SHOW IDENTITIES. Whether ids improve needs a capture with the
  pose detector, then rerender_from_clips.py --min-visible-kp at the chosen
  floor, watched (CLAUDE.md section 4).""")


if __name__ == "__main__":
    raise SystemExit(main())