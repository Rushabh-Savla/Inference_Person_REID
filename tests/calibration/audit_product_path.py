"""
End-to-end audit of the PRODUCT path: live pipeline -> offline reconcile -> render.

Produces REMEDIATION_PLAN.md Part H.10. This is the only script here that runs the
real deliverable path, so it is what catches whole-pipeline problems the unit-level
measurements cannot: frame-drop rate, clip/annotation alignment, whether every
tracklet actually received an identity, and whether the output video's timeline is
correct.

    python tests/calibration/audit_product_path.py [video] [seconds] [queue_depth]

Runs entirely in a temp directory with a DISTINCT camera name and a LOCAL Qdrant,
so it can never overwrite a real output_<cam>.mp4 or touch the shared gallery.
Pass a queue depth to A/B the load-shedding (shipped value is 2).
"""

import os
import shutil
import sys
import tempfile

import yaml

from _common import project_root, header

ROOT = project_root()
sys.path.insert(0, ROOT)                       # so `from main import ...` resolves
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.pop("QDRANT_URL", None)             # force the local store
os.environ.pop("QDRANT_API_KEY", None)

import cv2
from collections import defaultdict, Counter

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "test_v2.avi"
SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
QUEUE = int(sys.argv[3]) if len(sys.argv) > 3 else 2

if not os.path.exists(os.path.join(ROOT, VIDEO)):
    raise SystemExit(f"[calib] {VIDEO!r} not found in {ROOT}")


def frames_in(path):
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def video_meta(path):
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps


work = tempfile.mkdtemp(prefix="reid_audit_")
cam = "audit_cam"
try:
    for f in ("yolo11n.pt", "yolo11n-pose.pt"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            os.symlink(p, os.path.join(work, f))
    os.symlink(os.path.join(ROOT, "src"), os.path.join(work, "src"))
    os.symlink(os.path.join(ROOT, VIDEO), os.path.join(work, os.path.basename(VIDEO)))

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    cfg["store"]["url"] = None
    cfg["store"]["path"] = os.path.join(work, "qdrant")
    cfg["live"]["run"]["max_duration_sec"] = SECONDS
    cfg["live"]["run"]["device"] = "cpu"
    cfg["live"]["reconcile"]["keep_frames"] = True
    cfg["live"]["metrics"]["log_interval_sec"] = 0
    cfg["live"]["inference"]["max_inference_queue"] = QUEUE

    src_meta = video_meta(os.path.join(ROOT, VIDEO))
    print(f"[calib] source {VIDEO}: {src_meta[0]} frames @ {src_meta[1]:.1f} fps")
    print(f"[calib] running product path: cpu, max {SECONDS}s, "
          f"max_inference_queue={QUEUE}\n")

    cwd = os.getcwd()
    os.chdir(work)
    try:
        from live.pipeline import LivePipeline
        pipe = LivePipeline([(cam, os.path.basename(VIDEO))], cfg)
        pipe.run()
    finally:
        os.chdir(cwd)

    renderer = pipe.renderers[0] if pipe.renderers else None
    capt = pipe.captures[0]
    clip = os.path.join(work, f"._live_src_{cam}.mp4")
    out = os.path.join(work, f"output_{cam}.mp4")

    header("A. FRAME BUDGET")
    n_annos = len(renderer.annotations) if renderer else 0
    rendered = renderer.rendered if renderer else 0
    nclip, nout = frames_in(clip), frames_in(out)
    dropped = 100.0 * (1 - rendered / max(1, capt.frame_index))
    print(f"  captured from source        : {capt.frame_index}")
    print(f"  reached the render stage    : {rendered}")
    print(f"  => DISCARDED before render  : {dropped:.1f}%")
    print(f"  annotation entries recorded : {n_annos}")
    print(f"  frames in temp clip         : {nclip}")
    print(f"  frames in output video      : {nout}")

    header("B. ALIGNMENT INVARIANT (annotations must equal clip frames)")
    if nclip is None:
        print("  temp clip MISSING -> the re-render is impossible")
    elif nclip == n_annos:
        print(f"  OK   {n_annos} annotations == {nclip} clip frames")
        print("  box geometry is correctly paired with pixels")
    else:
        print(f"  *** MISALIGNED *** annotations={n_annos} clip={nclip} "
              f"(diff {n_annos - nclip})")
        print("  every box from the divergence point on is drawn on the WRONG frame")

    header("C. TIMELINE CORRECTNESS   (Part H.10 / issue #46)")
    om = video_meta(out)
    if om and src_meta and src_meta[1] > 0:
        real_seconds = capt.frame_index / src_meta[1]
        out_seconds = om[0] / max(om[1], 1e-9)
        print(f"  real content captured : {real_seconds:.1f}s "
              f"({capt.frame_index} frames @ {src_meta[1]:.1f} fps)")
        print(f"  output video duration : {out_seconds:.2f}s "
              f"({om[0]} frames @ {om[1]:.1f} fps)")
        if out_seconds > 0:
            print(f"  => playback is {real_seconds / out_seconds:.1f}x TOO FAST")
        print("  cause: one clip frame per PROCESSED frame at a fixed fps, ignoring")
        print("  every dropped frame. WriterStage has correct pacing but is never")
        print("  constructed in reconcile mode.")

    header("D. DID EVERY TRACKLET GET AN IDENTITY?")
    per_track, no_gid, total = defaultdict(Counter), defaultdict(int), 0
    offset = None
    while True:
        pts, offset = pipe.store.client.scroll(
            pipe.store.collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            if pl.get("run_id") != pipe.run_id:
                continue
            total += 1
            gid = pl.get("reid_id", pl.get("global_id"))
            key = (pl.get("camera"), pl.get("track_id"))
            if gid is None:
                no_gid[key] += 1
            else:
                per_track[key][gid] += 1
        if offset is None:
            break
    gids = {g for c in per_track.values() for g in c}
    print(f"  stored observations         : {total}")
    print(f"  tracklets WITH an identity  : {len(per_track)}")
    print(f"  tracklets WITHOUT an identity: {len(no_gid)}  {dict(no_gid)}")
    print(f"  final identities            : {len(gids)} -> {sorted(gids)}")
    multi = {k: dict(c) for k, c in per_track.items() if len(c) > 1}
    print(f"  tracklets with >1 identity (must be 0): {len(multi)} {multi}")
    tracks_per_gid = defaultdict(list)
    for k, c in per_track.items():
        tracks_per_gid[c.most_common(1)[0][0]].append(k[1])
    for g, ts in sorted(tracks_per_gid.items()):
        print(f"    identity {g}: {len(ts)} tracklet(s) {sorted(ts)}")
    if no_gid:
        print("\n  Tracklets without an identity render as a bare 'ID <track_id>'")
        print("  (issues #25 / #33). Either min_tracklet_observations suppressed")
        print("  them, or fewer than 2 tracklets survived and reconcile returned early.")

    header("E. DUPLICATE BOXES IN THE DELIVERABLE")
    if renderer and n_annos:
        from _common import iou as _iou
        cap0 = cv2.VideoCapture(clip)
        ok, f0 = cap0.read()
        cap0.release()
        H, W = (f0.shape[:2] if ok else (None, None))
        total_boxes = oob = dup = 0
        for anno in renderer.annotations:
            for d in anno:
                total_boxes += 1
                if W and (d["x1"] < 0 or d["y1"] < 0 or d["x2"] > W or d["y2"] > H):
                    oob += 1
            for i in range(len(anno)):
                for j in range(i + 1, len(anno)):
                    p, q = anno[i], anno[j]
                    if _iou((p["x1"], p["y1"], p["x2"], p["y2"]),
                            (q["x1"], q["y1"], q["x2"], q["y2"])) >= 0.5:
                        dup += 1
        print(f"  clip resolution           : {W}x{H}")
        print(f"  boxes drawn               : {total_boxes}")
        print(f"  boxes outside frame bounds: {oob}")
        print(f"  overlapping pairs IoU>=0.5: {dup}   <- duplicate-box count")

    header("F. DISK COST (issue #65)")
    if nclip and os.path.exists(clip):
        b = os.path.getsize(clip)
        per_frame_kb = b / nclip / 1024
        print(f"  temp clip: {b / 1e6:.1f} MB / {nclip} frames = "
              f"{per_frame_kb:.1f} KB per frame at {W}x{H}")
        if nclip < 50:
            print(f"  *** UNRELIABLE: only {nclip} frames. Container overhead and the")
            print("      first keyframe dominate at this length, so KB/frame is")
            print("      inflated. Re-run with a longer duration for a usable figure")
            print("      (a 157-frame run gave 51 KB/frame at 1920x1080).")
        print("  scaled to the production cameras, assuming NO frames dropped:")
        for label, w, h, fps in (("cam_213 1920x1080@25", 1920, 1080, 25),
                                 ("cam_224 2560x1440@15", 2560, 1440, 15),
                                 ("cam_206 1920x1080@25", 1920, 1080, 25),
                                 ("cam_219 2560x1440@25", 2560, 1440, 25)):
            scale = (w * h) / max(1, (W or 1) * (H or 1))
            mbps = per_frame_kb * 1024 * scale * fps / 1e6
            print(f"    {label}: ~{mbps:.1f} MB/s -> {mbps * 3600 / 1000:.1f} GB/hour")
        print("  No size cap, no rotation, no free-space check. keep_frames: true")
        print("  means these are NOT deleted.")

    header("NOTE ON GENERALISING THESE NUMBERS")
    print("""  A FILE source is read as fast as the CPU allows -- far faster than a
  real 15/25 fps RTSP feed -- so the drop rate here is a WORST CASE and is not
  representative of the GPU box on live streams. What it does establish reliably:
  WHERE frames are shed (the inference queue, not the capture slot), whether the
  alignment invariant holds, whether every tracklet got an id, and the timeline
  error. Use run1.log from a real capture for the true drop rate.""")

finally:
    shutil.rmtree(work, ignore_errors=True)
