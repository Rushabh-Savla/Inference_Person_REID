"""
Add floor positions to a run that was captured BEFORE its calibration existed.

    python tools/backfill_geometry.py <run_id>
    python tools/backfill_geometry.py <run_id> --dry-run     # report, write nothing

=========================== WHY THIS EXISTS ==================================

A floor frame is fitted FROM a run (tools/fit_floor_frame.py), so the first run on
any deployment is necessarily captured before any calibration exists. Without this
tool that run could never carry positions, and testing the reachability veto would
need a SECOND capture -- people walking a room again for five minutes, to learn
something the first run already contains.

It does not need one, because the first run already stored everything a position is
made of: `bbox` and `ts` have been in every observation payload since the live
reconcile landed. The foot point is bottom-centre of the box, so applying the
calibration is a matrix multiply over rows already in Qdrant. No frames, no
detection, no cameras.

===================== HOW THIS RESPECTS INVARIANT 1 ==========================

The rule (geometry/__init__.py) is that OFFLINE RECONCILE never computes geometry --
it consumes positions recorded for it. This tool does not weaken that: it is an
explicit, one-off, operator-run migration that *becomes* the recorded geometry, and
reconcile still only ever reads `payload["floor"]`. Afterwards the run behaves
exactly like one captured with `geometry.enabled: true`.

What would break the invariant is reconcile doing this implicitly, per-run,
mid-decision -- because then re-fitting a calibration would silently change a
finished run's identities with nothing to say why. Here the change is deliberate,
logged, and stamped with the `calib_version` responsible, so two reconciles of the
run still agree unless someone runs this again on purpose.

============================== HONEST CAVEAT =================================

Backfilling the SAME run the calibration was fitted from is mildly self-serving: the
fit already saw some of these feet, so the positions will look better here than on
unseen footage. That is fine for the two things this is for -- checking the veto's
plumbing, and watching whether its decisions are right on video -- and it is not a
generalisation test. The first honest generalisation test is the NEXT capture, which
costs nothing extra because `geometry.enabled: true` will record positions live by
then.

WHAT IT TOUCHES. `payload["floor"]` only. Vectors, reid ids, and every other payload
field are left exactly as they are, so a backfill can never change who is who --
only whether the veto has data to judge them with.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import yaml                                                         # noqa: E402

from geometry.calibration import load_calibration                   # noqa: E402
from geometry.floor import FloorFrame, foot_point                    # noqa: E402

BATCH = 256


def header(text):
    print("\n" + "=" * 76)
    print(text)
    print("=" * 76)


def parse_args():
    p = argparse.ArgumentParser(
        description="Add floor positions to an already-captured run.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("run_id")
    p.add_argument("--calibration", default=None,
                   help="record path (default from config.yaml geometry.calibration_path)")
    p.add_argument("--log-path", default=None,
                   help="sidecar jsonl to write (default logs/geometry_<run_id>.jsonl)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change; write nothing")
    p.add_argument("--overwrite", action="store_true",
                   help="replace positions that are already present")
    p.add_argument("--url", default=None, help="Qdrant URL override")
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml")) as f:
        cfg = yaml.safe_load(f) or {}
    gcfg = cfg.get("geometry") or {}

    calib_path = args.calibration or gcfg.get("calibration_path")
    if calib_path and not os.path.isabs(calib_path):
        calib_path = os.path.join(root, calib_path)
    record = load_calibration(calib_path, required=True)
    frame = FloorFrame(record)
    if not frame.enabled:
        raise SystemExit(
            f"[backfill] {calib_path} has no calibrated camera -- nothing to apply.\n"
            f"        Fit one first: python tools/fit_floor_frame.py {args.run_id}")

    header("THE CALIBRATION BEING APPLIED")
    for line in record.summary().splitlines():
        print(f"  {line}")

    from database.store import PersonVectorStore
    store_cfg = cfg.get("store") or {}
    store = PersonVectorStore(
        path=store_cfg.get("path", "qdrant_data"),
        url=args.url or os.environ.get("QDRANT_URL") or store_cfg.get("url") or None,
        api_key=os.environ.get("QDRANT_API_KEY") or None)

    header(f"SCANNING run {args.run_id}")
    todo = []                        # (point_id, camera, track_id, frame, ts, bbox)
    counts = defaultdict(int)
    max_extent = defaultdict(lambda: [0, 0])
    offset = None
    while True:
        pts, offset = store.client.scroll(store.collection, limit=1000,
                                          offset=offset, with_payload=True,
                                          with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            if pl.get("run_id") != args.run_id:
                continue
            cam = pl.get("camera")
            counts[f"total:{cam}"] += 1
            if pl.get("floor") is not None and not args.overwrite:
                counts[f"already:{cam}"] += 1
                continue
            bbox, ts = pl.get("bbox"), pl.get("ts")
            if bbox is None or ts is None:
                counts[f"no_bbox_or_ts:{cam}"] += 1
                continue
            if not frame.is_calibrated(cam):
                counts[f"uncalibrated:{cam}"] += 1
                continue
            ext = max_extent[cam]
            ext[0] = max(ext[0], float(bbox[2]))
            ext[1] = max(ext[1], float(bbox[3]))
            todo.append((p.id, cam, pl.get("track_id"), pl.get("frame"),
                         float(ts), [float(v) for v in bbox]))
        if offset is None:
            break

    cams = sorted({k.split(":", 1)[1] for k in counts})
    for cam in cams:
        print(f"  {cam}: {counts[f'total:{cam}']} observation(s)"
              f"  already positioned {counts[f'already:{cam}']}"
              f"  missing bbox/ts {counts[f'no_bbox_or_ts:{cam}']}"
              f"  uncalibrated {counts[f'uncalibrated:{cam}']}")
    if not counts:
        raise SystemExit(
            f"[backfill] run {args.run_id!r} has no observations in this store. "
            f"Run tools/fit_floor_frame.py {args.run_id} to see the store inventory.")
    if not todo:
        raise SystemExit(
            "[backfill] nothing to do -- every observation is either already "
            "positioned (use --overwrite), missing bbox/ts, or from an "
            "uncalibrated camera.")

    # A homography belongs to the pixel space it was fitted in. We cannot see the
    # frames here, so the boxes are the only evidence of their resolution: the
    # largest box extent is a lower bound on the frame size. A big shortfall means
    # this run was probably captured at another resolution, which would tilt every
    # position with no error anywhere -- so warn loudly rather than proceed quietly.
    header("PIXEL-SPACE CHECK")
    suspicious = False
    for cam in sorted(max_extent):
        want = record.image_size(cam)
        got = max_extent[cam]
        ratio = min(got[0] / want[0], got[1] / want[1]) if want else 0
        note = ""
        if want and ratio < 0.5:
            note = "   <- SUSPICIOUS: boxes never reach half the calibrated frame"
            suspicious = True
        print(f"  {cam}: calibrated for {want}, largest box corner "
              f"({got[0]:.0f}, {got[1]:.0f}){note}")
    if suspicious:
        print("\n  Boxes this small can just mean nobody walked near the frame edge,")
        print("  but they can also mean the run was captured at a DIFFERENT")
        print("  resolution than the calibration was fitted at -- in which case")
        print("  every position below is wrong and nothing says so. Check")
        print("  source.resize_width for the run before trusting the veto.")

    header("APPLYING")
    log_path = args.log_path or (gcfg.get("log_path")
                                 or "logs/geometry_<run_id>.jsonl")
    log_path = log_path.replace("<run_id>", args.run_id)
    if not os.path.isabs(log_path):
        log_path = os.path.join(root, log_path)

    written = skipped = 0
    pending = []
    fh = None
    if not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        fh = open(log_path, "a", buffering=1)

    def flush():
        for point_id, floor in pending:
            store.client.set_payload(collection_name=store.collection,
                                     payload={"floor": floor},
                                     points=[point_id])
        pending.clear()

    for point_id, cam, track_id, frame_index, ts, bbox in todo:
        pos = frame.position(cam, bbox)
        if pos is None:
            skipped += 1
            continue
        floor = pos.as_dict()
        floor["calib_version"] = record.calib_version
        written += 1
        if args.dry_run:
            continue
        pending.append((point_id, floor))
        if len(pending) >= BATCH:
            flush()
        fp = foot_point(bbox)
        fh.write(json.dumps({
            "run_id": args.run_id, "camera": cam,
            "track_id": None if track_id is None else int(track_id),
            "frame": None if frame_index is None else int(frame_index),
            "ts": ts, "bbox": [round(v, 2) for v in bbox],
            "foot_img": [round(fp[0], 2), round(fp[1], 2)] if fp else None,
            "foot_source": pos.source, "clipped": pos.clipped,
            "floor": [round(pos.x, 4), round(pos.y, 4)],
            "floor_error": round(pos.error, 4), "group": pos.group,
            "calib_version": record.calib_version, "units": record.units,
            "backfilled": True,
        }) + "\n")
    if not args.dry_run:
        flush()
        fh.close()

    total = written + skipped
    pct = (100.0 * written / total) if total else 0.0
    print(f"  {written}/{total} observation(s) positioned ({pct:.1f}%), "
          f"{skipped} unavailable")
    if args.dry_run:
        print("\n  --dry-run: NOTHING was written.")
        return 0
    print(f"  payload updated in place; sidecar -> {os.path.relpath(log_path, root)}")
    if written == 0:
        print("\n  !! Nothing was positioned. Almost always the pixel-space check "
              "above.")
        return 1
    print("\n  Next -- turn the veto on for THIS run only and watch the result:")
    print(f"    python tests/calibration/sweep_reconcile_thresholds.py "
          f"{args.run_id} --geometry")
    print(f"    python tests/calibration/rerender_from_clips.py "
          f"{args.run_id} --geometry     # WATCH IT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
