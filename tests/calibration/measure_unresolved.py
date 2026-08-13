"""
measure_unresolved.py  --  WHY do visible people render as grey "UNRESOLVED"?

    python tests/calibration/measure_unresolved.py <run_id> [--clips DIR]

THE COMPLAINT THIS EXISTS FOR. The operator watched a re-render and reported that
many plainly visible people carried no identity at all. That is a different failure
from every one this project has been measuring: not a WRONG id, an ABSENT one. And it
is upstream of every threshold, scoring mode, veto and camera-bias question, because a
person who never reaches the store is invisible to all of them -- including to every
calibration script, which reads stored observations.

HOW A BOX ENDS UP GREY. The final video draws boxes from the annotations SIDECAR,
which holds every tracked detection the live pass saw. The id comes from `gid_map`,
built from the STORE. Anything in the first and not the second renders as UNRESOLVED
(main.py: `unresolved=(gid_map.get((name, track_id)) is None)`; drawing.py prints the
word and a neutral grey, deliberately not a palette colour). Three ways to fall in
the gap, and only the third is logged anywhere today:

  1. track_id is None -- ByteTrack had not confirmed the box yet. TrackEmbedder skips
     it (`if det.track_id is None: continue`), so nothing is ever embedded or stored.
  2. every crop was REJECTED by the quality gate -- min_height / min_area /
     min_box_area_ratio / min_aspect / MAX_ASPECT / min_blur / brightness /
     max_occlusion_ratio. reid/service.py records these in `last_quality_rejected`
     and NOTHING READS IT: no counter, no per-camera breakdown, nothing in the run
     summary. A person can be on screen for ten seconds with every crop rejected and
     leave no trace at all.
  3. fewer than `min_tracklet_observations` stored observations -- reconcile clears
     the id (this one does print a line per tracklet).

WHAT THE OUTPUT IS FOR. Cause 1 and 3 are cheap and mostly benign (brief blips).
Cause 2 is the dangerous one, so the second table separates "a track that existed for
three frames" from "a track that was on screen for ten seconds and produced nothing".
The second is a real person the pipeline threw away, and the duration column is how
you tell.

NEEDS NO MODEL. Sidecar JSON plus a Qdrant payload scan. Run it on the box that holds
the clips (the A6000); point --clips at their directory if not the repo root.
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

from _common import bootstrap, header

ROOT = bootstrap()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "persons")


def load_sidecar(path):
    """-> (annotations, frame_ts, run_id). Accepts a bare list or a dict wrapper,
    because the sidecar's shape has changed once already and a hard assumption here
    would make this script fail on exactly the old runs it is most useful for."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        annos = raw.get("annotations") or raw.get("frames") or []
        return annos, raw.get("frame_ts") or [], raw.get("run_id")
    return raw, [], None


def store_tracklets(run_id):
    """(camera, track) -> (n_observations, assigned_gid_or_None)."""
    import requests
    obs = collections.Counter()
    gid = {}
    offset = None
    while True:
        body = {"limit": 1000, "with_vector": False,
                "with_payload": ["camera", "track_id", "reid_id", "global_id"],
                "filter": {"must": [{"key": "run_id", "match": {"value": run_id}}]}}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                          json=body, timeout=300)
        r.raise_for_status()
        res = r.json()["result"]
        for p in res["points"]:
            pl = p["payload"]
            k = (pl.get("camera"), int(pl.get("track_id")))
            obs[k] += 1
            g = pl.get("reid_id", pl.get("global_id"))
            if g is not None:
                gid[k] = int(g)
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return {k: (n, gid.get(k)) for k, n in obs.items()}


# EVERY gate in reid/service.py, because all of them are reproducible here.
#
# An earlier version of this file excluded min_blur and brightness, claiming they
# "need pixels". They do -- and the pixels are in the clip, which is sitting next to
# the sidecar. Worse, the argument offered for skipping them ("Laplacian variance
# fluctuates frame to frame, so neither can hold for a whole track") is true of BLUR
# and FALSE OF BRIGHTNESS: a STATIC object in shadow, or against a window, has
# constant brightness, so min_brightness/max_brightness can reject all of its frames.
# For the case this script exists to explain -- a track that never stored anything --
# brightness is therefore a leading candidate, not an excluded one.
GEO_GATES = ("too_narrow", "too_short", "too_small", "too_tiny_in_frame",
             "bad_aspect_lo", "bad_aspect_hi", "occluded")
PIXEL_GATES = ("blurry", "bad_brightness")
ALL_GATES = GEO_GATES + PIXEL_GATES
PIXEL_SAMPLES = 24            # frames sampled per track; the gates are per-frame


def load_quality_cfg():
    """reid.quality as it SHIPS. Read from config.yaml so this can never drift from
    the gate that actually ran."""
    defaults = {"min_width": 24, "min_height": 64, "min_area": 2500,
                "min_box_area_ratio": 0.002, "min_aspect": 0.20,
                "max_aspect": 1.20, "max_occlusion_ratio": 1.0,
                "min_blur": 20.0, "min_brightness": 20.0, "max_brightness": 235.0}
    try:
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        q = ((cfg.get("reid") or {}).get("quality") or {})
        return {k: q.get(k, v) for k, v in defaults.items()}
    except Exception as e:                                        # noqa: BLE001
        print(f"  (could not read reid.quality from config.yaml: {e}; using "
              f"defaults)")
        return defaults


def clip_size(clip_dir, cam):
    """(width, height) of a camera's clip, for the box_area_ratio gate. One frame
    probe, no model."""
    import cv2
    path = os.path.join(clip_dir, f"._live_src_{cam}.mp4")
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return (w, h) if w > 0 and h > 0 else None


def _occlusion(mine, others):
    """Largest fraction of `mine` covered by another box -- verbatim the rule in
    reid/service.py::_occlusion_ratio."""
    x1, y1, x2, y2 = mine
    area = max(1, (x2 - x1) * (y2 - y1))
    worst = 0.0
    for ox1, oy1, ox2, oy2 in others:
        iw = min(x2, ox2) - max(x1, ox1)
        ih = min(y2, oy2) - max(y1, oy1)
        if iw > 0 and ih > 0:
            worst = max(worst, (iw * ih) / area)
    return worst


def box_motion(boxes):
    """Std-dev of the box CENTRE across a track, in pixels -- "is this thing moving?"

    The cheapest discriminator between a person and a static false positive (a coat
    on a chair, a poster, a reflection), and it needs no pixels at all. A person
    walking through a 2560-wide frame moves hundreds of pixels; a stationary object
    is near zero. A track that spans an ENTIRE clip with near-zero motion is almost
    certainly not a person, which changes what the quality gate result even means.
    """
    if not boxes:
        return None
    cx = [((b[0][0] + b[0][2]) / 2.0) for b in boxes]
    cy = [((b[0][1] + b[0][3]) / 2.0) for b in boxes]
    return float(np.hypot(np.std(cx), np.std(cy)))


def pixel_gate_failures(clip_dir, cam, track_boxes, q, samples=PIXEL_SAMPLES):
    """-> {blurry: frac, bad_brightness: frac} by decoding the clip.

    `track_boxes` is [(frame_index, (x1,y1,x2,y2)), ...]. Reproduces
    reid/service.py::_crop_quality's blur and brightness terms exactly: Laplacian
    variance and mean gray of the BGR crop. Returns None values when the clip cannot
    be read, so a missing clip degrades the row rather than the script.
    """
    import cv2
    path = os.path.join(clip_dir, f"._live_src_{cam}.mp4")
    if not os.path.exists(path) or not track_boxes:
        return {g: None for g in PIXEL_GATES}
    idx = np.linspace(0, len(track_boxes) - 1,
                      min(samples, len(track_boxes))).round().astype(int)
    want = {int(track_boxes[i][0]): track_boxes[i][1] for i in idx}
    cap = cv2.VideoCapture(path)
    blur_fail = bright_fail = seen = 0
    try:
        for fi in sorted(want):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = want[fi]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            seen += 1
            if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < q["min_blur"]:
                blur_fail += 1
            b = float(gray.mean())
            if not (q["min_brightness"] <= b <= q["max_brightness"]):
                bright_fail += 1
    finally:
        cap.release()
    if not seen:
        return {g: None for g in PIXEL_GATES}
    return {"blurry": blur_fail / seen, "bad_brightness": bright_fail / seen}


def gate_failures(boxes, q, frame_size):
    """-> {gate: fraction of boxes it would reject}. None for a gate that cannot be
    evaluated (box_area_ratio without the frame size)."""
    n = len(boxes)
    fails = {g: 0 for g in GEO_GATES}
    frame_area = (frame_size[0] * frame_size[1]) if frame_size else None
    for mine, others, _fi in boxes:
        x1, y1, x2, y2 = mine
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect = w / max(1, h)
        if w < q["min_width"]:
            fails["too_narrow"] += 1
        if h < q["min_height"]:
            fails["too_short"] += 1
        if area < q["min_area"]:
            fails["too_small"] += 1
        if frame_area and (area / frame_area) < q["min_box_area_ratio"]:
            fails["too_tiny_in_frame"] += 1
        if aspect < q["min_aspect"]:
            fails["bad_aspect_lo"] += 1
        if aspect > q["max_aspect"]:
            fails["bad_aspect_hi"] += 1
        if _occlusion(mine, others) > q["max_occlusion_ratio"]:
            fails["occluded"] += 1
    out = {g: (c / n if n else 0.0) for g, c in fails.items()}
    if not frame_area:
        out["too_tiny_in_frame"] = None
    return out


NO_TRACK = "no track_id (ByteTrack unconfirmed)"
NOT_STORED = "never reached the store (quality gate / throttle)"
CLEARED = "stored but id CLEARED (< min_tracklet_observations)"
RESOLVED = "resolved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--clips", default=".")
    ap.add_argument("--long-track-sec", type=float, default=2.0,
                    help="a never-stored track visible longer than this is a person "
                         "the pipeline threw away, not a blip")
    args = ap.parse_args()

    store = store_tracklets(args.run_id)
    if not store:
        print(f"[calib] run {args.run_id}: no stored observations at "
              f"{QDRANT_URL} -- wrong host, or the run persisted nothing.")
        return 1
    sidecars = sorted(glob.glob(os.path.join(args.clips,
                                             "._live_src_*.annotations.json")))
    if not sidecars:
        print(f"[calib] no ._live_src_*.annotations.json under {args.clips!r}. "
              f"Runs captured before the sidecar existed cannot be measured; "
              f"keep_frames must have been true.")
        return 1

    print(f"[calib] run {args.run_id}  qdrant={QDRANT_URL}  clips={args.clips}")
    header("A. EVERY PERSON-BOX IN THE FINAL VIDEO, BY WHETHER IT CARRIES AN ID")
    print(f"  {'camera':<10}{'frames':>7}{'boxes':>8}{'resolved':>10}"
          f"{'no track_id':>13}{'not stored':>12}{'cleared':>9}{'UNRESOLVED':>12}")
    grand = collections.Counter()
    per_cam_tracks = {}
    for sc in sidecars:
        cam = os.path.basename(sc)[len("._live_src_"):-len(".annotations.json")]
        annos, fts, clip_run = load_sidecar(sc)
        # HARD REFUSAL on a cross-run join. Clip filenames carry no run_id --
        # `._live_src_<cam>.mp4` is overwritten by every run while keep_frames
        # preserves it -- so the files on disk belong to the LAST run. Joining an
        # earlier run's store against them compares two disjoint ByteTrack id
        # spaces, which reads as a total identity failure: measured on 064551 vs
        # 120409's clips, cam_219 came out 100% UNRESOLVED and the headline was
        # 84% of all boxes. That number was a join error, not a pipeline failure.
        if clip_run is not None and clip_run != args.run_id:
            print(f"  REFUSED {os.path.basename(sc)}: sidecar is from run "
                  f"{clip_run}, not {args.run_id}. ByteTrack renumbers every run, "
                  f"so joining them would report a meaningless UNRESOLVED rate. "
                  f"Measure run {clip_run} instead, or point --clips at a copy of "
                  f"{args.run_id}'s footage.")
            continue
        if clip_run is None:
            print(f"  {os.path.basename(sc)}: NO run_id in the sidecar -- cannot "
                  f"verify it belongs to {args.run_id}. Check its frame count "
                  f"against the run's stored frame span before trusting this.")
        counts = collections.Counter()
        # per-track: first/last frame index, heights, and the FULL boxes plus the
        # other boxes present in the same frame -- the latter is what makes the
        # occlusion gate reproducible without re-running the detector.
        seen = collections.defaultdict(
            lambda: {"first": None, "last": None, "h": [], "boxes": []})
        for fi, raw_boxes in enumerate(annos):
            parsed = []
            for b in raw_boxes:
                if isinstance(b, dict):
                    tid = b.get("track_id")
                    x1, y1, x2, y2 = b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")
                else:
                    x1, y1, x2, y2, tid = b[0], b[1], b[2], b[3], b[4]
                parsed.append((tid, x1, y1, x2, y2))
            others = [p[1:] for p in parsed if None not in p[1:]]
            for tid, x1, y1, x2, y2 in parsed:
                if tid is None:
                    counts[NO_TRACK] += 1
                    continue
                tid = int(tid)
                rec = seen[tid]
                rec["first"] = fi if rec["first"] is None else rec["first"]
                rec["last"] = fi
                if None not in (x1, y1, x2, y2):
                    rec["h"].append(int(y2) - int(y1))
                    mine = (int(x1), int(y1), int(x2), int(y2))
                    # (box, co-present boxes, FRAME INDEX) -- the index is what lets
                    # the pixel gates seek the clip to this exact frame.
                    rec["boxes"].append((mine, [o for o in others if o != mine], fi))
                entry = store.get((cam, tid))
                if entry is None:
                    counts[NOT_STORED] += 1
                elif entry[1] is None:
                    counts[CLEARED] += 1
                else:
                    counts[RESOLVED] += 1
        boxes = sum(counts.values())
        unres = boxes - counts[RESOLVED]
        grand.update(counts)
        per_cam_tracks[cam] = (seen, fts, len(annos))
        print(f"  {cam:<10}{len(annos):>7}{boxes:>8}{counts[RESOLVED]:>10}"
              f"{counts[NO_TRACK]:>13}{counts[NOT_STORED]:>12}{counts[CLEARED]:>9}"
              f"{(unres / boxes if boxes else 0):>11.0%}")
    tot = sum(grand.values())
    if tot:
        print(f"\n  ALL CAMERAS: {tot} person-boxes drawn, "
              f"{tot - grand[RESOLVED]} ({(tot - grand[RESOLVED]) / tot:.0%}) carry "
              f"NO identity.")
        for cause in (NO_TRACK, NOT_STORED, CLEARED):
            if grand[cause]:
                print(f"    {grand[cause] / tot:5.1%}  {cause}")

    header("B. THE TRACKS THAT NEVER REACHED THE STORE -- blip, or a real person?")
    print("  A never-stored track that was on screen for a long time is a person the")
    print("  QUALITY GATE threw away. reid/service.py records the rejection reason in")
    print("  `last_quality_rejected` and nothing reads it, so the reason is not")
    print("  recoverable after the fact -- only the duration is.\n")
    print(f"  {'camera':<10}{'never-stored tracks':>20}{'of which > ' + str(args.long_track_sec) + 's':>18}"
          f"{'longest (s)':>13}{'median box h':>14}")
    worst = []
    for cam, (seen, fts, nframes) in sorted(per_cam_tracks.items()):
        fps = None
        if fts and len(fts) > 1 and fts[0] is not None and fts[-1] is not None:
            span = fts[-1] - fts[0]
            fps = (len(fts) - 1) / span if span > 0 else None
        never = [(t, r) for t, r in seen.items() if (cam, t) not in store]
        durs = []
        for t, r in never:
            nf = (r["last"] - r["first"] + 1)
            durs.append((nf / fps if fps else float("nan"), t, r))
        durs.sort(reverse=True)
        long_ones = [d for d in durs if d[0] == d[0] and d[0] > args.long_track_sec]
        hs = [int(np.median(r["h"])) for _, _, r in durs if r["h"]]
        print(f"  {cam:<10}{len(never):>20}{len(long_ones):>18}"
              f"{(durs[0][0] if durs else float('nan')):>13.1f}"
              f"{(int(np.median(hs)) if hs else 0):>14}")
        worst.extend((cam, d[0], d[1], d[2]) for d in durs[:5])
    if worst:
        worst.sort(key=lambda x: -(x[1] if x[1] == x[1] else 0))
        print("\n  Longest never-stored tracks (scrub the clip to these frames and look):")
        print(f"    {'camera':<10}{'track':>7}{'sec':>7}{'frames':>16}{'median box h':>14}")
        for cam, sec, t, r in worst[:12]:
            span = "{}-{}".format(r["first"], r["last"])
            med_h = int(np.median(r["h"])) if r["h"] else 0
            print(f"    {cam:<10}{t:>7}{sec:>7.1f}{span:>16}{med_h:>14}")

    header("C. ATTRIBUTION -- replay EVERY gate on the never-stored tracks")
    print("  All seven gates are reproducible: the geometric ones from the sidecar,")
    print("  blur and brightness by decoding the clip. No instrumentation, no model.")
    print("  `% fail` is the fraction of that track's boxes the gate would reject, and")
    print("  a gate at 100% is a sufficient explanation -- a never-stored track had")
    print("  EVERY crop rejected. `motion` is the std-dev of the box centre in px:")
    print("  near-zero over a long track means a STATIC object, not a person, which")
    print("  changes what any gate result means.\n")
    q = load_quality_cfg()
    print("  gates in force: " + ", ".join(f"{k}={v}" for k, v in sorted(q.items())))
    print()
    print(f"  {'camera':<10}{'track':>6}{'boxes':>7}{'secs':>6}{'motion':>8}"
          + "".join(f"{name:>12}" for name in ALL_GATES))
    for cam, (seen, fts, nframes) in sorted(per_cam_tracks.items()):
        size = clip_size(args.clips, cam)
        if size is None:
            print(f"  {cam}: could not read the clip's resolution -> "
                  f"box_area_ratio and the pixel gates are not checkable")
        fps = None
        if fts and len(fts) > 1 and fts[0] is not None and fts[-1] is not None:
            span = fts[-1] - fts[0]
            fps = (len(fts) - 1) / span if span > 0 else None
        never = [(t, r) for t, r in seen.items() if (cam, t) not in store]
        never.sort(key=lambda tr: -(tr[1]["last"] - tr[1]["first"]))
        for t, r in never:
            boxes = r["boxes"]
            if not boxes:
                continue
            nf = r["last"] - r["first"] + 1
            fails = gate_failures(boxes, q, size)
            fails.update(pixel_gate_failures(
                args.clips, cam, [(b[2], b[0]) for b in boxes], q))
            mot = box_motion(boxes)
            row = (f"  {cam:<10}{t:>6}{len(boxes):>7}"
                   f"{(nf / fps if fps else float('nan')):>6.1f}"
                   f"{(mot if mot is not None else float('nan')):>8.1f}")
            for name in ALL_GATES:
                v = fails.get(name)
                cell = "--" if v is None else f"{v:.0%}"
                if v is not None and v >= 0.999:
                    cell += " *"
                row += f"{cell:>12}"
            print(row)
    print("\n  * = rejects EVERY box of that track, i.e. a sufficient explanation.")
    print("  motion < ~5 px over a track spanning the whole clip => not a person.")

    header("WHAT TO DO WITH THIS")
    print("  * A high `not stored` fraction with LONG durations => the crop-quality")
    print("    gate is rejecting real people. The gate is in reid/service.py; the")
    print("    candidates for an office are max_aspect (1.20 -- a seated person is")
    print("    near-square and gets rejected outright) and min_box_area_ratio /")
    print("    min_height for anyone far from the camera.")
    print("  * A high `no track_id` fraction => ByteTrack is not confirming boxes;")
    print("    that is a tracker/detector question, not an identity one.")
    print("  * A high `cleared` fraction => min_tracklet_observations. Cheapest fix,")
    print("    and the only one already visible in the reconcile log.")
    print("  * LOW `motion` + `bad_aspect_hi` at 100% + a track spanning the whole")
    print("    clip => a STATIC FALSE POSITIVE (furniture, a poster, a reflection)")
    print("    that YOLO keeps calling a person. The quality gate is then behaving")
    print("    CORRECTLY -- nothing enters the gallery -- and the only defect is")
    print("    cosmetic: the renderer still draws a grey UNRESOLVED box around it for")
    print("    the whole video, which reads as a systemic identity failure. Measured")
    print("    on 20260804_120409, ONE such track was 1099 of 1150 unresolved boxes:")
    print("    removing it takes the run from 14.4% unresolved to 0.6%.")
    print("    DO NOT build a motion filter from this. A seated person is also")
    print("    static, and this deployment is an office -- motion alone would reject")
    print("    exactly the people it must keep. Furniture is separated from a seated")
    print("    person by ASPECT (wide vs near-square), not by stillness.")
    print("  * Instrumenting `last_quality_rejected` is still worth doing for future")
    print("    runs, but it is NOT a prerequisite: every gate is reproducible from")
    print("    clip + sidecar, which is how the above was attributed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
