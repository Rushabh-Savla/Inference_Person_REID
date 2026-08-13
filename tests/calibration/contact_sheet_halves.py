#!/usr/bin/env python3
"""
LOOK at the two halves a same-tracklet control score was computed from.

    python tests/calibration/contact_sheet_halves.py --clips .
    python tests/calibration/contact_sheet_halves.py --clips . --tracklets cam_219:14,cam_219:7
    python tests/calibration/contact_sheet_halves.py --clips . --out sheets

WHY. degrade_crops_causal.py measured, on run 20260804_094039, the cosine between
the two halves of ONE ByteTrack track -- provably one person, one camera, one
lighting condition. Those "control" numbers ranged from 0.594 to 0.950:

    cam_219:14  0.594      cam_219:7   0.950
    cam_219:6   0.622      cam_224:37  0.946
    cam_213:19  0.704      cam_219:26  0.898

A 0.35 spread inside single tracks is FIVE TIMES the resolution effect the same
script measured (0.045 at 6x downscale) and three times its blur effect. So
something else dominates, and the leading candidate is that the low-scoring tracks
are the ones where the person TURNED during the track, making half A front-facing
and half B rear-facing -- the mode-change failure config.yaml's `scoring` comment
describes, appearing WITHIN a track rather than across a re-appearance.

That is not a hypothesis a cosine can settle. It is a hypothesis about what the
crops look like, so this writes them out and you look. Two seconds per sheet.

WHAT TO CONCLUDE
  low control + visibly front/back halves   -> orientation is the dominant term.
        No threshold and no crop gate reaches it; it needs a view-aware
        representation or a second signal that is not appearance.
  low control + halves that look the SAME   -> the backbone is failing on this
        domain, which is a different and more serious conversation.

Selection matches degrade_crops_causal.py exactly -- same --min-height, --min-obs
and --max-per-track defaults, same clip/sidecar pairing, same first-half /
second-half split -- so a sheet corresponds to that script's control row for the
same tracklet. Read-only over clips; no model, no GPU, no Qdrant.
"""

import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import bootstrap, arg, validate_flags, header             # noqa: E402

bootstrap()

from detector import crop_person                                       # noqa: E402
from types import SimpleNamespace                                      # noqa: E402

FLAGS = ("--clips", "--tracklets", "--out", "--min-height", "--min-obs",
         "--max-per-track", "--tile-height")
validate_flags(FLAGS)

CLIPS = arg("--clips", ".")
WANT = [t.strip() for t in (arg("--tracklets", "") or "").split(",") if t.strip()]
OUT = arg("--out", "sheets")
MIN_HEIGHT = float(arg("--min-height", "350"))
MIN_OBS = int(arg("--min-obs", "10"))
MAX_PER_TRACK = int(arg("--max-per-track", "24"))
TILE_H = int(arg("--tile-height", "192"))


def load_clip_crops(clip_dir):
    """-> {(camera, track_id): [crop, ...]}, sidecar-paired and index-aligned."""
    import glob
    import json
    out = defaultdict(list)
    clips = sorted(glob.glob(os.path.join(clip_dir, "._live_src_*.mp4")))
    if not clips:
        raise SystemExit(f"[sheet] no ._live_src_*.mp4 in {clip_dir!r}.")
    for clip in clips:
        side = os.path.splitext(clip)[0] + ".annotations.json"
        if not os.path.exists(side):
            print(f"  [skip] {os.path.basename(clip)} has no .annotations.json")
            continue
        with open(side) as f:
            blob = json.load(f)
        cam = blob.get("camera") or os.path.basename(clip)[len("._live_src_"):-4]
        anns = blob.get("annotations") or []
        cap = cv2.VideoCapture(clip)
        for _i, boxes in enumerate(anns):
            ok, frame = cap.read()
            if not ok:
                break
            for b in (boxes or []):
                tid = b.get("track_id")
                if tid is None:
                    continue
                key = (cam, int(tid))
                if WANT and f"{cam}:{tid}" not in WANT:
                    continue
                if len(out[key]) >= MAX_PER_TRACK:
                    continue
                if float(b["y2"]) - float(b["y1"]) < MIN_HEIGHT:
                    continue
                crop = crop_person(frame, SimpleNamespace(
                    x1=int(b["x1"]), y1=int(b["y1"]),
                    x2=int(b["x2"]), y2=int(b["y2"])))
                if crop is None or crop.size == 0:
                    continue
                out[key].append(crop)
        cap.release()
    return {k: v for k, v in out.items() if len(v) >= MIN_OBS}


def strip(crops, tile_h):
    """Crops side by side at a common height, aspect preserved."""
    tiles = []
    for c in crops:
        h, w = c.shape[:2]
        nw = max(1, int(round(w * tile_h / max(h, 1))))
        tiles.append(cv2.resize(c, (nw, tile_h), interpolation=cv2.INTER_AREA))
    if not tiles:
        return None
    return np.hstack(tiles)


def label_bar(width, text, height=26):
    bar = np.full((height, width, 3), 32, dtype=np.uint8)
    cv2.putText(bar, text, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (235, 235, 235), 1, cv2.LINE_AA)
    return bar


def main():
    by_track = load_clip_crops(CLIPS)
    if not by_track:
        raise SystemExit(
            f"[sheet] no tracklet in {CLIPS!r} has >= {MIN_OBS} crops at >= "
            f"{MIN_HEIGHT:.0f} px" + (f" matching {WANT}" if WANT else "") + ".")
    os.makedirs(OUT, exist_ok=True)

    header("CONTACT SHEETS -- first half (TOP) vs second half (BOTTOM)")
    print("  The split is the SAME one degrade_crops_causal.py scored, so each sheet")
    print("  is that script's 'control' row for this tracklet. If the two rows show")
    print("  the person from different sides, orientation is what the control number")
    print("  was measuring -- not resolution, not the bar.")
    print()
    written = []
    for key in sorted(by_track, key=lambda t: (str(t[0]), t[1])):
        crops = by_track[key]
        mid = len(crops) // 2
        a, b = crops[:mid], crops[mid:]
        top, bot = strip(a, TILE_H), strip(b, TILE_H)
        if top is None or bot is None:
            continue
        w = max(top.shape[1], bot.shape[1])

        def pad(img):
            if img.shape[1] == w:
                return img
            return np.hstack([img, np.full((img.shape[0], w - img.shape[1], 3),
                                           32, dtype=np.uint8)])

        hs = [c.shape[0] for c in crops]
        sheet = np.vstack([
            label_bar(w, f"{key[0]}:{key[1]}   n={len(crops)}  "
                         f"crop h med={int(np.median(hs))}px   "
                         f"FIRST HALF (n={len(a)})"),
            pad(top),
            label_bar(w, f"SECOND HALF (n={len(b)})"),
            pad(bot),
        ])
        path = os.path.join(OUT, f"halves_{key[0]}_{key[1]}.png")
        cv2.imwrite(path, sheet)
        written.append((path, key, len(crops), int(np.median(hs))))
        print(f"  {path}   ({len(a)} + {len(b)} crops, h med "
              f"{int(np.median(hs))}px)")

    print()
    print(f"  {len(written)} sheet(s) in {OUT}/")
    print()
    print("  The two worth opening first are the extremes of the control column:")
    print("    halves_cam_219_14.png   control 0.594   <- lowest")
    print("    halves_cam_219_7.png    control 0.950   <- highest, same camera")
    print("  Same camera, same run, same lighting. If 14's rows are front-then-back")
    print("  and 7's are one consistent view, the question is answered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
