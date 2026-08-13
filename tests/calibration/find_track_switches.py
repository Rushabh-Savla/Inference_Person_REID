#!/usr/bin/env python3
"""
Find ByteTrack ID SWITCHES: where inside a track does the person change?

    python tests/calibration/find_track_switches.py --clips .
    python tests/calibration/find_track_switches.py --clips . --cam cam_219

Read-only over the frozen clips + sidecars. No Qdrant, no camera time, no labels.

WHY THIS EXISTS
===============
`reconcile.py`'s docstring says "one ByteTrack track is (barring an id switch)
provably one person", and every prototype, threshold and veto in the identity layer
rests on that parenthesis. On 2026-08-04 the operator read two contact sheets and
reported that BOTH halves of cam_219:6 and cam_219:14 are DIFFERENT PEOPLE. The
parenthesis is load-bearing and it does not hold.

That invalidated a chain of conclusions, so the point of this script is to stop
assuming and start locating. A chimeric tracklet is worse than a missing one: its
prototype is a mean over two identities, so it merges with the wrong people AND
splits from the right ones -- the operator's symptom in both directions at once.

WHY SCANNING BEATS THE SPLIT-HALF TEST IT REPLACES
  The split-half control cut every track at its MIDPOINT. That only sees a switch
  that happens to occur near the middle, and it reports a mixture score rather than
  a switch. Scanning every split point instead finds the cut that MINIMISES the
  similarity between the two sides, which for a real switch lands on the switch
  frame and returns a much sharper number -- and, unlike the midpoint test, it says
  WHERE, so the tracklet can be split rather than discarded.

THE RANDOM BASELINE IS WHAT MAKES THE NUMBER MEAN ANYTHING
  A track's two temporal sides can differ because the person turned, because the
  light changed, or because it is two people. A RANDOM partition of the same
  observations controls for all of it at once: it puts the same mixture on both
  sides, so it stays high no matter which of those is true. The gap between the
  random baseline and the best temporal cut is therefore a measure of TEMPORAL
  STRUCTURE, and a hard, localised drop is the signature of a switch rather than of
  a gradual turn.

  Read `spread` too. A switch is a STEP: scores are high on both sides of one frame
  and low across it, so the scan curve is bimodal and `spread` (p90 - min) is large.
  A person slowly turning gives a shallow bowl -- a smaller spread at the same
  minimum. That is the one distinction this script can offer between "two people"
  and "one person changing view", and it is suggestive, not proof: only the contact
  sheet at the reported frame settles it, which is why the frame is printed.

NOT A FIX, AND DELIBERATELY NOT WIRED TO ANYTHING. It reports; you look.
"""

import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (bootstrap, arg, validate_flags, reid_weights,      # noqa: E402
                     reid_model, reid_tap, header)

bootstrap()

from reid.extractor import ReIDExtractor                               # noqa: E402
from detector import crop_person                                       # noqa: E402
from identity.reconcile import _prototype                              # noqa: E402
from types import SimpleNamespace                                      # noqa: E402

FLAGS = ("--clips", "--cam", "--min-height", "--min-obs", "--max-per-track",
         "--device", "--edge", "--min-side-frac", "--flag-gap", "--seed")
validate_flags(FLAGS)

CLIPS = arg("--clips", ".")
ONLY_CAM = arg("--cam")
# Deliberately LOWER than the degradation scripts' 350. Those needed pixels to give
# away; this only needs an embeddable crop, and a switch anywhere corrupts a
# tracklet regardless of how big the person was. 96 is above reid.quality's
# min_height of 64 without excluding most of cam_213.
MIN_HEIGHT = float(arg("--min-height", "96"))
MIN_OBS = int(arg("--min-obs", "8"))
MAX_PER_TRACK = int(arg("--max-per-track", "40"))
DEVICE = arg("--device", "cuda")
# Both sides of a cut need enough observations for a prototype to mean anything.
EDGE = int(arg("--edge", "3"))
# ...and a FRACTION, which is the part that matters. With edge=3 alone the scan was
# free to cut off the first or last three crops, and on run 20260804_094039 that is
# where it landed for almost every "suspect": 3, 4, 5, 34 or 37 out of 40. Those are
# exactly the frames where a person is entering or leaving view -- clipped by the
# frame boundary, occluded, motion-blurred -- so a 3-crop prototype built from them
# disagrees with everything and the scan reported entry/exit noise as a switch.
# cam_213:19 took the highest gap of all 36 tracklets (0.746) cutting at 5/40, while
# its contact sheet shows one person in a patterned shirt throughout.
#
# A real switch divides a track into two SUBSTANTIAL parts, so requiring both sides
# to hold this fraction of the observations costs nothing real and removes the
# artifact. 0.25 still finds a switch anywhere in the middle half of a track.
MIN_SIDE_FRAC = float(arg("--min-side-frac", "0.25"))
FLAG_GAP = float(arg("--flag-gap", "0.20"))
SEED = int(arg("--seed", "1234"))


def load_clip_crops(clip_dir):
    """-> {(cam, tid): [(frame_index, crop), ...]} in capture order."""
    import glob
    import json
    out = defaultdict(list)
    clips = sorted(glob.glob(os.path.join(clip_dir, "._live_src_*.mp4")))
    if not clips:
        raise SystemExit(f"[switch] no ._live_src_*.mp4 in {clip_dir!r}.")
    for clip in clips:
        side = os.path.splitext(clip)[0] + ".annotations.json"
        if not os.path.exists(side):
            print(f"  [skip] {os.path.basename(clip)} has no .annotations.json")
            continue
        with open(side) as f:
            blob = json.load(f)
        cam = blob.get("camera") or os.path.basename(clip)[len("._live_src_"):-4]
        if ONLY_CAM and cam != ONLY_CAM:
            continue
        cap = cv2.VideoCapture(clip)
        for i, boxes in enumerate(blob.get("annotations") or []):
            ok, frame = cap.read()
            if not ok:
                break
            for b in (boxes or []):
                tid = b.get("track_id")
                if tid is None:
                    continue
                key = (cam, int(tid))
                if len(out[key]) >= MAX_PER_TRACK:
                    continue
                if float(b["y2"]) - float(b["y1"]) < MIN_HEIGHT:
                    continue
                crop = crop_person(frame, SimpleNamespace(
                    x1=int(b["x1"]), y1=int(b["y1"]),
                    x2=int(b["x2"]), y2=int(b["y2"])))
                if crop is None or crop.size == 0:
                    continue
                out[key].append((i, crop))
        cap.release()
    return {k: v for k, v in out.items() if len(v) >= MIN_OBS}


def scan(v):
    """-> (min_score, best_k, curve) over cuts leaving a real tracklet on both sides."""
    lo = max(EDGE, int(round(len(v) * MIN_SIDE_FRAC)))
    if len(v) - lo < lo:
        return None
    scores, ks = [], []
    for k in range(lo, len(v) - lo + 1):
        pa, pb = _prototype(v[:k]), _prototype(v[k:])
        if pa is None or pb is None:
            continue
        scores.append(float(pa @ pb))
        ks.append(k)
    if not scores:
        return None
    arr = np.asarray(scores)
    j = int(arr.argmin())
    return float(arr[j]), ks[j], arr


def random_baseline(v, rng, trials=5):
    """Mean similarity of RANDOM halves -- the same mixture on both sides."""
    out = []
    idx = np.arange(len(v))
    mid = len(v) // 2
    for _ in range(trials):
        rng.shuffle(idx)
        pa, pb = _prototype(v[idx[:mid]]), _prototype(v[idx[mid:]])
        if pa is not None and pb is not None:
            out.append(float(pa @ pb))
    return float(np.mean(out)) if out else float("nan")


def main():
    print(__doc__.split("WHY THIS EXISTS")[0].strip())
    by_track = load_clip_crops(CLIPS)
    if not by_track:
        raise SystemExit(f"[switch] no tracklet in {CLIPS!r} has >= {MIN_OBS} crops "
                         f"at >= {MIN_HEIGHT:.0f} px.")

    ex = ReIDExtractor(weights=reid_weights(), model=reid_model(),
                       tap=reid_tap(), device=DEVICE)
    header("SETUP")
    print(f"  ReID: {ex.describe()}")
    print(f"  min_height={MIN_HEIGHT:.0f} min_obs={MIN_OBS} cap={MAX_PER_TRACK} "
          f"edge={EDGE} min_side_frac={MIN_SIDE_FRAC} flag_gap={FLAG_GAP}")
    print(f"  {len(by_track)} tracklet(s)")

    header("SCAN -- the split point that MINIMISES similarity within each track")
    print("  worst_cut : lowest similarity across any cut. random : same observations,")
    print("  shuffled, so temporal structure is removed. gap : random - worst_cut,")
    print("  i.e. how much of the disagreement is ORDERED IN TIME.")
    print("  spread    : p90 - min of the scan curve. A switch is a STEP, so it is")
    print("              large; a gradual turn is a shallow bowl, so it is small.")
    print("  frame     : the clip frame index at the worst cut -- go LOOK there.")
    print()
    print(f"  {'tracklet':<16}{'n':>4}{'worst_cut':>11}{'random':>9}{'gap':>8}"
          f"{'spread':>9}{'frame':>8}{'split':>10}")

    rng = np.random.default_rng(SEED)
    rows = []
    for key in sorted(by_track, key=lambda t: (str(t[0]), t[1])):
        items = by_track[key]
        v = np.asarray(ex.extract_batch([c for _f, c in items]), dtype=np.float32)
        if v.ndim != 2 or v.shape[0] != len(items):
            continue
        got = scan(v)
        if got is None:
            continue
        worst, k, curve = got
        rand = random_baseline(v, rng)
        gap = rand - worst
        spread = float(np.percentile(curve, 90) - curve.min())
        frame = items[k][0] if k < len(items) else items[-1][0]
        rows.append((key, len(items), worst, rand, gap, spread, frame, k))
        flag = "  <== SUSPECT" if gap >= FLAG_GAP else ""
        print(f"  {key[0]}:{key[1]:<8}{len(items):>4}{worst:>11.3f}{rand:>9.3f}"
              f"{gap:>8.3f}{spread:>9.3f}{frame:>8}{f'{k}/{len(items)}':>10}{flag}")

    header("SUMMARY")
    n_sus = sum(1 for r in rows if r[4] >= FLAG_GAP)
    print(f"  {n_sus}/{len(rows)} tracklet(s) with gap >= {FLAG_GAP:.2f}")
    if rows:
        gaps = np.asarray([r[4] for r in rows])
        print(f"  gap: min={gaps.min():.3f} median={np.median(gaps):.3f} "
              f"max={gaps.max():.3f}")
    print()
    print("  CONFIRM BEFORE BELIEVING ANY OF IT. The gap says a track is temporally")
    print("  inconsistent; it does NOT say why. Two people, one person turning, and a")
    print("  lighting change all produce it. Only the crops settle that:")
    print("    python tests/calibration/contact_sheet_halves.py --clips . \\")
    print("      --tracklets <cam:tid,...> --min-height 96 --min-obs 8")
    print()
    print("  KNOWN GROUND TRUTH so far, from the operator reading the sheets:")
    print("    cam_219:6  and cam_219:14  are BOTH two different people.")
    print("  Those two had the largest midpoint gaps (0.368, 0.348) of 18 tracklets,")
    print("  which is what suggests the gap tracks switches -- on n=2. Do not treat")
    print("  the flag_gap default of 0.20 as calibrated; it is a starting point, and")
    print("  the sheets are what calibrate it.")
    print()
    print("  IF THE SUSPECTS CONFIRM: the fix is upstream of scoring entirely --")
    print("  either ByteTrack association (appearance in the matching step, or")
    print("  stricter gating) or splitting a tracklet at this frame BEFORE reconcile")
    print("  builds a prototype over two people. No threshold, mode or veto can")
    print("  repair a chimeric tracklet, which is why this outranks all of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
