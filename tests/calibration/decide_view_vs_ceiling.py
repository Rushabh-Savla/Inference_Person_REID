#!/usr/bin/env python3
"""
THE DECIDING MEASUREMENT: is mean-pooling losing view information, or is the
embedding the ceiling on this domain?

    python tests/calibration/decide_view_vs_ceiling.py --clips .

Crops come from the frozen clips + sidecars. No camera time, no labels, no headcount,
no Qdrant. Read-only.

WHAT IS ALREADY SETTLED, so this does not re-litigate it
=======================================================
One tracklet split in half is provably ONE person, one camera, one lighting
condition -- the easiest comparison that exists. On run 20260804_094039 those
same-person control cosines ranged 0.594 to 0.950 (mean 0.806, n=18), while a
CO-PRESENT pair in cam_219 -- provably two people, since one body cannot be two
simultaneous detections -- scored 0.843, and the operator's own known re-appearance
pair scored 0.574.

To merge that person you need a bar <= 0.594. To reject that stranger you need one
> 0.843. No bar satisfies both, so no threshold was ever going to work. Also killed
by controlled degradation on the same tracklets: resolution (6x downscale costs
0.045) and sharpness (sigma-5 blur costs 0.119) are both real and both an order of
magnitude short of the 0.27 gap.

What remains is WHERE the same-person signal goes. Two candidates, and they imply
completely different work:

  #1  MEAN-POOLING destroys it. A prototype averaged over front-facing and
      rear-facing observations sits between the two modes and matches neither, so
      the information is present in the observations and lost in the summary.
      -> better aggregation. No new model.

  #5  THE EMBEDDING is the ceiling. No view of the person matches any other view,
      so there is nothing for a better summary to recover.
      -> stronger representation, and re-ranking over these prototypes would
         inherit the failure rather than fix it.

HOW THIS SEPARATES THEM
  A mean is one summary; max_exemplar is the best single OBSERVATION pair. If the
  observations contain a matching view that the mean buried, max_exemplar finds it
  and #1 is confirmed. If even the single best pair of observations across the two
  halves scores low, no summary can help and #5 is confirmed.

  The RANDOM split is the control for that control. A temporal split separates
  early from late, so it confounds "the person turned" with "embeddings drift
  frame to frame". A random partition of the same observations destroys the
  temporal structure while keeping everything else. Temporal low + random high
  means the change is real and ordered in time; both low means per-frame
  instability, which is a different problem entirely.

THE DECISION RULES ARE PRE-REGISTERED. They are printed BEFORE the results, below,
because the last two hypotheses in this project were killed by criteria stated in
advance and the ones before that were argued about after the fact.
"""

import os
import random
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
from identity.reconcile import (score_observation_sets, _prototype,     # noqa: E402
                                PROTOTYPE, MAX_EXEMPLAR, CONSENSUS, VIEW_MEDOID)
from types import SimpleNamespace                                      # noqa: E402

FLAGS = ("--clips", "--cam", "--min-height", "--min-obs", "--max-per-track",
         "--device", "--seed", "--top-frac")
validate_flags(FLAGS)

CLIPS = arg("--clips", ".")
ONLY_CAM = arg("--cam")
MIN_HEIGHT = float(arg("--min-height", "350"))
MIN_OBS = int(arg("--min-obs", "10"))
MAX_PER_TRACK = int(arg("--max-per-track", "24"))
DEVICE = arg("--device", "cuda")
SEED = int(arg("--seed", "1234"))
TOP_FRAC = float(arg("--top-frac", "0.25"))       # matches config consensus_top_frac

# ---- PRE-REGISTERED DECISION RULES ---------------------------------------------
RULES = """
  Judged on the LOW-CONTROL tracklets (prototype split-half < 0.70), because those
  are the ones the identity failures come from. A rule that only holds on the easy
  tracklets decides nothing.

    max_exemplar on the low controls >= 0.85   -> #1 MEAN-POOLING. The matching view
                                                  exists and the mean buried it.
                                                  Next: better aggregation (Task 4).
                                                  No new model needed.

    max_exemplar on the low controls <  0.70   -> #5 EMBEDDING CEILING. No view
                                                  matches any other view. Next:
                                                  stronger representation (Task 5).
                                                  Re-ranking would inherit this.

    0.70 <= max_exemplar < 0.85                -> PARTIAL. Some view information is
                                                  recoverable but not enough to clear
                                                  any usable bar. Report both, do
                                                  neither yet.

    random ~ 0.95 while temporal ~ 0.59        -> appearance changes ACROSS the track
                                                  (orientation). View-aware
                                                  representation is the target.

    random ~ temporal ~ 0.6                    -> frame-to-frame INSTABILITY, not view
                                                  change. Different problem;
                                                  investigate before proceeding.

  A stranger reference is printed beside every number, because a same-person score
  only means something relative to what two different people score in the same
  camera. Any mode whose same-person figure sits below its stranger figure has
  gained nothing regardless of how much it improved.
"""


def load_clip_crops(clip_dir):
    """-> {(camera, track_id): [crop, ...]}. Sidecar `annotations` is index-aligned
    with clip frames, same pairing rule as rerender_from_clips.load_clips."""
    import glob
    import json
    out = defaultdict(list)
    frames = defaultdict(set)
    clips = sorted(glob.glob(os.path.join(clip_dir, "._live_src_*.mp4")))
    if not clips:
        raise SystemExit(f"[decide] no ._live_src_*.mp4 in {clip_dir!r}.")
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
        for _i, boxes in enumerate(blob.get("annotations") or []):
            ok, frame = cap.read()
            if not ok:
                break
            for b in (boxes or []):
                tid = b.get("track_id")
                if tid is None:
                    continue
                key = (cam, int(tid))
                # CO-PRESENCE IS RECORDED BEFORE THE FILTERS, deliberately. It is a
                # statement about the world -- these two tracklets were detected in
                # the same frame, so they are two people -- and it does not depend on
                # whether either crop was big enough to embed or fell inside the
                # per-track cap. Recording it after the filters truncated the frame
                # sets and UNDERCOUNTED provable stranger pairs, which is the one
                # statistic the whole inversion question turns on.
                frames[key].add(_i)
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
    return ({k: v for k, v in out.items() if len(v) >= MIN_OBS},
            {k: v for k, v in frames.items() if len(out.get(k, ())) >= MIN_OBS})


def score_halves(rows_a, rows_b, mode):
    """Reconcile's OWN scorer, so these numbers are the ones reconcile would use."""
    pa, pb = _prototype(list(rows_a)), _prototype(list(rows_b))
    if pa is None or pb is None:
        return float("nan")
    return float(score_observation_sets(np.asarray(rows_a, dtype=np.float32),
                                        np.asarray(rows_b, dtype=np.float32),
                                        pa, pb, mode, TOP_FRAC))


def main():
    print(__doc__.split("WHAT IS ALREADY SETTLED")[0].strip())
    header("PRE-REGISTERED DECISION RULES (stated before any result)")
    print(RULES)

    by_track, frames = load_clip_crops(CLIPS)
    if not by_track:
        raise SystemExit(
            f"[decide] no tracklet in {CLIPS!r} has >= {MIN_OBS} crops at >= "
            f"{MIN_HEIGHT:.0f} px. Lower --min-height / --min-obs.")

    ex = ReIDExtractor(weights=reid_weights(), model=reid_model(),
                       tap=reid_tap(), device=DEVICE)
    header("SETUP")
    print(f"  ReID: {ex.describe()}")
    print(f"  clips={CLIPS!r} min_height={MIN_HEIGHT:.0f} min_obs={MIN_OBS} "
          f"cap={MAX_PER_TRACK} consensus_top_frac={TOP_FRAC}")
    print(f"  {len(by_track)} usable tracklet(s)")

    # Embed once. Every column below is a different way of SUMMARISING these same
    # vectors, so re-embedding per column would confound the comparison.
    vecs = {}
    for key, crops in by_track.items():
        v = np.asarray(ex.extract_batch(crops), dtype=np.float32)
        if v.ndim == 2 and v.shape[0] == len(crops):
            vecs[key] = v

    header("1. TEMPORAL SPLIT-HALF, all three modes  +  2. RANDOM SPLIT (prototype)")
    print("  temporal = first half vs second half. random = same observations,")
    print("  shuffled partition, so temporal structure is destroyed and nothing else.")
    print()
    print(f"  {'tracklet':<16}{'proto':>8}{'max_ex':>9}{'consen':>9}{'vmedoid':>9}"
          f"{'rand_proto':>12}{'rand-temp':>11}")

    rng = random.Random(SEED)
    rows = []
    for key in sorted(vecs, key=lambda t: (str(t[0]), t[1])):
        v = vecs[key]
        mid = len(v) // 2
        a, b = v[:mid], v[mid:]
        p = score_halves(a, b, PROTOTYPE)
        mx = score_halves(a, b, MAX_EXEMPLAR)
        cs = score_halves(a, b, CONSENSUS)
        vm = score_halves(a, b, VIEW_MEDOID)
        idx = list(range(len(v)))
        rng.shuffle(idx)
        ra, rb = v[idx[:mid]], v[idx[mid:]]
        rp = score_halves(ra, rb, PROTOTYPE)
        rows.append((key, p, mx, cs, rp, vm))
        print(f"  {key[0]}:{key[1]:<8}{p:>8.3f}{mx:>9.3f}{cs:>9.3f}{vm:>9.3f}"
              f"{rp:>12.3f}{rp - p:>11.3f}")

    # Stranger reference: co-present tracklets in one camera are provably two people.
    # Without it a same-person number cannot be judged.
    header("3. STRANGER REFERENCE -- co-present pairs, provably two people")
    print("  Any mode whose same-person figure sits below its stranger figure has")
    print("  gained nothing, however much it improved in absolute terms.")
    print()
    strangers = defaultdict(list)
    copresent = defaultdict(list)
    for cam in sorted({k[0] for k in vecs}, key=str):
        ks = [k for k in vecs if k[0] == cam]
        for i, ka in enumerate(ks):
            for kb in ks[i + 1:]:
                shared = frames.get(ka, set()) & frames.get(kb, set())
                for mode, tag in ((PROTOTYPE, "proto"), (MAX_EXEMPLAR, "max_ex"),
                                  (CONSENSUS, "consen"), (VIEW_MEDOID, "vmed")):
                    sc = score_halves(vecs[ka], vecs[kb], mode)
                    strangers[(cam, tag)].append(sc)
                    if shared:
                        copresent[(cam, tag)].append(sc)

    def _cell(vals, pct=None):
        x = np.asarray([v for v in vals if v == v])
        if not len(x):
            return "-"
        return f"{np.percentile(x, pct):.3f}" if pct else f"{x.max():.3f}"

    print("  ALL same-camera distinct pairs -- UNLABELLED, so an upper bound:")
    print(f"  {'camera':<10}{'n':>5}" + "".join(
        f"{h:>12}" for h in ("proto p95", "proto MAX", "max_ex p95", "max_ex MAX",
                             "consen p95", "consen MAX", "vmed p95", "vmed MAX")))
    for cam in sorted({c for c, _ in strangers}, key=str):
        row = f"  {cam:<10}{len(strangers[(cam, 'proto')]):>5}"
        for tag in ("proto", "max_ex", "consen", "vmed"):
            row += (f"{_cell(strangers[(cam, tag)], 95):>12}"
                    f"{_cell(strangers[(cam, tag)]):>12}")
        print(row)
    print()
    print("  CO-PRESENT pairs only -- PROVABLY two people. This is the real ceiling,")
    print("  and the MAX matters as well as p95: a hard veto is judged by its worst")
    print("  case, while CLAUDE.md section 6.3 prefers p95 because max grows with")
    print("  sample size. Both are printed so neither can be quoted alone.")
    print()
    print(f"  {'camera':<10}{'n':>5}{'proto p95':>11}{'proto MAX':>11}"
          f"{'max_ex p95':>12}{'max_ex MAX':>12}{'consen p95':>12}{'consen MAX':>12}"
          f"{'vmed p95':>11}{'vmed MAX':>11}")
    for cam in sorted({c for c, _ in copresent}, key=str):
        n = len(copresent[(cam, "proto")])
        row = f"  {cam:<10}{n:>5}"
        for tag in ("proto", "max_ex", "consen", "vmed"):
            row += f"{_cell(copresent[(cam, tag)], 95):>11}{_cell(copresent[(cam, tag)]):>11}"
        print(row)
    print()
    print("  READ THIS AGAINST THE LOW CONTROLS. A mode has only helped if the")
    print("  low-control same-person score clears its OWN camera's co-present MAX.")
    print("  If max_exemplar lifts the same-person figure and the stranger MAX by")
    print("  the same amount, the window is still empty and nothing was gained --")
    print("  that is the 'other MAX' trap (0.819 -> 0.936 at 48 vs 90 frames).")

    header("VERDICT")
    arr = {i: np.asarray([r[i] for r in rows], dtype=float) for i in (1, 2, 3, 4, 5)}
    low = [r for r in rows if r[1] == r[1] and r[1] < 0.70]
    print(f"  n = {len(rows)} tracklet(s);  {len(low)} with prototype control < 0.70")
    for i, name in ((1, "prototype"), (2, "max_exemplar"), (3, "consensus"),
                    (5, "view_medoid"), (4, "random prototype")):
        x = arr[i][~np.isnan(arr[i])]
        print(f"    {name:<18} all: mean={x.mean():.3f} min={x.min():.3f}")
    if not low:
        print("\n  NO low-control tracklet in this sample -- the pre-registered rules")
        print("  are all conditioned on those, so NOTHING is decided. Widen the")
        print("  sample (--min-obs 6, or another run's clips) before concluding.")
        return 0

    lm = np.asarray([r[2] for r in low], dtype=float)
    lp = np.asarray([r[1] for r in low], dtype=float)
    lr = np.asarray([r[4] for r in low], dtype=float)
    print(f"\n  LOW-CONTROL subset (n={len(low)}):")
    print(f"    prototype        mean={lp.mean():.3f}")
    print(f"    max_exemplar     mean={lm.mean():.3f}   <- the deciding number")
    print(f"    random prototype mean={lr.mean():.3f}")
    lv = np.asarray([r[5] for r in low], dtype=float)
    print(f"    view_medoid      mean={lv.mean():.3f}")
    for k, p, mx, cs, rp, vm in low:
        print(f"      {k[0]}:{k[1]:<8} proto={p:.3f} max_ex={mx:.3f} "
              f"vmedoid={vm:.3f} rand={rp:.3f}")

    print()
    if lm.mean() >= 0.85:
        print("  => BRANCH 1 FIRED: MEAN-POOLING. max_exemplar recovers the low")
        print("     controls to >= 0.85, so a matching view exists in the")
        print("     observations and the mean was burying it. Go to Task 4 (better")
        print("     aggregation). No new model needed. Check the stranger table")
        print("     first: max_exemplar raises stranger scores too.")
    elif lm.mean() < 0.70:
        print("  => BRANCH 2 FIRED: EMBEDDING CEILING. Even the single best pair of")
        print("     observations across the two halves scores < 0.70 on the tracklets")
        print("     that matter, so no summary can recover what is not there. Go to")
        print("     Task 5 (stronger representation). Re-ranking over these")
        print("     prototypes would inherit the failure.")
    else:
        print("  => PARTIAL: max_exemplar recovers some view information but not")
        print("     enough to clear a usable bar. Report both; do neither yet.")

    if lr.mean() - lp.mean() >= 0.20:
        print("\n  RANDOM >> TEMPORAL: the appearance change is real and ordered in")
        print("  time, i.e. the person changed view across the track. A view-aware")
        print("  representation is the target, not a better average.")
    elif abs(lr.mean() - lp.mean()) < 0.10:
        print("\n  RANDOM ~ TEMPORAL: not a view change -- the observations disagree")
        print("  even when time is shuffled out. That is frame-to-frame instability")
        print("  and it is a DIFFERENT problem. Investigate before Task 4 or 5.")

    print()
    print("  NEXT, and only this: run contact_sheet_halves.py on the low-control")
    print("  tracklets and LOOK. Three readings, and the third is checked first:")
    print("  the person turned (view change), the halves look identical (domain")
    print("  failure), or the halves are TWO DIFFERENT PEOPLE -- a ByteTrack id")
    print("  switch mid-track, which contaminates the control rather than")
    print("  indicting the model, and invalidates every tracklet on that track.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
