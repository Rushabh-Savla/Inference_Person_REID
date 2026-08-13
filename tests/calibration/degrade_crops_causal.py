#!/usr/bin/env python3
"""
Does effective RESOLUTION (or SHARPNESS) causally destroy same-person cosine?

    python tests/calibration/degrade_crops_causal.py --clips .            # all cameras
    python tests/calibration/degrade_crops_causal.py --clips . --cam cam_213
    python tests/calibration/degrade_crops_causal.py --clips . --scales 2,4,6

WHY THIS EXISTS, AND WHY THE CORRELATION WAS NOT ENOUGH
=======================================================
`tools/crop_scale_vs_score.py` measured corr(cosine, crop-height ratio) = -0.49
over 259 time-disjoint same-camera tracklet pairs on run 20260804_094039. That is
suggestive and NOT causal, for three reasons that no amount of extra pairs fixes:

  1. The disjoint set is UNLABELLED. Time-disjoint same-camera pairs are a mix of
     one person returning and two different people appearing at different times.
     With several people in frame most of those 259 pairs are probably strangers,
     so "the cost falls on same-person pairs" reads a label into the data that is
     not there. That claim is WITHDRAWN.
  2. The co-present comparison had n=28 and a 95% CI of roughly [-0.61, +0.07] --
     it includes zero, and it overlaps the disjoint CI heavily. It also has a
     narrower h_ratio range (mean 1.56 vs 2.16), and restricted range attenuates a
     correlation on its own. The two numbers were never comparable.
  3. h_ratio encodes WHERE IN THE ROOM someone was, which encodes lighting, pose
     distribution and viewing angle. Scale is entangled with everything that
     changes as a person walks.

So this script holds all of that fixed instead. One tracklet, split in half:
person, clothing, camera, lighting, pose distribution and room position are
IDENTICAL on both sides. The ONLY difference is that one half's crops are degraded
before embedding. Any cosine drop is caused by the degradation, because nothing
else varies.

    control   cosine(proto(A), proto(B))                       -- baseline
    scale k   cosine(proto(A), proto(downscale B by k, back))   -- resolution loss
    blur s    cosine(proto(A), proto(gaussian-blur B by s))     -- sharpness loss

The scale arm is faithful to the real mechanism: a distant person is captured at
low resolution and the extractor then upscales to 384x128. Downscaling a crop and
handing it to the same `extract_batch` reproduces exactly that path -- it does not
simulate it.

WHAT EACH OUTCOME MEANS
  control ~0.95 -> scale-4 ~0.4   the mechanism is real and quantified. A
                                  resolution-aware rule is justified.
  control ~0.95 -> scale-4 ~0.9   resolution is NOT the driver; the -0.49 was
                                  confounded, and an absolute size gate would
                                  have cost recall for nothing.

THE BLUR ARM ALSO SETTLES A CONFOUNDED MEASUREMENT. cam_219 was called "5-50x
softer" than cam_213 from stored Laplacian variance. That comparison is invalid:
blur is computed on the RAW crop before resize (reid/service.py::_crop_quality),
and Laplacian variance is not scale-invariant -- the same physical detail spread
over more pixels gives smaller per-pixel gradients, so cam_219's large crops read
"soft", while cam_213's small, noisy, distant crops read "sharp" partly on sensor
noise. This arm asks the question that actually matters: at matched crop size,
does softening cost discrimination in THIS feature space?

NO CONFIG IS READ FOR POLICY AND NOTHING IS WRITTEN. Read-only over clips; it does
not touch Qdrant at all.
"""

import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (bootstrap, arg, reid_weights, reid_model, reid_tap,  # noqa: E402
                     validate_flags, header)

bootstrap()

from reid.extractor import ReIDExtractor                          # noqa: E402
from detector import crop_person                                  # noqa: E402
from types import SimpleNamespace                                 # noqa: E402

FLAGS = ("--clips", "--cam", "--scales", "--blurs", "--gammas", "--min-height",
         "--min-obs", "--max-per-track", "--device")
validate_flags(FLAGS)

CLIPS = arg("--clips", ".")
ONLY_CAM = arg("--cam")
SCALES = [float(x) for x in (arg("--scales", "2,4,6") or "").split(",") if x]
BLURS = [float(x) for x in (arg("--blurs", "1.5,3.0,5.0") or "").split(",") if x]
# EXPOSURE arm. Added 2026-08-04 after the contact sheets showed cam_219:14 -- the
# WORST control of all 18 at 0.594 -- differs between its halves almost entirely in
# BRIGHTNESS, not view: first half nearly black, second half normally lit, same
# seated person, same garment, same pose. Orientation cannot explain that one, and
# max_exemplar helps it least (0.809 vs 0.944 for the orientation case) precisely
# because there is no better-matched VIEW to find.
#
# Gamma > 1 darkens. Applied on the 0..255 crop before any resize, which is where a
# real under-exposure lives. It matters that FastReID normalises with FIXED dataset
# constants -- (batch - mean) / std in backends.py, not per-image -- so a dark crop
# stays dark relative to the training distribution rather than being corrected.
GAMMAS = [float(x) for x in (arg("--gammas", "1.8,2.6,3.4") or "").split(",") if x]
# Only tracklets with real pixels to give away can be degraded meaningfully: you
# cannot downscale a 90 px crop by 4 and learn anything about a 600 px one.
MIN_HEIGHT = float(arg("--min-height", "350"))
MIN_OBS = int(arg("--min-obs", "10"))
MAX_PER_TRACK = int(arg("--max-per-track", "24"))
DEVICE = arg("--device", "cuda")


def load_clip_crops(clip_dir):
    """-> {(camera, track_id): [crop, ...]} from ._live_src_*.mp4 + sidecars.

    Same pairing rule as rerender_from_clips.load_clips and
    compare_backbones.collect_from_clips: the sidecar's `annotations` list is
    INDEX-ALIGNED with the clip's frames, so entry i belongs to frame i.
    """
    import glob
    import json
    out = defaultdict(list)
    clips = sorted(glob.glob(os.path.join(clip_dir, "._live_src_*.mp4")))
    if not clips:
        raise SystemExit(
            f"[degrade] no ._live_src_*.mp4 in {clip_dir!r}. A run only keeps clips "
            f"when live.reconcile.keep_frames was on.")
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
        anns = blob.get("annotations") or []
        cap = cv2.VideoCapture(clip)
        for i, boxes in enumerate(anns):
            ok, frame = cap.read()
            if not ok:
                break
            if not boxes:
                continue
            for b in boxes:
                tid = b.get("track_id")
                if tid is None:
                    continue
                key = (cam, int(tid))
                if len(out[key]) >= MAX_PER_TRACK:
                    continue
                h = float(b["y2"]) - float(b["y1"])
                if h < MIN_HEIGHT:
                    continue
                crop = crop_person(frame, SimpleNamespace(
                    x1=int(b["x1"]), y1=int(b["y1"]),
                    x2=int(b["x2"]), y2=int(b["y2"])))
                if crop is None or crop.size == 0:
                    continue
                out[key].append(crop)
        cap.release()
    return {k: v for k, v in out.items() if len(v) >= MIN_OBS}


def lap_var(crop):
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def downscale_up(crop, k):
    """Lose resolution the way distance does, then let the extractor upscale.

    Shrinking to 1/k and handing the SMALL crop to extract_batch is the faithful
    path: the pipeline resizes whatever it is given to 384x128, so a distant person
    is exactly a small crop stretched up. Returned at 1/k size, not restored here.
    """
    h, w = crop.shape[:2]
    nh, nw = max(2, int(round(h / k))), max(2, int(round(w / k)))
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)


def darken(crop, gamma):
    """Under-expose by gamma on the 0..255 scale. gamma > 1 darkens."""
    lut = np.clip(((np.arange(256) / 255.0) ** float(gamma)) * 255.0,
                  0, 255).astype(np.uint8)
    return cv2.LUT(crop, lut)


def mean_luma(crop):
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())


def gaussian_soften(crop, sigma):
    ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(crop, (ksize, ksize), sigma)


def proto(ex, crops):
    if not crops:
        return None
    v = ex.extract_batch(crops)
    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 2 or v.shape[0] == 0:
        return None
    m = v.mean(axis=0)
    n = float(np.linalg.norm(m))
    return None if n == 0 else (m / n)


def main():
    by_track = load_clip_crops(CLIPS)
    if not by_track:
        raise SystemExit(
            f"[degrade] no tracklet in {CLIPS!r} has >= {MIN_OBS} crops at >= "
            f"{MIN_HEIGHT:.0f} px. Lower --min-height / --min-obs, or point --clips "
            f"at a run whose people were closer to the camera.")

    ex = ReIDExtractor(weights=reid_weights(), model=reid_model(),
                       tap=reid_tap(), device=DEVICE)
    header("SETUP")
    print(f"  ReID: {ex.describe()}")
    print(f"  clips={CLIPS!r}  min_height={MIN_HEIGHT:.0f}px  min_obs={MIN_OBS}  "
          f"cap={MAX_PER_TRACK}/tracklet")
    print(f"  {len(by_track)} usable tracklet(s):")
    for k in sorted(by_track, key=lambda t: (str(t[0]), t[1])):
        cs = by_track[k]
        hs = np.array([c.shape[0] for c in cs])
        lv = np.array([lap_var(c) for c in cs])
        print(f"    {k[0]}:{k[1]:<6} n={len(cs):<4} h med={np.median(hs):.0f} "
              f"lapvar med={np.median(lv):.0f}")

    header("THE EXPERIMENT -- one tracklet split in half, one half degraded")
    print("  Both halves are the SAME person, camera, lighting and room position.")
    print("  Only the right-hand half is degraded, so any drop is caused by it.")
    print()
    print("  'scale k' = that half's crops shrunk to 1/k before the extractor's own")
    print("  resize to 384x128 -- the exact path a distant person takes.")
    print("  'blur s'  = gaussian sigma s at full size, isolating sharpness from size.")
    print()
    cols = ([f"scale{int(k)}x" if k == int(k) else f"scale{k}x" for k in SCALES]
            + [f"blur{s}" for s in BLURS]
            + [f"gam{g}" for g in GAMMAS])
    print(f"  {'tracklet':<16}{'control':>9}" + "".join(f"{c:>10}" for c in cols))

    agg = defaultdict(list)
    for key in sorted(by_track, key=lambda t: (str(t[0]), t[1])):
        crops = by_track[key]
        mid = len(crops) // 2
        a, b = crops[:mid], crops[mid:]
        pa = proto(ex, a)
        pb = proto(ex, b)
        if pa is None or pb is None:
            continue
        control = float(pa @ pb)
        agg["control"].append(control)
        row = f"  {key[0]}:{key[1]:<8}{control:>9.3f}"
        for k in SCALES:
            pd = proto(ex, [downscale_up(c, k) for c in b])
            s = float(pa @ pd) if pd is not None else float("nan")
            agg[f"scale{k}"].append(s)
            row += f"{s:>10.3f}"
        for sg in BLURS:
            pd = proto(ex, [gaussian_soften(c, sg) for c in b])
            s = float(pa @ pd) if pd is not None else float("nan")
            agg[f"blur{sg}"].append(s)
            row += f"{s:>10.3f}"
        for g in GAMMAS:
            dk = [darken(c, g) for c in b]
            pd = proto(ex, dk)
            s = float(pa @ pd) if pd is not None else float("nan")
            agg[f"gam{g}"].append(s)
            agg[f"luma{g}"].append(float(np.mean([mean_luma(c) for c in dk])))
            row += f"{s:>10.3f}"
        agg["luma_native"].append(float(np.mean([mean_luma(c) for c in b])))
        print(row)

    header("AGGREGATE")
    n = len(agg["control"])
    if n == 0:
        raise SystemExit("[degrade] no tracklet produced two embeddable halves.")

    def ci95(xs):
        x = np.asarray([v for v in xs if v == v], dtype=float)
        if len(x) < 2:
            return float("nan"), float("nan")
        se = x.std(ddof=1) / np.sqrt(len(x))
        return float(x.mean()), float(1.96 * se)

    cm, ch = ci95(agg["control"])
    print(f"  n = {n} tracklet(s)")
    print(f"  {'condition':<14}{'mean cosine':>13}{'95% CI':>16}{'drop vs control':>18}")
    print(f"  {'control':<14}{cm:>13.3f}{'+/- %.3f' % ch:>16}{'--':>18}")
    for k in SCALES:
        m, h = ci95(agg[f"scale{k}"])
        print(f"  {('scale %gx' % k):<14}{m:>13.3f}{'+/- %.3f' % h:>16}"
              f"{cm - m:>18.3f}")
    for sg in BLURS:
        m, h = ci95(agg[f"blur{sg}"])
        print(f"  {('blur sigma %g' % sg):<14}{m:>13.3f}{'+/- %.3f' % h:>16}"
              f"{cm - m:>18.3f}")
    for g in GAMMAS:
        m, h = ci95(agg[f"gam{g}"])
        lm, _ = ci95(agg[f"luma{g}"])
        print(f"  {('gamma %g' % g):<14}{m:>13.3f}{'+/- %.3f' % h:>16}"
              f"{cm - m:>18.3f}   mean luma {lm:.0f}")
    ln, _ = ci95(agg["luma_native"])
    print(f"  (native mean luma of the undegraded half: {ln:.0f})")
    print()
    print("  CALIBRATE THE GAMMA ARM AGAINST THE REAL CASE. cam_219:14's dark half")
    print("  is the thing being modelled, so the row that matters is whichever gamma")
    print("  lands near ITS mean luma -- compare the luma column against the native")
    print("  figure above and read the drop at the matching row, not the largest one.")

    print()
    print("  HOW TO READ IT. A drop whose CI clears zero is a CAUSED effect -- the")
    print("  halves differ in nothing else. Compare the magnitude against the numbers")
    print("  the identity decisions actually turn on: cam_219's known same-person")
    print("  fragment pair measured 0.574, and a PROVEN co-present stranger pair in")
    print("  the same camera measured 0.843. If scale-4x costs ~0.3, resolution alone")
    print("  can move a pair across that entire range and a resolution-aware rule is")
    print("  justified. If it costs ~0.05, resolution is a red herring here and an")
    print("  absolute size gate would have cost recall for nothing.")
    print()
    print("  STILL NOT ESTABLISHED BY THIS SCRIPT: that a size gate is the right")
    print("  remedy. It shows a mechanism and its size, not a policy. The mismatch")
    print("  finding (92 vs 101 px scored 0.858 while 90 vs 606 px scored -0.026)")
    print("  argues against an absolute floor, and every cam_219 crop is already")
    print("  371-766 px, so no size gate touches the reported cam_219 bug at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
