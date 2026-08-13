"""
Reconcile threshold measurement -- prototype vs prototype.

Produces REMEDIATION_PLAN.md Part H.4. This is the measurement that matters MOST
for the product, because reconcile is what assigns the ids in output_<cam>.mp4.
Reconcile compares TRACKLET PROTOTYPES (means over many observations, so denoised
on BOTH sides), which is a different and much cleaner distribution than the
single-crop-vs-bank scores in measure_score_separation.py. The two are NOT
comparable and their thresholds must be set independently.

    python tests/calibration/measure_reconcile_thresholds.py [video] [frames] [stride]

Phase 1 of reconcile merges TIME-DISJOINT same-camera tracklets at
`identity.reconcile.same_camera_threshold` (0.90). We simulate a
dropped-then-reacquired track by splitting one track's observations into two
disjoint halves -- which is exactly what fragmentation produces -- and ask
whether the two halves' prototypes clear the bar.
"""

import sys

import numpy as np

from _common import (bootstrap, pick_video, sample_frames, reid_weights,
                     reid_model, reid_tap,
                     DETECT_WEIGHTS, collect_track_embeddings,
                     proven_distinct_pairs, header, footnote_sample_size)

bootstrap()

from reid.extractor import ReIDExtractor
from detector import PersonDetector
from identity.reconcile import _prototype

VIDEO = pick_video(sys.argv[1] if len(sys.argv) > 1 else None)
NFRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 90
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 4

SAME_CAM_THR = 0.90        # identity.reconcile.same_camera_threshold
CROSS_THR = 0.63           # identity.reconcile.threshold

# Backbone AND tap come from config (env-overridable) -- these thresholds are
# specific to one feature space, so measuring a different model or tap than the
# pipeline will run makes every number below inapplicable.
ex = ReIDExtractor(weights=reid_weights(), model=reid_model(), tap=reid_tap(),
                   device="cpu")
print(f"[calib] ReID: {ex.describe()}")
det = PersonDetector(model_path=DETECT_WEIGHTS, confidence_threshold=0.4,
                     person_class_id=0, tracker_config="bytetrack.yaml",
                     pose_ensemble=None, iou=0.60)

frames = sample_frames(VIDEO, NFRAMES, STRIDE)
print(f"[calib] {VIDEO}: {len(frames)} frames @ stride {STRIDE}, "
      f"{frames[0].shape[1]}x{frames[0].shape[0]}")

by_track, cooccur, _ = collect_track_embeddings(
    frames, det, lambda c: ex.extract_batch(c))
usable, pairs, excluded = proven_distinct_pairs(by_track, cooccur, min_obs=4)

print("\n  observations per track:")
for t in sorted(by_track):
    print(f"    track {t:3d}: {len(by_track[t]):4d}")


header("A. PHASE 1 -- do two TIME-DISJOINT halves of ONE person merge at 0.90?")
print("  (each track split first-half / second-half = a dropped-then-reacquired track)\n")
frag = []
for t in sorted(by_track):
    vs = by_track[t]
    if len(vs) < 8:
        continue
    mid = len(vs) // 2
    pa, pb = _prototype(vs[:mid]), _prototype(vs[mid:])
    if pa is None or pb is None:
        continue
    s = float(pa @ pb)
    frag.append(s)
    verdict = "MERGES" if s >= SAME_CAM_THR else "FAILS -> becomes a SECOND identity"
    print(f"    track {t:3d}: fragment-vs-fragment cosine = {s:.3f}   {verdict}")

if frag:
    fs = np.array(frag)
    print(f"\n    n={len(fs)} mean={fs.mean():.3f} min={fs.min():.3f} max={fs.max():.3f}")
    print(f"\n    {'threshold':>10}{'fragments that MERGE':>24}")
    for thr in (0.75, 0.80, 0.85, 0.90, 0.95):
        mark = "  <-- configured" if abs(thr - SAME_CAM_THR) < 1e-9 else ""
        print(f"    {thr:>10.2f}{100 * (fs >= thr).mean():>23.1f}%{mark}")


header("B. DIFFERENT PEOPLE, prototype vs prototype -- must stay BELOW the bars")
protos = {t: _prototype(v) for t, v in usable.items()}
diff = [float(protos[a] @ protos[b]) for a, b in pairs
        if protos.get(a) is not None and protos.get(b) is not None]
if diff:
    dv = np.array(diff)
    print(f"    n={len(dv)} mean={dv.mean():.3f} p95={np.percentile(dv, 95):.3f} "
          f"MAX={dv.max():.3f}")
    print(f"\n    {'threshold':>10}{'WRONG-PERSON merges':>24}")
    for thr, label in ((CROSS_THR, "reconcile.threshold (cross-cam)"),
                       (SAME_CAM_THR, "same_camera_threshold")):
        print(f"    {thr:>10.2f}{100 * (dv >= thr).mean():>23.1f}%   {label}")
    print("\n    NOTE: reconcile routes any cluster pair SHARING a camera to the")
    print("    same-camera bar via pair_threshold(), so same-camera strangers are")
    print("    protected at 0.90, not 0.63. Only camera-DISJOINT clusters get 0.63.")
    print("    That routing is what the stale-`roots` defect can bypass (Phase 4).")


header("C. IS THERE A BAR THAT MERGES FRAGMENTS AND REJECTS STRANGERS?")
if frag and diff:
    fs, dv = np.array(frag), np.array(diff)
    lo, hi = dv.max(), fs.min()
    print(f"    same-person fragments : min={fs.min():.3f}  p5={np.percentile(fs, 5):.3f}")
    print(f"    different people      : MAX={dv.max():.3f}  p95={np.percentile(dv, 95):.3f}")
    print(f"    worst-case separation : {hi - lo:+.3f}")
    if hi > lo:
        inside = lo < SAME_CAM_THR < hi
        print(f"    -> any bar in ({lo:.3f}, {hi:.3f}) is perfect on this sample")
        print(f"    -> configured {SAME_CAM_THR:.2f} is "
              f"{'INSIDE' if inside else 'OUTSIDE'} that window")
        if not inside and SAME_CAM_THR >= hi:
            print("       (too STRICT: it rejects genuine fragments for no benefit)")
    else:
        print("    -> OVERLAPPING: no single same-camera bar separates them here")

    print("\n    CAUTION: 'MAX' grows with sample size, and n is small. The stable")
    print("    finding is the DIRECTION (0.90 sits above the fragment distribution),")
    print("    not the exact window. Confirm with the Phase 9 sweep.")


header("D. min_tracklet_observations -- which tracklets get SUPPRESSED?")
print("  A suppressed tracklet has its reid_id CLEARED, so build_gid_map finds no")
print("  id and drawing.py falls back to a bare 'ID <track_id>' -- which on screen")
print("  reads as the identity vanishing mid-walk.\n")
counts = sorted(len(v) for v in by_track.values())
print(f"    observations per tracklet: {counts}")
for m in (1, 2, 3, 5):
    n = sum(1 for c in counts if c < m)
    mark = "  <-- configured" if m == 3 else ""
    print(f"    min_tracklet_observations={m}: {n}/{len(counts)} suppressed{mark}")
print("\n  Reconcile ALSO returns no ids at all when fewer than 2 tracklets survive")
print("  (`if len(keys) < 2: return {}`) -- see Phase 4, #25.")
print("  This clip is lightly fragmented; under real frame dropping short")
print("  tracklets are far more common.")

footnote_sample_size(usable, pairs, excluded)
