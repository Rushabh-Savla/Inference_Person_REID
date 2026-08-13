"""
Compare ReID BACKBONES on identical crops, with threshold-free metrics.

TWO MODES
=========
  # A) one video: detect+track here, single camera
  python tests/calibration/compare_backbones.py register_file.avi 60 6

  # B) a FROZEN RUN's clips + sidecars: real multi-camera, no camera time
  python tests/calibration/compare_backbones.py --clips /path/to/run/dir

  # explicit pairing (default = config.yaml's model + osnet_ain baseline)
  ... --models fastreid_sbs_R101_ibn=src/reid/weights/msmt_sbs_R101-ibn.pth,\
osnet_ain_x1_0=src/reid/weights/osnet_ain_x1_0.pth

Mode B is the one that matters. The live path keeps a CLEAN processed-frame clip
(`._live_src_<cam>.mp4`) plus an `.annotations.json` sidecar holding every box and
track_id (render.py `_capture` -- "No drawing here"), so a finished run can be
RE-EMBEDDED with any backbone offline. That is a real multi-camera comparison for
zero camera time.

WHY NO COSINE BAR IS REPORTED
=============================
Every threshold in config.yaml lives in ONE model's feature space. Comparing two
backbones with a bar measures the bar, not the backbone; picking a fresh bar per
candidate re-introduces exactly the fitting-to-noise problem review_links.py's
docstring warns about. So everything here is RANKING-based and scale-free:

  * ROC AUC  -- P(same-person pair scores above different-person pair). 0.5 is
                chance. Invariant to any monotonic rescaling of cosine, which is
                what changing backbone does.
  * R@1      -- for a held-out observation, is the nearest prototype its own?
  * margin   -- p5(same) - p95(diff). Printed because Part H is written in it,
                NOT because it transfers across models. Do not compare it.

THE THREE METHODOLOGICAL RULES
==============================
1. WITHIN one camera, co-occurrence in a frame proves two tracks are different
   people: one person cannot be two simultaneous detections.

2. ACROSS cameras it proves NOTHING HERE. These cameras overlap -- cam_224 and
   cam_219 share a room -- so one person legitimately appears in two cameras at
   the same instant. Using cross-camera co-occurrence as a negative would label
   the very links the system exists to make as errors. So cross-camera evidence
   comes ONLY from operator labels (calibration/link_labels.jsonl).

3. Bank queries are held OUT of the prototype they are scored against, or a query
   matches itself at ~1.0 and inflates the same-person side.

WHAT THIS STILL CANNOT DO
=========================
The labelled cross-camera set is tiny (11 pairs, 3 of them "different", one of
those inferred rather than stated). It can catch a backbone that is badly wrong
cross-camera; it cannot finely rank two good ones. Growing that file with
`review_links.py --label` is what buys resolving power.
"""

import json
import os
import sys
from itertools import combinations
from types import SimpleNamespace

import numpy as np

from _common import (bootstrap, pick_video, sample_frames, reid_weights,
                     reid_model, DETECT_WEIGHTS, MIN_CROP_H, MIN_CROP_W,
                     collect_track_embeddings, proven_distinct_pairs, unit,
                     describe, header)

bootstrap()

import cv2

from reid.extractor import ReIDExtractor
from detector import crop_person


def arg(flag, default=None):
    return (sys.argv[sys.argv.index(flag) + 1]
            if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv)
            else default)


FLAGS = ("--clips", "--models", "--labels", "--run", "--max-per-track",
         "--min-obs", "--device")
_flagvals = {sys.argv[sys.argv.index(f) + 1] for f in FLAGS
             if f in sys.argv and sys.argv.index(f) + 1 < len(sys.argv)}
positional = [a for a in sys.argv[1:]
              if not a.startswith("--") and a not in _flagvals]

CLIPS = arg("--clips")
LABELS = arg("--labels", os.path.join("calibration", "link_labels.jsonl"))
RUN_FILTER = arg("--run")
MAX_PER_TRACK = int(arg("--max-per-track", "16"))
MIN_OBS = int(arg("--min-obs", "6"))
DEVICE = arg("--device", "cpu")

_spec = arg("--models")
if _spec:
    MODELS = [tuple(p.split("=", 1)) for p in _spec.split(",") if p.strip()]
else:
    MODELS = [(reid_model(), reid_weights())]
    _prev = "src/reid/weights/osnet_ain_x1_0.pth"
    if reid_model() != "osnet_ain_x1_0" and os.path.exists(_prev):
        MODELS.append(("osnet_ain_x1_0", _prev))


# --------------------------------------------------------------- collection

def collect_from_clips(clip_dir):
    """Frozen run -> ({key: [crop, ...]}, {within-camera co-occurring key pairs}).

    Mirrors rerender_from_clips.load_clips (kept local so this script has no
    dependency on that one's module-level state). Keys are "cam:track_id", which
    is exactly the form link_labels.jsonl uses, so labels join directly.
    """
    import glob
    crops, cooccur = {}, set()
    clips = sorted(glob.glob(os.path.join(clip_dir, "._live_src_*.mp4")))
    if not clips:
        raise SystemExit(f"[calib] no ._live_src_*.mp4 in {clip_dir!r}. "
                         f"A run only keeps clips when live.render.keep_frames "
                         f"was on -- runs before that sidecar existed cannot be "
                         f"re-embedded (the box geometry was only in memory).")
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
        kept = 0
        for i, boxes in enumerate(anns):
            ok, frame = cap.read()
            if not ok:
                break
            if not boxes:
                continue
            tids = []
            for b in boxes:
                tid = b.get("track_id")
                if tid is None:
                    continue
                key = f"{cam}:{tid}"
                tids.append(tid)
                if len(crops.get(key, ())) >= MAX_PER_TRACK:
                    continue
                crop = crop_person(frame, SimpleNamespace(
                    x1=int(b["x1"]), y1=int(b["y1"]),
                    x2=int(b["x2"]), y2=int(b["y2"])))
                if crop is None or crop.size == 0:
                    continue
                h, w = crop.shape[:2]
                if h < MIN_CROP_H or w < MIN_CROP_W:
                    continue
                crops.setdefault(key, []).append(crop)
                kept += 1
            # rule 1: within ONE camera, same-frame => different people
            for a, b in combinations(sorted(set(tids)), 2):
                cooccur.add((f"{cam}:{a}", f"{cam}:{b}"))
        cap.release()
        ntracks = len({k for k in crops if k.startswith(cam + ":")})
        print(f"  {cam:<12} {len(anns):>5} annotated frames -> {ntracks:>3} "
              f"tracklets, {kept:>4} crops kept (cap {MAX_PER_TRACK}/tracklet)")
    return crops, cooccur


def collect_from_video(video, nframes, stride):
    """One video -> the same two structures, by running detect+track ONCE."""
    from detector import PersonDetector
    frames = sample_frames(video, nframes, stride)
    print(f"  video={video} frames={len(frames)} stride={stride}")
    det = PersonDetector(model_path=DETECT_WEIGHTS, confidence_threshold=0.4,
                         person_class_id=0, tracker_config="bytetrack.yaml",
                         pose_ensemble=None, iou=0.60)
    cam = os.path.splitext(os.path.basename(video))[0]
    # embed_fn returns the crops themselves, so by_track holds CROPS. Reuses the
    # harness's exact crop filtering rather than reimplementing it.
    by_track, cooc, _ = collect_track_embeddings(
        frames, det, embed_fn=lambda cs: cs)
    crops = {f"{cam}:{t}": v for t, v in by_track.items()}
    return crops, {(f"{cam}:{a}", f"{cam}:{b}") for a, b in cooc}


def load_labels(path, run_filter=None):
    """-> [(key_a, key_b, verdict, stated)] from the operator label file."""
    if not os.path.exists(path):
        print(f"  [labels] {path} not found -- cross-camera section skipped.")
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "_comment" in d or not d.get("a") or not d.get("b"):
                continue
            if run_filter and d.get("run_id") != run_filter:
                continue
            out.append((d["a"], d["b"], d.get("verdict"),
                        bool(d.get("stated"))))
    return out


# --------------------------------------------------------------- metrics

def roc_auc(pos, neg):
    """AUC via the Mann-Whitney U identity, ties = 0.5. Hand-written because
    sklearn is not a dependency of this project and this is a dozen lines."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def holdout(obs):
    """One tracklet's observations -> (bank, query), disjoint and INTERLEAVED.

    Interleaved, not first-half/second-half: consecutive frames are highly
    correlated, and a temporal split makes every query systematically later (and
    often more occluded) than its bank, which reads as a model weakness rather
    than a sampling artifact.
    """
    return obs[0::2], obs[1::2]


def evaluate(emb, pairs, labels):
    """emb {key: (N,D)} -> all metrics for one backbone."""
    same_obs, diff_obs = [], []
    for e in emb.values():
        for i, j in combinations(range(len(e)), 2):
            same_obs.append(float(e[i] @ e[j]))
    for a, b in pairs:
        for va in emb[a]:
            for vb in emb[b]:
                diff_obs.append(float(va @ vb))

    protos, queries = {}, {}
    for k, e in emb.items():
        bank, query = holdout(list(e))
        if bank and query:
            protos[k] = unit(np.mean(np.stack(bank), axis=0))
            queries[k] = query

    same_p, diff_p = [], []
    for k, qs in queries.items():
        same_p += [float(unit(q) @ protos[k]) for q in qs]
    for a, b in pairs:
        if a in protos and b in queries:
            diff_p += [float(unit(q) @ protos[a]) for q in queries[b]]
        if b in protos and a in queries:
            diff_p += [float(unit(q) @ protos[b]) for q in queries[a]]

    keys = sorted(protos)
    hits = total = 0
    if keys:
        P = np.stack([protos[k] for k in keys])
        for k, qs in queries.items():
            if k not in protos:
                continue
            for q in qs:
                total += 1
                hits += int(keys[int(np.argmax(P @ unit(q)))] == k)

    # --- operator-labelled pairs (the ONLY valid cross-camera evidence) -----
    lab_same, lab_diff, lab_rows, lab_missing = [], [], [], 0
    for a, b, verdict, stated in labels:
        if a not in protos or b not in protos:
            lab_missing += 1
            continue
        s = float(protos[a] @ protos[b])
        cross = a.split(":")[0] != b.split(":")[0]
        lab_rows.append((a, b, verdict, stated, cross, s))
        (lab_same if verdict == "same" else lab_diff).append(s)

    return dict(
        same_obs=same_obs, diff_obs=diff_obs, same_p=same_p, diff_p=diff_p,
        auc_obs=roc_auc(same_obs, diff_obs), auc_p=roc_auc(same_p, diff_p),
        r1=(hits / total if total else float("nan")), hits=hits, total=total,
        margin=(np.percentile(same_p, 5) - np.percentile(diff_p, 95)
                if same_p and diff_p else float("nan")),
        lab_same=lab_same, lab_diff=lab_diff, lab_rows=lab_rows,
        lab_missing=lab_missing,
        lab_auc=roc_auc(lab_same, lab_diff),
        ntracks=len(emb),
    )


# --------------------------------------------------------------- run it

header("0. CROPS  (collected ONCE; every backbone embeds exactly these)")
if CLIPS:
    crops, cooccur = collect_from_clips(CLIPS)
else:
    video = pick_video(positional[0] if positional else None)
    nframes = int(positional[1]) if len(positional) > 1 else 60
    stride = int(positional[2]) if len(positional) > 2 else 6
    crops, cooccur = collect_from_video(video, nframes, stride)

usable, pairs, excluded = proven_distinct_pairs(crops, cooccur, min_obs=MIN_OBS)
cams = sorted({k.split(":")[0] for k in usable})
print(f"\n  cameras                            : {len(cams)} {cams}")
print(f"  tracklets with >={MIN_OBS} crops         : {len(usable)}")
print(f"  WITHIN-CAMERA proven-distinct pairs: {len(pairs)}")
print(f"  crops each backbone will embed     : {sum(len(v) for v in usable.values())}")
if excluded:
    print(f"  excluded {len(excluded)} pair(s) that never co-occur -- each may be ONE")
    print("  fragmented person, so they are not valid negatives.")
if not pairs:
    raise SystemExit("[calib] no within-camera proven-distinct pairs -- nothing "
                     "to compare. More frames, or a longer run.")

labels = load_labels(LABELS, RUN_FILTER)
ncross = sum(1 for a, b, *_ in labels if a.split(":")[0] != b.split(":")[0])
print(f"  operator labels loaded             : {len(labels)} "
      f"({ncross} cross-camera)"
      + (f" [run={RUN_FILTER}]" if RUN_FILTER else ""))

results = {}
for name, weights in MODELS:
    header(f"MODEL: {name}")
    try:
        # tap=None so each backend uses its own default: OSNet -> post_relu,
        # FastReID -> refuses a tap at all. Forcing one would crash FastReID.
        ex = ReIDExtractor(weights=weights, model=name, device=DEVICE,
                           tap=None, max_batch=32)
    except Exception as e:                                     # noqa: BLE001
        print(f"  [SKIP] could not load: {type(e).__name__}: {e}")
        continue
    print(f"  {ex.describe()}")
    emb = {k: ex.extract_batch(v) for k, v in usable.items()}
    r = evaluate(emb, pairs, labels)
    r["dim"] = ex.embedding_dim
    results[name] = r

    print(describe(r["same_obs"], "same-tracklet obs pairs"))
    print(describe(r["diff_obs"], "different-person obs pairs"))
    print(describe(r["same_p"], "query -> own prototype"))
    print(describe(r["diff_p"], "query -> OTHER prototype"))
    print(f"  {'ROC AUC (observation level)':<34} {r['auc_obs']:.4f}")
    print(f"  {'ROC AUC (prototype level)':<34} {r['auc_p']:.4f}")
    print(f"  {'R@1 (held-out query)':<34} {r['r1']:.4f}  "
          f"({r['hits']}/{r['total']})")
    print(f"  {'margin p5(same)-p95(diff)':<34} {r['margin']:+.4f}   "
          f"<- space-specific, NOT comparable across models")
    allv = np.concatenate(list(emb.values()))
    print(f"  {'negative / exact-zero dims':<34} "
          f"{100 * (allv < 0).mean():.1f}% / {100 * (allv == 0).mean():.1f}%")

    if r["lab_rows"]:
        note = ("" if not r["lab_missing"] else
                f"  [{r['lab_missing']} unusable: tracklet absent from this "
                f"footage, or below the crop floor]")
        print(f"\n  OPERATOR-LABELLED PAIRS (prototype cosine){note}")
        for a, b, verdict, stated, cross, s in sorted(
                r["lab_rows"], key=lambda t: -t[5]):
            print(f"    {s:+.3f}  {verdict:<9} {'cross' if cross else 'same '}-cam  "
                  f"{a} <-> {b}{'' if stated else '   (INFERRED, weaker evidence)'}")
        print(f"    labelled AUC: {r['lab_auc']:.4f}  "
              f"(n_same={len(r['lab_same'])}, n_diff={len(r['lab_diff'])})")
    elif labels:
        print(f"\n  [labels] none of the {len(labels)} labelled tracklets appear "
              f"in this footage -- labels are per-run; pass --run or point --clips "
              f"at the run they were recorded on.")

# --------------------------------------------------------------- verdict
header("VERDICT  (AUC and R@1 are comparable; margin is not)")
if len(results) < 2:
    print("  only one backbone produced results -- nothing to compare.")
else:
    print(f"  {'model':<26} {'dim':>5} {'AUC obs':>9} {'AUC proto':>10} "
          f"{'R@1':>7} {'lab AUC':>8} {'margin':>8}")
    for n, r in results.items():
        print(f"  {n:<26} {r['dim']:>5} {r['auc_obs']:>9.4f} {r['auc_p']:>10.4f} "
              f"{r['r1']:>7.4f} {r['lab_auc']:>8.4f} {r['margin']:>+8.4f}")

    ranked = sorted(results, key=lambda n: (results[n]["auc_p"],
                                            results[n]["r1"]), reverse=True)
    top, second = ranked[0], ranked[1]
    d_auc = results[top]["auc_p"] - results[second]["auc_p"]
    d_r1 = results[top]["r1"] - results[second]["r1"]
    saturated = [n for n, r in results.items()
                 if r["auc_p"] >= 0.9999 and r["r1"] >= 0.9999]

    if len(saturated) > 1:
        # Refusing to name a winner is the POINT. Two models both at a perfect
        # 1.0000 are not "tied but one is better" -- the measurement has no
        # resolving power left, and naming whichever sorted first would be
        # inventing a result.
        print(f"\n  NO VERDICT -- this footage is SATURATED for "
              f"{len(saturated)} backbones: {', '.join(saturated)}")
        print("  A measurement every candidate passes cannot rank them.")
        if not CLIPS:
            print("  This is a single-camera clip. Re-run with --clips against a")
            print("  frozen multi-camera run, which is the only data here with")
            print("  cross-camera pairs in it.")
        else:
            print("  Even on multi-camera clips the WITHIN-camera task can")
            print("  saturate. The discriminating signal is the labelled")
            print("  cross-camera pairs above -- and 11 labels cannot finely rank")
            print("  two good models. Grow them: review_links.py --label")
    elif abs(d_auc) < 1e-4 and abs(d_r1) < 1e-4:
        print(f"\n  NO VERDICT -- {top} and {second} are indistinguishable "
              f"(AUC and R@1 both within 1e-4).")
    else:
        print(f"\n  highest prototype-level AUC: {top} "
              f"({d_auc:+.4f} AUC, {d_r1:+.4f} R@1 vs {second})")
        if not CLIPS:
            print("  NECESSARY, NOT SUFFICIENT: single camera, one clip. Says")
            print("  nothing about cross-camera domain shift. Confirm with --clips.")

print(f"\n  SAMPLE: {len(usable)} tracklets over {len(cams)} camera(s), "
      f"{len(pairs)} within-camera proven-distinct pairs, "
      f"{ncross} cross-camera labels.")
print("  Treat small samples as hypotheses. See REMEDIATION_PLAN.md Part H.")
