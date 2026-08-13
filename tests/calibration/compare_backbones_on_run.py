"""
compare_backbones_on_run.py  --  swap the EMBEDDER, hold everything else fixed.

    python tests/calibration/compare_backbones_on_run.py 20260806_095141
    python tests/calibration/compare_backbones_on_run.py 20260806_095141 \
        --onnx ~/resnet50_market1501_aicity156.onnx \
        --onnx-size 256x128 --onnx-std 0.226,0.226,0.226
    python tests/calibration/compare_backbones_on_run.py 20260806_095141 \
        --dir clips_20260806_095141

WHY THIS EXISTS. Every threshold in config.yaml lives in the CURRENT model's
score space, so adopting a different backbone voids all of them at once
(extractor.py's own header says so). Worse, the usual way to try one -- change
`reid.model`, capture, look -- re-runs detection and tracking, which renumbers
every track and therefore destroys the labels needed to tell whether it helped.
That is how a month goes by with no answer.

This changes ONE thing. The stored payload already carries each observation's
`bbox`, `camera`, `track_id` and `ts`, and the clip's sidecar carries a
wall-clock stamp per frame. So the exact same pixels, in the exact same boxes,
under the exact same track ids, can be re-embedded by any model. Nothing
upstream moves, nothing gets renumbered, and no capture is needed.

WHAT IT MEASURES, and why this and not mAP. The failure here is not that scores
are low, it is that the same-person and different-person DISTRIBUTIONS OVERLAP:
on run 20260806_095141 p95 of "top different" was 0.660 against p5 of "worst
same" 0.355. A benchmark number cannot tell you whether a model closes that; the
overlap on your own footage can. Per model:

  * PROVABLE STRANGERS -- tracklet pairs co-present in ONE camera. Two boxes in
    one frame are two people; that is proof, not inference, and needs no labels.
  * SPLIT-HALF SAME-PERSON -- each tracklet's first half against its second. One
    tracklet is one person over time, so this is a same-person pair for free.
    OPTIMISTIC (see the function's own note) but comparable ACROSS models.
  * RANDOM-PAIR CONTROL -- the collapse check. A model can lower stranger scores
    by separating identities OR by mapping every crop to one direction, and both
    look identical if the stranger number is all you watch.
  * LABELLED SAME-PERSON from calibration/tracklet_pairs.jsonl, when it exists.
  * THE HEADLINE PAIR: cam_213:0019 vs cam_213:0020 on run 20260806_095141 --
    2.6s of co-presence, 9 co-occurring instants, so provably two people --
    which FastReID scores 0.779 here. A model that still puts two strangers up
    there has not fixed anything, whatever its benchmark says.

READ-ONLY. Never writes to the store, never touches config.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


# --------------------------------------------------------------- fetch

def fetch_observations(url, collection, run_id):
    """-> {(camera, track_id): [(ts, bbox), ...]}, chronological."""
    out = defaultdict(list)
    offset = None
    while True:
        body = {"limit": 500, "with_vector": False,
                "with_payload": ["camera", "track_id", "ts", "bbox"],
                "filter": {"must": [{"key": "run_id",
                                     "match": {"value": run_id}}]}}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{url}/collections/{collection}/points/scroll",
                          json=body, timeout=300)
        r.raise_for_status()
        res = r.json()["result"]
        for p in res["points"]:
            pl = p["payload"]
            if pl.get("ts") is None or not pl.get("bbox"):
                continue
            out[(pl["camera"], int(pl["track_id"]))].append(
                (float(pl["ts"]), [float(v) for v in pl["bbox"]]))
        offset = res.get("next_page_offset")
        if offset is None:
            break
    for k in out:
        out[k].sort()
    return dict(out)


def load_crops(observations, clip_dir, cap_per_tracklet):
    """Cut every stored box out of the clip it came from.

    Joined on WALL CLOCK, never on frame index. The clip holds PROCESSED frames
    -- a run may read 2138 and keep 2069 -- so clip index N is not payload
    `frame` N, and matching positionally would pair one moment's box with
    another moment's pixels. `frame_ts` is the only thing relating the two, the
    same rule reconcile follows for cross-camera timing.

    Crops come out in ASCENDING FRAME ORDER within each camera, which is what
    makes split_half_pairs() a genuine first-half-vs-second-half in time rather
    than an arbitrary partition.
    """
    import cv2

    by_cam = defaultdict(list)
    for (cam, tid), obs in observations.items():
        for ts, bbox in obs:
            by_cam[cam].append((ts, bbox, (cam, tid)))

    crops, owners = [], []
    for cam, items in sorted(by_cam.items()):
        side = os.path.join(clip_dir, f"._live_src_{cam}.annotations.json")
        clip = os.path.join(clip_dir, f"._live_src_{cam}.mp4")
        if not (os.path.exists(side) and os.path.exists(clip)):
            print(f"  {cam}: clip or sidecar missing -- skipped")
            continue
        with open(side) as f:
            meta = json.load(f)
        frame_ts = np.asarray(meta.get("frame_ts") or [], dtype=np.float64)
        if frame_ts.size == 0:
            print(f"  {cam}: sidecar has no frame_ts -- cannot join. Skipped.")
            continue

        # Cap per tracklet BEFORE decoding: a 900-frame tracklet contributes no
        # more to a prototype than an evenly spread sample of it, and decoding
        # every frame of every camera is the slow part.
        per_tr = defaultdict(list)
        for ts, bbox, key in items:
            per_tr[key].append((ts, bbox))
        wanted = defaultdict(list)
        for key, lst in per_tr.items():
            lst.sort()
            if cap_per_tracklet and len(lst) > cap_per_tracklet:
                idx = np.linspace(0, len(lst) - 1, cap_per_tracklet).astype(int)
                lst = [lst[i] for i in idx]
            for ts, bbox in lst:
                wanted[int(np.abs(frame_ts - ts).argmin())].append((bbox, key))

        cap = cv2.VideoCapture(clip)
        if not cap.isOpened():
            print(f"  {cam}: cannot open {clip}")
            continue
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got = 0
        for fi in sorted(wanted):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            for bbox, key in wanted[fi]:
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                crops.append(frame[y1:y2, x1:x2].copy())
                owners.append(key)
                got += 1
        cap.release()
        print(f"  {cam}: {got} crop(s) from {len(wanted)} frame(s)")
    return crops, owners


# --------------------------------------------------------------- embedders

def embed_fastreid(crops, batch):
    """The SHIPPING model, via the pipeline's own extractor.

    from_config() rather than a hardcoded path, so this cannot drift from what
    actually runs -- extractor.py records that five diagnostic scripts once
    hardcoded a checkpoint absent from the tree and had been silently measuring
    nothing for months.
    """
    from reid.extractor import ReIDExtractor
    ex = ReIDExtractor.from_config()
    print(f"  {ex.describe()}")
    out = []
    for i in range(0, len(crops), batch):
        out.append(ex.extract_batch(crops[i:i + batch]))
    return np.concatenate(out, axis=0)


def embed_onnx(crops, path, size, mean, std, rgb, batch):
    """Any ONNX ReID model -- ReIdentificationNet, TransReID, an export.

    PREPROCESSING IS NOT GUESSABLE AND MUST COME FROM THE MODEL CARD. Feed a
    model the wrong input size, channel order or normalisation and it still
    returns confident-looking vectors -- no error, just worse matching, which is
    exactly the silent failure this harness exists to avoid.

    For NVIDIA ReIdentificationNet (resnet50_market1501_aicity156.onnx) the TAO
    spec gives input_size [256, 128], pixel_mean [0.485, 0.456, 0.406] and
    pixel_std [0.226, 0.226, 0.226] -- note the UNIFORM std, which is NOT the
    ImageNet [0.229, 0.224, 0.225] this function defaults to. Its raw output is
    not L2-normalised; prototypes() normalises before averaging, so that part is
    already handled.

    Sanity test after a run: if provable-stranger p95 comes back near 0.99, or
    if the random-pair control rises alongside it, preprocessing is wrong and
    every crop is collapsing toward one direction.
    """
    import cv2
    import onnxruntime as ort

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(path, providers=providers)
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    print(f"  onnx {os.path.basename(path)}  input {iname}{ishape}  {providers[0]}")

    H, W = size
    mean_a = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_a = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    out = []
    for i in range(0, len(crops), batch):
        chunk = crops[i:i + batch]
        arr = np.empty((len(chunk), 3, H, W), dtype=np.float32)
        for j, c in enumerate(chunk):
            im = cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR)
            if rgb:
                im = im[:, :, ::-1]
            im = im.transpose(2, 0, 1).astype(np.float32) / 255.0
            arr[j] = (im - mean_a) / std_a
        feats = sess.run(None, {iname: arr})[0]
        feats = np.asarray(feats, dtype=np.float32).reshape(len(chunk), -1)
        out.append(feats)
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------- prototypes

def _proto_of(features, rows):
    """Renormalised mean of the given rows, or None if degenerate."""
    m = features[rows].astype(np.float32)
    m = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
    p = m.mean(axis=0)
    n = np.linalg.norm(p)
    return None if n <= 0 else p / n


def prototypes(features, owners):
    """Mean of L2-normalized vectors, renormalized -- reconcile's _prototype().

    Deliberately identical, including the fact that it is UNWEIGHTED, so the
    comparison measures the model rather than the difference between two
    pooling functions.
    """
    idx = defaultdict(list)
    for i, k in enumerate(owners):
        idx[k].append(i)
    protos = {}
    for k, rows in idx.items():
        p = _proto_of(features, rows)
        if p is not None:
            protos[k] = p
    return protos


# --------------------------------------------------------------- evaluation

def spans(observations):
    return {k: (v[0][0], v[-1][0]) for k, v in observations.items() if v}


def provable_strangers(observations, min_overlap):
    """Tracklet pairs co-present in ONE camera -> provably two people.

    Same camera only. Cross-camera co-presence proves nothing in this deployment
    because cam_219 and cam_224 share a room, so a person can legitimately be in
    both at once (CLAUDE.md 6.1).

    min_overlap exists because a single shared frame is not co-presence: a track
    that ByteTrack dropped and re-acquired with one frame of double detection
    would be counted as two people and would score HIGH, manufacturing exactly
    the false stranger this measurement must not contain.
    """
    sp = spans(observations)
    keys = sorted(sp)
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] != b[0]:
                continue
            ov = min(sp[a][1], sp[b][1]) - max(sp[a][0], sp[b][0])
            if ov >= min_overlap:
                out.append((a, b, ov))
    return out


def split_half_pairs(features, owners, min_obs=6):
    """Same-person pairs with no labels: each tracklet's first half vs its second.

    One tracklet is one person over time, so its two halves are the same person
    by construction. That makes the same-person side of the separation
    measurable on any run, with no operator and no label file -- which matters,
    because with only the stranger side a measurement can show a model is NOT
    better but never that it is.

    OPTIMISTIC BY CONSTRUCTION, and this project has already been burned by
    exactly this control: CLAUDE.md records the published "any bar in (0.434,
    0.810) is perfect" window as having been measured by splitting one
    continuous track in half, while a genuine re-appearance scored 0.574 on the
    same footage against a 0.942 split-half control. Two halves of one track
    share clothing, lighting and usually viewpoint; a real re-appearance shares
    none of those reliably.

    So read the absolute value as a CEILING. What it is good for is COMPARING
    models on identical crops, and one direction is decisive: a model that
    cannot separate provable strangers from the EASY same-person case has no
    chance at the hard one.

    A chimeric tracklet -- one track_id covering two people, confirmed in
    cam_219 by contact sheet -- scores low here for a reason that is not the
    model's fault, so the worst few are reported by name rather than buried in a
    percentile. min_obs keeps each half large enough that one bad crop cannot
    dominate its prototype.
    """
    idx = defaultdict(list)
    for i, k in enumerate(owners):
        idx[k].append(i)

    pairs = []
    for k, rows in idx.items():
        if len(rows) < min_obs:
            continue
        mid = len(rows) // 2
        a = _proto_of(features, rows[:mid])
        b = _proto_of(features, rows[mid:])
        if a is not None and b is not None:
            pairs.append((k, float(a @ b)))
    return pairs


def random_pair_scores(protos, n=2000, seed=0):
    """Cosines between random tracklet prototypes -- the COLLAPSE control.

    A model can lower stranger scores two ways: by separating identities, or by
    mapping every crop to nearly the same direction. Both look like success if
    the stranger number is all that is watched, and the second is exactly what a
    wrong input size, channel order or normalisation produces.

    Random tracklet pairs are mostly different people, so under a healthy model
    they sit near the stranger distribution. Under a collapsed one everything
    rises toward 1.0 together. The reading is therefore comparative, never
    absolute: strangers falling while this HOLDS is real separation; both moving
    together is collapse.
    """
    keys = sorted(protos)
    if len(keys) < 2:
        return []
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i, j = rng.integers(0, len(keys), 2)
        if i == j:
            continue
        out.append(float(protos[keys[i]] @ protos[keys[j]]))
    return out


def load_labels(path, run_id):
    """Operator-confirmed same-person pairs, if a label file exists."""
    if not os.path.exists(path):
        return []
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("run_id") != run_id or not rec.get("same"):
                continue
            a, b = rec.get("a"), rec.get("b")
            if a and b:
                pairs.append(((a[0], int(a[1])), (b[0], int(b[1]))))
    return pairs


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def report(tag, protos, strangers, labels, headline, halves, randoms):
    print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}")
    print(f"  {len(protos)} tracklet prototype(s), "
          f"dim={len(next(iter(protos.values())))}")

    diff = [float(protos[a] @ protos[b])
            for a, b, _ in strangers if a in protos and b in protos]
    same = [float(protos[a] @ protos[b])
            for a, b in labels if a in protos and b in protos]
    d = np.asarray(diff) if diff else None

    if d is not None:
        print(f"\n  PROVABLE STRANGERS (co-present in one camera), n={len(d)}")
        print(f"    p50 {pct(d,50):.3f}   p95 {pct(d,95):.3f}   max {d.max():.3f}")
    else:
        print("\n  PROVABLE STRANGERS: none found -- no two tracklets overlap in "
              "time in any single camera. Nothing to measure against.")

    if same:
        s = np.asarray(same)
        print(f"\n  LABELLED SAME-PERSON, n={len(s)}")
        print(f"    p50 {pct(s,50):.3f}   p5  {pct(s,5):.3f}   min {s.min():.3f}")
        if d is not None:
            print(f"\n  SEPARATION (labelled): p5(same) - p95(different) "
                  f"= {pct(s, 5) - pct(d, 95):+.3f}")
            print("    Positive = a bar exists between them. Negative = they")
            print("    OVERLAP and no single threshold separates.")
    else:
        print("\n  LABELLED SAME-PERSON: none for this run "
              "(tests/calibration/review_links.py <run> --label records them).")

    if halves:
        h = np.asarray([v for _, v in halves])
        print(f"\n  SPLIT-HALF SAME-PERSON (no labels needed), n={len(h)}")
        print(f"    p50 {pct(h,50):.3f}   p5  {pct(h,5):.3f}   min {h.min():.3f}")
        if d is not None:
            print(f"\n  SEPARATION (split-half): p5(same) - p95(different) "
                  f"= {pct(h, 5) - pct(d, 95):+.3f}   <-- RANK MODELS ON THIS")
            print("    OPTIMISTIC: two halves of one track share clothing,")
            print("    lighting and usually viewpoint, so this is a CEILING. A")
            print("    real re-appearance is far harder -- 0.574 measured on this")
            print("    footage against a 0.942 split-half control. Use it to rank,")
            print("    never as an absolute.")
        worst = sorted(halves, key=lambda kv: kv[1])[:3]
        if worst:
            print("    worst tracklets: " + ", ".join(
                f"{k[0]}:{k[1]:04d} {v:.3f}" for k, v in worst))
            print("    (a very low one is usually a CHIMERA -- one track_id over")
            print("     two people -- not a model failure)")

    if randoms:
        r = np.asarray(randoms)
        tail = f"   (strangers p95 {pct(d,95):.3f})" if d is not None else ""
        print(f"\n  RANDOM-PAIR CONTROL, n={len(r)}")
        print(f"    p50 {pct(r,50):.3f}   p95 {pct(r,95):.3f}{tail}")
        print("    COLLAPSE CHECK: strangers falling while this HOLDS = real")
        print("    separation. Both falling together = the model maps everything")
        print("    to one direction and the stranger number means nothing.")

    if headline:
        a, b = headline
        if a in protos and b in protos:
            v = float(protos[a] @ protos[b])
            print(f"\n  HEADLINE PAIR  {a[0]}:{a[1]:04d} vs {b[0]}:{b[1]:04d}")
            print(f"    {v:.3f}   (FastReID scores these 0.779 here; they are")
            print(f"     provably two people -- 2.6s co-presence, 9 shared instants)")
    return {"diff": diff, "same": same, "halves": halves, "randoms": randoms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--dir", default=".", help="directory holding the clips")
    ap.add_argument("--url", default=os.environ.get("QDRANT_URL",
                                                    "http://localhost:6333"))
    ap.add_argument("--collection", default="persons")
    ap.add_argument("--labels", default="calibration/tracklet_pairs.jsonl")
    ap.add_argument("--cap", type=int, default=64,
                    help="max crops per tracklet (evenly spread)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-overlap", type=float, default=0.5,
                    help="seconds of co-presence before a pair counts as proof")
    ap.add_argument("--min-half-obs", type=int, default=6,
                    help="min crops for a tracklet to give a split-half pair")
    ap.add_argument("--skip-fastreid", action="store_true")
    ap.add_argument("--onnx", default="", help="path to an ONNX ReID model")
    ap.add_argument("--onnx-size", default="256x128", help="HxW")
    ap.add_argument("--onnx-mean", default="0.485,0.456,0.406")
    ap.add_argument("--onnx-std", default="0.229,0.224,0.225",
                    help="ImageNet default; ReIdentificationNet wants "
                         "0.226,0.226,0.226 per its TAO spec")
    ap.add_argument("--onnx-bgr", action="store_true",
                    help="feed BGR instead of RGB")
    args = ap.parse_args()

    print(f"[cmp] run {args.run_id}")
    obs = fetch_observations(args.url, args.collection, args.run_id)
    if not obs:
        raise SystemExit(
            f"[cmp] no observations for {args.run_id}. The store may have been "
            f"wiped since -- saved clips alone are not enough to re-render or "
            f"re-embed a run.")
    print(f"[cmp] {len(obs)} tracklet(s), "
          f"{sum(len(v) for v in obs.values())} observation(s)")

    print("[cmp] cutting crops from clips...")
    crops, owners = load_crops(obs, args.dir, args.cap)
    if not crops:
        raise SystemExit("[cmp] no crops -- check --dir points at this run's clips")
    print(f"[cmp] {len(crops)} crop(s)")

    strangers = provable_strangers(obs, args.min_overlap)
    labels = load_labels(args.labels, args.run_id)
    print(f"[cmp] {len(strangers)} provable stranger pair(s), "
          f"{len(labels)} labelled same-person pair(s)")

    headline = (("cam_213", 19), ("cam_213", 20))

    def run_report(tag, feats):
        protos = prototypes(feats, owners)
        return report(tag, protos, strangers, labels, headline,
                      halves=split_half_pairs(feats, owners, args.min_half_obs),
                      randoms=random_pair_scores(protos))

    if not args.skip_fastreid:
        print("\n[cmp] embedding with the SHIPPING model...")
        run_report("FastReID (shipping)", embed_fastreid(crops, args.batch))

    if args.onnx:
        H, W = (int(x) for x in args.onnx_size.lower().split("x"))
        print(f"\n[cmp] embedding with {os.path.basename(args.onnx)}...")
        feats = embed_onnx(
            crops, os.path.expanduser(args.onnx), (H, W),
            [float(x) for x in args.onnx_mean.split(",")],
            [float(x) for x in args.onnx_std.split(",")],
            not args.onnx_bgr, args.batch)
        run_report(f"ONNX {os.path.basename(args.onnx)}", feats)

    print(f"""
{'=' * 78}
HOW TO READ THIS
{'=' * 78}
  1. SEPARATION (split-half) IS THE RANKING NUMBER. p5(same) - p95(different),
     on identical crops, so it is comparable across models. Higher is better.
     Its ABSOLUTE value is optimistic -- split-half is the easy same-person
     case -- so use it to rank, not to declare the problem solved.

  2. THE RANDOM-PAIR CONTROL DECIDES WHETHER TO BELIEVE (1). A lower stranger
     p95 means nothing on its own: a model that maps every crop to the same
     direction lowers it too. Real separation shows as strangers falling while
     random pairs hold. Both falling together is collapse, and the usual cause
     is wrong preprocessing rather than a bad model.

  3. THE HEADLINE PAIR is one operator-verifiable case. FastReID puts two
     provably different people at 0.779. Lower is better.

  A DIFFERENT DIMENSION MEANS A DIFFERENT COLLECTION. Adopting a model whose
  width is not 2048 rebuilds Qdrant, and every threshold in config.yaml is void
  because it is a different feature space -- extractor.py says so explicitly.
  Budget for a full re-derivation, not a config edit.

  AND THIS DOES NOT PROVE IDENTITIES IMPROVE. It measures whether the scores
  separate. Whether that becomes correct reids still needs the model wired in
  as a backend, a capture, and a watch.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())