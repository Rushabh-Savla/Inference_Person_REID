"""
Is the same-person score drop a CROP-QUALITY problem rather than an appearance one?

Read-only. For one run: per same-camera tracklet pair, the prototype cosine beside
the two tracklets' crop geometry. If low-scoring same-camera pairs are systematically
pairs where one side is small/blurry and the other is not, the defect is the input
gate (reid.quality.min_height = 64 px, upscaled to 384x128 -- a 6x stretch), not the
similarity bar. That distinction matters because pixels-on-target is a physical
quantity that transfers across footage, while a cosine bar demonstrably does not.

    python crop_scale_vs_score.py <run_id> [--url http://localhost:6333]
"""

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
from database.store import PersonVectorStore                      # noqa: E402
from identity.reconcile import _prototype                         # noqa: E402

if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
    raise SystemExit(__doc__)
RUN = sys.argv[1]
URL = "http://localhost:6333"
for i, a in enumerate(sys.argv):
    if a == "--url" and i + 1 < len(sys.argv):
        URL = sys.argv[i + 1]

store = PersonVectorStore(url=URL, ensure_collection=False, read_only=True)

per = defaultdict(lambda: {"vecs": [], "h": [], "w": [], "blur": [], "ts": [],
                           "hb": []})
offset = None
while True:
    pts, offset = store.client.scroll(store.collection, limit=1000, offset=offset,
                                      with_payload=True, with_vectors=True)
    for p in pts:
        pl = p.payload or {}
        if pl.get("run_id") != RUN:
            continue
        key = (pl.get("camera"), pl.get("track_id"))
        rec = per[key]
        v = p.vector
        if isinstance(v, dict):
            v = next(iter(v.values()), None)
        if v is not None:
            rec["vecs"].append(np.asarray(v, dtype=np.float32))
        bb = pl.get("bbox")
        if bb and len(bb) == 4:
            rec["h"].append(float(bb[3]) - float(bb[1]))
            rec["w"].append(float(bb[2]) - float(bb[0]))
        cq = pl.get("crop_quality")
        if isinstance(cq, dict) and cq.get("blur") is not None:
            rec["blur"].append(float(cq["blur"]))
            # PAIRED (height, blur) per observation. Kept separately from the two
            # independent lists above because section 4 must not assume they are
            # index-aligned -- an observation can carry a bbox and no crop_quality.
            if bb and len(bb) == 4:
                rec["hb"].append((float(bb[3]) - float(bb[1]), float(cq["blur"])))
        if pl.get("ts") is not None:
            rec["ts"].append(float(pl["ts"]))
    if offset is None:
        break

if not per:
    raise SystemExit(f"[crop] run_id {RUN!r} has no observations in this store.")

print("=" * 78)
print(f"1. TRACKLET CROP GEOMETRY -- run {RUN}")
print("=" * 78)
print("  A crop is resized to 384x128 whatever its source size, so 'upscale' is how")
print("  much this tracklet's median crop had to be STRETCHED. reid.quality accepts")
print("  min_height=64 px, i.e. a 6x stretch, on the same footing as a 300 px crop.")
print()
print(f"  {'tracklet':<20}{'obs':>5}{'h med':>8}{'h min':>7}{'h max':>7}"
      f"{'upscale':>9}{'blur med':>10}")
stats = {}
for key in sorted(per, key=lambda k: (str(k[0]), k[1])):
    r = per[key]
    if not r["h"]:
        continue
    h = np.asarray(r["h"])
    blur = np.asarray(r["blur"]) if r["blur"] else np.array([float("nan")])
    stats[key] = {"h_med": float(np.median(h)), "blur_med": float(np.nanmedian(blur)),
                  "n": len(r["vecs"])}
    print(f"  {key[0]}:{key[1]:<12}{len(r['vecs']):>5}{np.median(h):>8.0f}"
          f"{h.min():>7.0f}{h.max():>7.0f}{384.0 / max(np.median(h), 1):>8.1f}x"
          f"{np.nanmedian(blur):>10.1f}")

protos = {k: _prototype(r["vecs"]) for k, r in per.items() if r["vecs"]}
protos = {k: v for k, v in protos.items() if v is not None}

print()
print("=" * 78)
print("2. SAME-CAMERA PAIRS -- score beside the crop-scale mismatch")
print("=" * 78)
print("  'h ratio' is the larger median crop height over the smaller. If the LOW")
print("  scores concentrate at high h ratio, the bar is not the problem: the two")
print("  sides were embedded from very different amounts of real pixel detail.")
print()
print(f"  {'pair':<30}{'cosine':>8}{'h_a':>6}{'h_b':>6}{'h ratio':>9}"
      f"{'blur_a':>8}{'blur_b':>8}{'overlap':>9}")

rows = []
keys = sorted(protos, key=lambda k: (str(k[0]), k[1]))
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        if a[0] != b[0]:
            continue
        if a not in stats or b not in stats:
            continue
        s = float(protos[a] @ protos[b])
        ha, hb = stats[a]["h_med"], stats[b]["h_med"]
        ratio = max(ha, hb) / max(min(ha, hb), 1.0)
        ta, tb = per[a]["ts"], per[b]["ts"]
        if ta and tb:
            ov = min(max(ta), max(tb)) - max(min(ta), min(tb))
        else:
            ov = float("nan")
        rows.append((s, a, b, ha, hb, ratio, stats[a]["blur_med"],
                     stats[b]["blur_med"], ov))

for s, a, b, ha, hb, ratio, ba, bb, ov in sorted(rows):
    pair = f"{a[0]}:{a[1]}+{b[1]}"
    flag = "  CO-PRESENT" if ov == ov and ov > 0 else ""
    print(f"  {pair:<30}{s:>8.3f}{ha:>6.0f}{hb:>6.0f}{ratio:>9.2f}"
          f"{ba:>8.1f}{bb:>8.1f}{ov:>9.1f}{flag}")

print()
print("=" * 78)
print("3. THE TEST")
print("=" * 78)
disjoint = [r for r in rows if not (r[8] == r[8] and r[8] > 0)]
copresent = [r for r in rows if r[8] == r[8] and r[8] > 0]
print(f"  time-DISJOINT same-camera pairs (could be one person): {len(disjoint)}")
print(f"  CO-PRESENT   same-camera pairs (provably two people):  {len(copresent)}")


def _corr_ci(rs):
    """Pearson r with a Fisher-z 95% CI. Returns (r, lo, hi, n) or None.

    The CI is not decoration. On this run the disjoint set gave r=-0.49 at n=259
    and the co-present set r=-0.31 at n=28; the second CI includes ZERO and the two
    overlap heavily, so the pair of numbers cannot support "scale hurts same-person
    pairs more than stranger pairs". That claim was made once and is withdrawn.
    """
    if len(rs) < 5:
        return None
    s = np.array([r[0] for r in rs])
    q = np.array([r[5] for r in rs])
    if s.std() == 0 or q.std() == 0:
        return None
    r = float(np.corrcoef(s, q)[0, 1])
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(len(rs) - 3)
    return r, float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)), len(rs)


def _mean_ci(x):
    x = np.asarray([v for v in x if v == v], dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


for name, rs in (("disjoint", disjoint), ("co-present", copresent)):
    got = _corr_ci(rs)
    if got is None:
        print(f"  {name}: too few pairs to correlate")
        continue
    r, lo, hi, n = got
    s = np.array([r_[0] for r_ in rs])
    q = np.array([r_[5] for r_ in rs])
    print(f"  {name}: n={n}  cosine mean={s.mean():.3f}")
    print(f"      corr(cosine, h_ratio) = {r:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]"
          + ("   <-- INCLUDES ZERO" if lo <= 0 <= hi else ""))
    # RANGE RESTRICTION, reported because it attenuates a correlation on its own.
    # Two people co-present in one frame are often at comparable depths, so this
    # subset's h_ratio span is narrower and its r is biased toward zero for that
    # reason alone -- nothing to do with the mechanism.
    print(f"      h_ratio: min={q.min():.2f} median={np.median(q):.2f} "
          f"max={q.max():.2f}  (span {q.max() - q.min():.2f})")
    m_lo, c_lo = _mean_ci(s[q <= np.median(q)])
    m_hi, c_hi = _mean_ci(s[q > np.median(q)])
    print(f"      MATCHED    scale (h ratio <= median): mean cosine "
          f"{m_lo:.3f} +/- {c_lo:.3f}  (n={int((q <= np.median(q)).sum())})")
    print(f"      MISMATCHED scale (h ratio >  median): mean cosine "
          f"{m_hi:.3f} +/- {c_hi:.3f}  (n={int((q > np.median(q)).sum())})")
    print()

print("  WHAT THIS CAN AND CANNOT SUPPORT.")
print("  The DISJOINT set is UNLABELLED: time-disjoint same-camera pairs mix one")
print("  person returning with two different people appearing at different times, so")
print("  its mean cosine is not a same-person figure and a drop across it is not a")
print("  'cost to same-person pairs'. h_ratio also encodes where in the room someone")
print("  stood, which encodes lighting, pose and viewing angle -- scale is entangled")
print("  with all of it. Correlation here cannot separate cause from confound.")
print("  For that, run the controlled version, which holds person, clothing, camera,")
print("  lighting and room position fixed and varies ONLY resolution:")
print("    python tests/calibration/degrade_crops_causal.py --clips .")

# --------------------------------------------------------------------------
header = "4. BLUR AT MATCHED CROP HEIGHT -- is any camera really softer?"
print()
print("=" * 78)
print(header)
print("=" * 78)
print("  Laplacian variance is NOT scale-invariant, and it is measured on the RAW")
print("  crop before resize (reid/service.py::_crop_quality). The same physical")
print("  detail spread over more pixels gives smaller per-pixel gradients, so a")
print("  camera whose people are simply CLOSER reads as 'softer'; and small, distant,")
print("  noisy crops read as 'sharp' partly on sensor noise. So comparing raw blur")
print("  across cameras with different crop sizes says nothing. Binning by height is")
print("  the minimum fix: only compare cameras inside the same band.")
print()
bands = [(100, 200), (200, 300), (300, 450), (450, 650), (650, 1200)]
by_cam_hb = defaultdict(list)
for (cam, _tid), r in per.items():
    by_cam_hb[cam].extend(r["hb"])
cams = sorted(by_cam_hb, key=str)
print(f"  {'height band':<16}" + "".join(f"{c:>22}" for c in cams))
for lo_h, hi_h in bands:
    cells = []
    for cam in cams:
        vals = [b for (h, b) in by_cam_hb[cam] if lo_h <= h < hi_h]
        cells.append(f"{np.median(vals):.0f} (n={len(vals)})" if vals else "-")
    if all(c == "-" for c in cells):
        continue
    print(f"  {f'{lo_h}-{hi_h} px':<16}" + "".join(f"{c:>22}" for c in cells))
print()
print("  Read DOWN a column to see the scale artifact inside one camera; read ACROSS")
print("  a row for the only honest cross-camera comparison. A camera that is still")
print("  markedly lower in a band it SHARES with another is genuinely softer, and")
print("  that is a lens/focus problem no threshold can address. If the differences")
print("  vanish once height is matched, the 'softer camera' reading was the scale")
print("  artifact again and should be dropped.")
print()
print("  Cheapest check of all, and it needs no metric: open ._live_src_cam_219.mp4")
print("  and look at it. A soft image is obvious in two seconds and no confound can")
print("  reach that observation.")
