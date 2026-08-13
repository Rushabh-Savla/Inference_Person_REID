"""
Same-person vs different-person score separation, on real footage.

Produces REMEDIATION_PLAN.md Part H.1 / H.2 / H.3 / H.5. This is the measurement
that sets every identity threshold in the system, so re-run it whenever the
footage, the ReID weights, or the feature tap changes.

    python tests/calibration/measure_score_separation.py [video] [frames] [stride]

Four things are compared:
  * feature tap      -- post-ReLU (what ships) vs post-BN (see Part G, D9/§7)
  * raw crop-to-crop -- the underlying feature quality
  * bank scoring     -- max(prototype, best_exemplar), what actually decides
  * alternatives     -- consensus (mean of top half) and prototype-only

WHY THE TAP MATTERS: torchreid's OSNet `fc` is Linear -> BatchNorm1d -> ReLU and
eval-mode forward() returns the post-ReLU activation. That confines the embedding
to the non-negative orthant, so cosine between ANY two vectors is >= 0 and the
usable range is compressed upward. Changing the tap INVALIDATES every threshold.
"""

import sys

import numpy as np
import torch

from _common import (bootstrap, pick_video, sample_frames, reid_weights,
                     reid_model,
                     DETECT_WEIGHTS, collect_track_embeddings,
                     proven_distinct_pairs, unit, describe, header, margin,
                     print_operating_points, footnote_sample_size)

bootstrap()

from reid.extractor import ReIDExtractor
from detector import PersonDetector
from live.identity_engine import ActiveIdentitySet

VIDEO = pick_video(sys.argv[1] if len(sys.argv) > 1 else None)
NFRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 6
BANK = 20                     # mirrors live.identity.bank_size / identity.bank_size

# The BACKBONE comes from config (CALIB_REID_MODEL / CALIB_REID_WEIGHTS override),
# so an A/B between checkpoints is two runs of this script and no code edit.
# The TAP deliberately does NOT: this script's whole subject is comparing the two
# taps, and the equivalence assertion below requires `ex` to be the shipped
# post-ReLU path. It also reaches into OSNet's own fc block, so it only applies
# to the torchreid OSNet family.
ex = ReIDExtractor(weights=reid_weights(), model=reid_model(), device="cpu")
print(f"[calib] ReID: {ex.describe()}")
model = ex.model

# THE TAP AXIS IS OSNET-ONLY, AND THIS SCRIPT USED TO ASSUME IT UNCONDITIONALLY.
# Comparing post-ReLU against post-BN means reaching into torchreid OSNet's own `fc`
# block (Linear -> BatchNorm1d -> ReLU). FastReID has no such block and no选able
# tap -- its EmbeddingHead returns the post-bnneck feature unconditionally -- so the
# surgery below raised
#     AttributeError: 'Sequential' object has no attribute 'global_avgpool'
# and the script died on line 71, before measuring anything.
#
# That mattered more than a broken script: this is the measurement that sets every
# identity threshold in the system, so from the FastReID switch until 2026-08-03
# there was NO WAY to derive a threshold for the backbone actually running. Nothing
# had been recalibrated, and this is why.
#
# So the tap axis is now discovered, not assumed. On OSNet both taps are compared as
# before; on any other backend there is exactly one feature space -- production --
# and everything else in this script is backend-agnostic and runs unchanged.
OSNET_FAMILY = all(hasattr(model, a) for a in ("global_avgpool", "featuremaps", "fc"))
TAPS = ("relu", "bn") if OSNET_FAMILY else ("production",)
PRODUCTION_TAP = "relu" if OSNET_FAMILY else "production"


@torch.no_grad()
def embed(crops, tap):
    """tap='production' is exactly what ships, for any backend.

    On the OSNet family only: 'relu' reproduces ReIDExtractor.extract_batch exactly
    and 'bn' skips the final ReLU, which is what makes the tap comparison possible.
    """
    if tap == "production":
        return ex.extract_batch(crops)
    batch = torch.stack([ex._preprocess(c) for c in crops]).to(ex.device)
    v = model.global_avgpool(model.featuremaps(batch)).view(batch.size(0), -1)
    v = model.fc[0](v)                       # Linear
    v = model.fc[1](v)                       # BatchNorm1d
    if tap == "relu":
        v = model.fc[2](v)                   # ReLU  <-- ships on OSNet
    v = torch.nn.functional.normalize(v, p=2, dim=1, eps=1e-12)
    return v.cpu().numpy().astype(np.float32)


# The production path must be bit-equal to what ships, or nothing below is comparable.
probe = [np.random.randint(0, 255, (200, 90, 3), np.uint8) for _ in range(3)]
delta = np.abs(embed(probe, PRODUCTION_TAP) - ex.extract_batch(probe)).max()
assert delta < 1e-5, (f"tap {PRODUCTION_TAP!r} does not reproduce extract_batch "
                      f"(diff {delta:.2e})")
print(f"[calib] tap {PRODUCTION_TAP!r} reproduces production extract_batch "
      f"(diff {delta:.1e})")
if not OSNET_FAMILY:
    print(f"[calib] tap comparison SKIPPED: {ex.describe()} has no selectable tap, "
          f"so there is one feature space and it is the one that ships.")

header("0. THE FEATURE SPACE THIS BACKBONE ACTUALLY PRODUCES   (Part H.5)")
for tap in TAPS:
    f = embed(probe, tap)
    print(f"  {tap:<11s}: min={f.min():+.4f} max={f.max():+.4f} "
          f"exact-zero dims={100 * (f == 0).mean():5.1f}%  "
          f"negative dims={100 * (f < 0).mean():5.1f}%")
if OSNET_FAMILY:
    print("  post-ReLU is confined to the non-negative orthant -> cosine of any two")
    print("  vectors is >= 0, compressing the usable range.")
else:
    print("  A space with NEGATIVE dimensions uses the full hypersphere, so cosine")
    print("  can go below 0 and the usable range is wider than post-ReLU OSNet's.")
    print("  EVERY THRESHOLD DERIVED UNDER OSNET IS THEREFORE MEANINGLESS HERE --")
    print("  that is what this script exists to re-derive.")

# ---------------------------------------------------------------- gather footage
frames = sample_frames(VIDEO, NFRAMES, STRIDE)
print(f"\n[calib] {VIDEO}: {len(frames)} frames @ stride {STRIDE}, "
      f"{frames[0].shape[1]}x{frames[0].shape[0]}")

det = PersonDetector(model_path=DETECT_WEIGHTS, confidence_threshold=0.4,
                     person_class_id=0, tracker_config="bytetrack.yaml",
                     pose_ensemble=None, iou=0.60)

data = {}
for tap in TAPS:
    det_tap = PersonDetector(model_path=DETECT_WEIGHTS, confidence_threshold=0.4,
                             person_class_id=0, tracker_config="bytetrack.yaml",
                             pose_ensemble=None, iou=0.60)
    by_track, cooccur, per_frame = collect_track_embeddings(
        frames, det_tap, lambda c, t=tap: embed(c, t))
    data[tap] = proven_distinct_pairs(by_track, cooccur, min_obs=6) + (per_frame,)

usable_r, pairs, excluded, per_frame_r = data[PRODUCTION_TAP]
if not pairs:
    raise SystemExit("[calib] no proven-distinct (co-visible) track pairs in this "
                     "clip -- cannot measure. Try more frames or busier footage.")


# ---------------------------------------------------------------- scoring modes
def raw_scores(usable, pairs):
    same, other = [], []
    for _, vs in usable.items():
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                same.append(float(vs[i] @ vs[j]))
    for a, b in pairs:
        for ea in usable[a]:
            for eb in usable[b]:
                other.append(float(ea @ eb))
    return np.array(same), np.array(other)


def _holdout(vs):
    """Split one track's observations into (bank, queries), DISJOINT.

    CRITICAL, and got wrong on the first attempt: if a query observation is also
    in the bank, the max-exemplar term in ActiveIdentitySet.score matches it
    against ITSELF and returns 1.000. That inflates the same-person distribution
    by however large a fraction of queries sit in the bank, which makes the
    measured margin depend on the frame count rather than on the model. The live
    engine documents this exact trap in _reinforce ("scoring this crop AFTER
    add_observation ... would return ~1.0 every frame").

    Bank comes from the EARLIER half, queries from the later half, which also
    matches production: the bank always precedes the query in time.
    """
    half = max(1, len(vs) // 2)
    return vs[:half][-BANK:], vs[half:]


def bank_scores(usable, pairs):
    """max(prototype, best_exemplar) -- ActiveIdentitySet.score, which is what
    same-camera reacquisition and cross-camera linking actually use."""
    same, other = [], []
    for a, b in pairs:
        for owner, guest in ((a, b), (b, a)):
            bank, queries = _holdout(usable[owner])
            st = ActiveIdentitySet(bank_size=BANK)
            for e in bank:
                st.add_observation(owner, "cam", unit(e), 0.0)
            for e in queries:
                same.append(st.score(owner, unit(e)))
            for e in usable[guest]:
                other.append(st.score(owner, unit(e)))
    return np.array(same), np.array(other)


def consensus_scores(usable, pairs, keep=0.5):
    """Mean of the top `keep` fraction of exemplars -- requires agreement, so a
    single poisoned exemplar cannot carry the match."""
    def score(bank, q):
        s = np.stack(bank) @ q
        k = max(1, int(len(s) * keep))
        return float(np.mean(np.sort(s)[-k:]))

    same, other = [], []
    for a, b in pairs:
        for owner, guest in ((a, b), (b, a)):
            bank_obs, queries = _holdout(usable[owner])
            bank = [unit(e) for e in bank_obs]
            for e in queries:
                same.append(score(bank, unit(e)))
            for e in usable[guest]:
                other.append(score(bank, unit(e)))
    return np.array(same), np.array(other)


def prototype_scores(usable, pairs):
    same, other = [], []
    for a, b in pairs:
        for owner, guest in ((a, b), (b, a)):
            bank, queries = _holdout(usable[owner])
            st = ActiveIdentitySet(bank_size=BANK)
            for e in bank:
                st.add_observation(owner, "cam", unit(e), 0.0)
            proto = st.prototype(owner)
            for e in queries:
                same.append(float(proto @ unit(e)))
            for e in usable[guest]:
                other.append(float(proto @ unit(e)))
    return np.array(same), np.array(other)


MODES = (("RAW crop-to-crop            (Part H.1)", raw_scores),
         ("BANK max(proto,exemplar)    (Part H.2)", bank_scores),
         ("CONSENSUS mean of top half  (Part H.3)", consensus_scores),
         ("PROTOTYPE only              (Part H.3)", prototype_scores))

summary = {}
for title, fn in MODES:
    header(title)
    print(f"  {'tap':<11}{'same p5':>9}{'other p95':>11}{'other MAX':>11}{'margin':>9}")
    for tap in TAPS:
        usable, prs, _, _ = data[tap]
        prs = [(a, b) for (a, b) in prs if a in usable and b in usable]
        s, o = fn(usable, prs)
        summary[(title, tap)] = (s, o)
        print(f"  {tap:<11}{np.percentile(s, 5):>9.3f}{np.percentile(o, 95):>11.3f}"
              f"{o.max():>11.3f}{margin(s, o):>+9.3f}")
    print()
    for tap in TAPS:
        s, o = summary[(title, tap)]
        print(f"  --- tap={tap} ---")
        print(describe(s, "same person"))
        print(describe(o, "DIFFERENT person"))
        print_operating_points(s, o)

header("SEPARABILITY SUMMARY -- higher margin is easier to threshold")
print(f"  {'mode':<40}{'tap':<11}{'margin':>9}{'other MAX':>11}")
for title, _ in MODES:
    for tap in TAPS:
        s, o = summary[(title, tap)]
        print(f"  {title:<40}{tap:<11}{margin(s, o):>+9.3f}{o.max():>11.3f}")

# Read the bars from config rather than restating literals. They were hardcoded
# here as 0.70 / 0.60 / 0.90, which is a second copy that silently goes stale the
# moment anyone tunes -- and the whole point of this section is to compare the bars
# IN FORCE against the distributions just measured.
print("\n  Configured thresholds IN FORCE, for comparison with the numbers above:")
try:
    from _common import load_config
    _c = load_config() or {}
    _live = ((_c.get("live") or {}).get("identity") or {})
    _rec = ((_c.get("identity") or {}).get("reconcile") or {})
    print(f"    live.identity.same_camera_threshold      = "
          f"{_live.get('same_camera_threshold')}")
    print(f"    live.identity.cross_camera_threshold     = "
          f"{_live.get('cross_camera_threshold')}")
    print(f"    identity.reconcile.same_camera_threshold = "
          f"{_rec.get('same_camera_threshold')}   per_camera="
          f"{_rec.get('per_camera') or '{}'}")
    print(f"    identity.reconcile.threshold (cross)     = "
          f"{_rec.get('threshold', (_c.get('identity') or {}).get('threshold'))}")
    print(f"    identity.reconcile.scoring               = {_rec.get('scoring')}")
except Exception as _e:                                          # noqa: BLE001
    print(f"    (could not read config.yaml: {_e})")

header("HOW MUCH OF THIS TO TRUST")
print("""  MEASURED AND STABLE across sample sizes -- safe to act on:
    * (OSNet only) post-BN beats post-ReLU on margin and lowers the
      different-person ceiling. Not applicable to a backend with no tap choice.
    * consensus scoring lowers the different-person ceiling vs max(proto,exemplar)
    * the different-person ceiling sits WELL ABOVE live.identity.same_camera_threshold
      (0.70), so that threshold is inside the range where strangers score

  NOT STABLE -- do not set a threshold from one run:
    * 'other MAX' is an extreme-value statistic and GROWS with sample size.
      Prefer p95. Two runs on the same clip at 48 vs 90 frames moved the raw MAX
      from 0.819 to 0.936.
    * the margin itself moved +0.055 -> +0.108 (bank, post-ReLU) between those
      same two runs.

  These are also SAME-CAMERA numbers only. Cross-camera same-person scores run
  lower, so cross_camera_threshold CANNOT be calibrated from this clip -- that
  needs a multi-camera frozen clip (REMEDIATION_PLAN.md Part E).

  Conclusion: use this script to compare OPTIONS (tap, scoring mode) on identical
  footage, not to pick a number. Numbers come from the Phase 9 sweep on frozen
  multi-camera footage.""")

footnote_sample_size(usable_r, pairs, excluded)
