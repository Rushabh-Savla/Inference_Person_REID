"""
Embedding-path contract check.  ASSERTS -- exits non-zero if anything regresses.

This is the one script here that is a genuine test rather than a measurement.
Everything it checks was verified clean during the 2026-07-30 audit
(REMEDIATION_PLAN.md Part D), and every item is a silent-failure mode: a broken
BGR->RGB swap, a transposed resize, or a batch/crop misalignment produces no
error and no crash -- only worse matching. Run this after ANY change to
reid/extractor.py or reid/service.py.

    python tests/calibration/verify_embedding_contract.py

Deliberately CPU-safe and needs no footage.
"""

import sys

from _common import bootstrap, reid_weights, reid_model, header

bootstrap()

import numpy as np
import torch

from reid.extractor import ReIDExtractor

# Whichever backbone config.yaml selects (CALIB_REID_MODEL / CALIB_REID_WEIGHTS
# override). The contract asserted below is backend-INDEPENDENT on purpose --
# it is the set of invariants ReIDExtractor owns for every model, so a newly
# added backend is checked by running this script against it, unchanged.
REID_WEIGHTS = reid_weights()
REID_MODEL = reid_model()
# What config.yaml actually ships, ignoring any env override -- used to decide
# whether the store-dimension check applies to this run.
_CONFIGURED_MODEL = reid_model(ignore_env=True)

FAILURES = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    except AssertionError as e:
        print(f"  [FAIL] {name} -- {e}")
        FAILURES.append(name)
    except Exception as e:                                  # noqa: BLE001
        print(f"  [ERROR] {name} -- {type(e).__name__}: {e}")
        FAILURES.append(name)


header(f"1. CHECKPOINT LOAD ({REID_MODEL}) -- is any real layer at random init?")

# This section is the one backbone-SPECIFIC check here: it re-derives the load
# independently of src/reid/backends.py so a bug in that loader cannot hide
# itself. It therefore knows how the torchreid OSNet family is built; for a
# non-OSNet backend it is skipped and the backend's own load-time guard is what
# stands (backends.py raises on any unexpected dropped key).
from reid.backends import OSNetBackend

if REID_MODEL in OSNetBackend.ARCHS:
    import torchreid

    raw = torch.load(REID_WEIGHTS, map_location="cpu", weights_only=False)
    state = raw["state_dict"] if "state_dict" in raw else raw
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    ref = torchreid.models.build_model(name=REID_MODEL, num_classes=1000,
                                       loss="softmax", pretrained=False)
    ref_sd = ref.state_dict()
    loaded = {k: v for k, v in state.items()
              if k in ref_sd and ref_sd[k].shape == v.shape}
    unloaded = [k for k in ref_sd if k not in loaded
                if "num_batches_tracked" not in k]
    dropped = [k for k in state if k not in loaded]

    print(f"  checkpoint keys {len(state)} | model keys {len(ref_sd)} | loaded {len(loaded)}")
    check("only the discarded classifier head is unloaded",
          lambda: (_ for _ in ()).throw(AssertionError(f"unloaded={unloaded}"))
          if unloaded != ["classifier.weight", "classifier.bias"]
          else f"unloaded={unloaded}")
    check("only the classifier head is dropped from the checkpoint",
          lambda: (_ for _ in ()).throw(AssertionError(f"dropped={dropped}"))
          if dropped != ["classifier.weight", "classifier.bias"] else None)
else:
    print(f"  [SKIP] independent key check knows only the OSNet family; "
          f"'{REID_MODEL}' relies on its own backend load guard.")

ex = ReIDExtractor(weights=REID_WEIGHTS, model=REID_MODEL, device="cpu")

# The authoritative width/size for THIS run: measured from the loaded model, not
# imported as a constant. A backend whose real output width disagrees with what
# the Qdrant collection is sized to is exactly the failure this replaces.
EMBEDDING_DIM = ex.embedding_dim
INPUT_SIZE = ex.input_size
print(f"  loaded: {ex.describe()}")


header("2. OUTPUT CONTRACT")

def _eval_mode():
    assert ex.model.training is False, "model is in TRAIN mode -- BatchNorm would " \
                                       "use batch stats and forward() returns logits"
    return "model.training is False"
check("model is in eval mode", _eval_mode)


def _shape_and_norm():
    crops = [np.random.randint(0, 255, (300, 140, 3), np.uint8) for _ in range(4)]
    d = ex.extract_batch(crops)
    assert d.shape == (4, EMBEDDING_DIM), f"expected (4,{EMBEDDING_DIM}), got {d.shape}"
    assert d.dtype == np.float32, f"expected float32, got {d.dtype}"
    norms = np.linalg.norm(d, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"not unit-normalised: {norms}"
    return f"shape={d.shape} norms~1.0"
check(f"{EMBEDDING_DIM}-d float32 output, L2-normalised", _shape_and_norm)


def _empty_batch():
    out = ex.extract_batch([])
    assert out.shape == (0, EMBEDDING_DIM), f"got {out.shape}"
    return f"returns (0, {EMBEDDING_DIM})"
check("empty batch is well-shaped", _empty_batch)


def _input_size():
    h, w = INPUT_SIZE
    assert h > w, (f"input_size {INPUT_SIZE} is not (H, W) with H > W -- person "
                   f"crops are tall and narrow, so a square/landscape shape here "
                   f"usually means the tuple got transposed")
    # Prove the resize is actually applied: a crop of a DIFFERENT shape must still
    # preprocess to exactly the model's input size. A dropped or transposed resize
    # produces no error, only worse matching.
    got = tuple(ex._preprocess(np.zeros((h + 61, w + 17, 3), np.uint8)).shape)
    assert got == (3, h, w), f"preprocess gave {got}, expected (3, {h}, {w})"
    return f"input_size={INPUT_SIZE} (H,W); preprocess -> {got}"
check("input size pinned and the resize is applied", _input_size)


def _store_dim_agrees():
    """Cross-layer guard: database/store.py keeps its own EMBEDDING_DIM so the
    storage layer does not import the model layer. A disagreement is legitimate
    ONLY while a backbone swap is mid-flight -- and it means the Qdrant
    collection must be rebuilt, so it should never pass unnoticed.

    Only meaningful for the SHIPPING model. A comparison run measuring some other
    backbone (CALIB_REID_MODEL=...) is expected to differ, and failing there would
    make an A/B look like a regression."""
    from database.store import EMBEDDING_DIM as STORE_DIM
    if REID_MODEL != _CONFIGURED_MODEL:
        return (f"skipped -- measuring {REID_MODEL}, not the configured "
                f"{_CONFIGURED_MODEL}; store is sized for the latter")
    assert STORE_DIM == EMBEDDING_DIM, (
        f"store.py defaults to {STORE_DIM}-d but {REID_MODEL} emits "
        f"{EMBEDDING_DIM}-d. The pipeline passes the measured width explicitly so "
        f"a fresh collection is correct, but any EXISTING collection is stale: "
        f"rebuild it, and update store.py's constant.")
    return f"both {EMBEDDING_DIM}"
check("database/store.py's dimension constant matches this model", _store_dim_agrees)


header("3. DETERMINISM, BATCH ORDER, BATCH INVARIANCE")

crops = [np.random.randint(0, 255, (200 + 40 * i, 90 + 15 * i, 3), np.uint8)
         for i in range(5)]


def _determinism():
    a, b = ex.extract_batch(crops), ex.extract_batch(crops)
    diff = np.abs(a - b).max()
    assert diff < 1e-6, f"non-deterministic, max diff {diff:.2e}"
    return f"max diff {diff:.2e}"
check("same batch twice gives identical vectors", _determinism)


def _batch_order():
    batched = ex.extract_batch(crops)
    singles = np.stack([ex.extract(c) for c in crops])
    order = (singles @ batched.T).argmax(axis=1)
    assert (order == np.arange(len(crops))).all(), \
        f"batch output MISALIGNED with input order: {order}"
    return f"argmax per row = {order.tolist()}"
check("batch output order matches input crop order", _batch_order)


def _batch_invariance():
    batched = ex.extract_batch(crops)
    singles = np.stack([ex.extract(c) for c in crops])
    diff = np.abs(batched - singles).max()
    assert diff < 1e-4, f"batching perturbs values by {diff:.2e} (BatchNorm not frozen?)"
    return f"max diff {diff:.2e}"
check("batching does not perturb values", _batch_invariance)


header("4. PREPROCESSING IS LOAD-BEARING")

def _bgr_rgb():
    c = np.zeros((256, 128, 3), np.uint8)
    c[:, :, 2] = 200          # strong red in BGR
    c[:, :, 1] = 60
    same = float(ex.extract(c) @ ex.extract(c[:, :, ::-1].copy()))
    assert same < 0.99, ("channel order makes no difference -- the BGR->RGB "
                         "cvtColor may have been removed")
    return f"cosine(correct, channel-swapped) = {same:.4f}"
check("BGR->RGB conversion changes the embedding", _bgr_rgb)


def _empty_crop_raises():
    try:
        ex.extract(np.zeros((0, 10, 3), np.uint8))
    except ValueError:
        return "raises ValueError"
    raise AssertionError("an empty crop was silently embedded -- that injects a "
                         "meaningless vector into matching")
check("empty crop raises instead of embedding garbage", _empty_crop_raises)


header("5. DEGENERATE INPUTS PRODUCE FINITE VECTORS")

for label, crop in (("1x1 px", np.full((1, 1, 3), 128, np.uint8)),
                    ("all black", np.zeros((200, 90, 3), np.uint8)),
                    ("all white", np.full((200, 90, 3), 255, np.uint8)),
                    ("2px wide", np.random.randint(0, 255, (200, 2, 3), np.uint8))):
    def _finite(c=crop, l=label):
        v = ex.extract(c)
        assert np.isfinite(v).all(), "non-finite values"
        assert abs(np.linalg.norm(v) - 1.0) < 1e-4, "not unit norm"
        return f"norm=1.0 finite, {int((v != 0).sum())}/{EMBEDDING_DIM} nonzero"
    check(f"{label} embeds without NaN", _finite)


header("6. TrackEmbedder CACHE -- no cross-track contamination")

def _cache_aliasing():
    from reid.service import TrackEmbedder
    from detector import Detection
    te = TrackEmbedder(ex, interval=10, ttl=300, quality=None,
                       max_embeddings_per_track=0, warmup_embeddings=3)
    frame = np.random.randint(0, 255, (600, 400, 3), np.uint8)
    dets = [Detection(x1=10, y1=10, x2=110, y2=310, confidence=.9,
                      class_id=0, track_id=1),
            Detection(x1=200, y1=20, x2=310, y2=330, confidence=.9,
                      class_id=0, track_id=2)]
    te.process(frame, dets, 0)
    e0, e1 = dets[0].embedding, dets[1].embedding
    assert e0 is not None and e1 is not None, "no embeddings attached"
    assert not np.allclose(e0, e1), "TWO DETECTIONS GOT THE SAME VECTOR"
    cached = te._cache[1]["embedding"]
    return (f"cosine(det0,det1)={float(e0 @ e1):.3f}; cache shares memory with "
            f"det0: {np.shares_memory(cached, e0)} (retention only, nothing mutates)")
check("distinct detections get distinct vectors", _cache_aliasing)


header("RESULT")
if FAILURES:
    print(f"  FAIL -- {len(FAILURES)} check(s) regressed: {FAILURES}")
    sys.exit(1)
print("  PASS -- embedding path contract intact.")
