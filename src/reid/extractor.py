"""
extractor.py  --  STAGE 5:  ReID feature extraction.

============================ THE BIG PICTURE ================================
This module has EXACTLY ONE job:

    OpenCV BGR person crop  ->  L2-normalized embedding.

That's it. It does NOT assign identities, search a database, or know anything
about cameras, tracks, timestamps, YOLO, ByteTrack, or Qdrant. It is a pure
function wrapped in a class (the class only exists so we load the heavy model
weights ONCE and reuse them for every crop).

WHY the strict boundary: an embedding is a reusable measurement of "what this
person looks like." Deciding WHO they are (matching, gallery, global ids) is a
separate, stateful concern with its own failure modes (camera topology, time
windows, motion). Mixing the two is how ReID systems rot. Keep this dumb.

--------------------- WHAT LIVES HERE vs IN A BACKEND -----------------------
WHICH network runs is a `backends.py` concern (see that file's header). This
file owns only the things that must be true no matter which network it is:

  * batching, and chunking it at max_batch so a crowded frame cannot OOM;
  * the forward lock that serialises the ONE shared model across camera threads;
  * L2-normalization -- applied here, once, so no backend can forget it;
  * the numpy hand-off and the empty-crop rejection.

Those are exactly the invariants tests/calibration/verify_embedding_contract.py
asserts. Keeping them out of the per-model code means a backbone experiment
cannot silently break them, which matters because every one of these failure
modes is invisible: no error, no crash, just worse matching.

The default backend is OSNet-AIN x1_0 (`osnet_ain_x1_0`), selectable via
`reid.model` in config.yaml.

--------------------------- EMBEDDING DIMENSION ----------------------------
EMBEDDING_DIM below is the default backend's width (512) and is what the Qdrant
collection is sized to. It is a property of the CHECKPOINT, not a free choice.
Prefer the per-instance `extractor.embedding_dim`, which is MEASURED from the
loaded model -- the module constant is the historical default kept for the
calibration harness and for `database/store.py`'s guard.

If a backend of a different width is adopted, that number propagates outward:
the Qdrant collection is rebuilt, and every threshold in config.yaml is void
because it is a different feature space (same rule as the tap flag, #39).

------------------------------ WHY L2-NORMALIZE ----------------------------
We divide each embedding by its L2 norm so every vector lands on the unit
hypersphere. Then cosine similarity between two people is just a dot product,
and Euclidean distance becomes a monotonic function of cosine -- so downstream
matching (and Qdrant's COSINE metric) is well-defined and scale-invariant. The
raw magnitude of a ReID feature carries no identity information; only its
DIRECTION does.
============================================================================
"""

import threading
from typing import List, Optional

import numpy as np
import torch

from reid.backends import (DEFAULT_BACKEND, IMAGENET_MEAN, IMAGENET_STD,
                           OSNET_INPUT_SIZE, build_backend)


# --- Constants (see module docstring for the "why" of each) ----------------

# The DEFAULT BACKEND's input as (Height, Width) and output width -- i.e.
# `backends.DEFAULT_BACKEND` (osnet_ain_x1_0), NOT the model that ships. Re-exported
# from backends.py so the long-standing `from reid.extractor import EMBEDDING_DIM,
# INPUT_SIZE` imports keep resolving.
#
# READ THIS BEFORE USING EITHER. They describe the fallback, not the run. What
# actually ships is `reid.model: fastreid_sbs_R101_ibn` -- 2048-d at 384x128 -- so
# these two constants are 512 and (256, 128) and are WRONG for it by design. The
# authoritative values are the per-instance `.embedding_dim` / `.input_size`,
# measured off the loaded net. `database.store.EMBEDDING_DIM` is separately 2048
# because a fresh collection must match the SHIPPING model; the two are not
# supposed to be equal.
INPUT_SIZE = OSNET_INPUT_SIZE
EMBEDDING_DIM = 512

__all__ = ["ReIDExtractor", "EMBEDDING_DIM", "INPUT_SIZE",
           "IMAGENET_MEAN", "IMAGENET_STD"]


class ReIDExtractor:
    """
    Loads one ReID backend and turns person crops into embeddings.

    Typical use:
        extractor = ReIDExtractor(weights="weights/osnet_ain_x1_0.pth")
        emb = extractor.extract(person_crop)     # (512,) float32, ||emb|| == 1
    """

    def __init__(self, weights: str, device: Optional[str] = None,
                 max_batch: int = 32, tap: Optional[str] = None,
                 model: Optional[str] = None):
        """
        weights : path to a person-ReID checkpoint (.pth). NOT ImageNet weights
                  -- those do not separate identities.
        device  : "cuda", "cpu", or None to auto-pick. Kept explicit so the
                  same code runs on the CPU dev box and a GPU deployment box
                  without edits.
        max_batch : #56. Largest number of crops in ONE forward pass; more are
                  processed in chunks. The batch used to be sized by however many
                  people happened to be in view, which is an OOM on a smaller GPU
                  exactly when the scene is busiest. BatchNorm is frozen in eval
                  mode (running stats, not per-batch), so chunking cannot change a
                  single embedding -- verified by the batch-invariance check in
                  tests/calibration/verify_embedding_contract.py. 0 = unbounded.
        tap     : #39. Where in the network the feature is read from. Meaning is
                  backend-specific and it is passed through UNCHANGED -- including
                  None, so each backend applies its own default rather than
                  inheriting OSNet's. That matters because the tap is not a
                  universal concept: FastReID's eval feature is always post-bnneck
                  and its backend REJECTS a tap rather than ignore one.
        model   : `reid.model` -- which backend to load. None = the shipped
                  default (osnet_ain_x1_0), which is what every threshold in
                  config.yaml was derived against.
        """
        self.max_batch = max(0, int(max_batch))
        # #50: the ONE thing that genuinely needs serialising across cameras is the
        # shared model's forward pass. The lock used to live in InferenceStage and
        # wrapped the whole of TrackEmbedder.process -- cropping, a float64
        # Laplacian blur check, an O(N^2) occlusion loop and all preprocessing --
        # so four cameras serialised on work that has nothing to do with the shared
        # model, contradicting that lock's own docstring. Owning it here keeps the
        # critical section to exactly the shared resource.
        self._fwd_lock = threading.Lock()
        self.tap = tap
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.backend = build_backend(model or DEFAULT_BACKEND, weights=weights,
                                     device=self.device, tap=self.tap)

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, *,
                    device: Optional[str] = None,
                    config_path: str = "config.yaml") -> "ReIDExtractor":
        """Build the extractor the PIPELINE would build, from config.yaml.

        Exists so demo and diagnostic scripts cannot drift from the shipping
        model. Before this, five scripts hardcoded a
        `weights/osnet_x1_0_market1501.pth` that is not in this tree at all, so
        they had been broken since long before the FastReID switch -- and a
        backbone change would silently have left any that DID work measuring the
        old model.

        cfg    : an already-loaded config dict (the whole file, or just its
                 `reid:` block). None reads `config_path`.
        device : overrides `reid.device`. Pass "cpu" to force the dev box.
        """
        if cfg is None:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        reid = cfg.get("reid", cfg) or {}
        if "weights" not in reid:
            raise KeyError(
                f"no `reid.weights` in the config given to from_config() "
                f"(keys seen: {sorted(reid)[:8]}). Pass the whole config.yaml "
                f"dict or its reid: block.")
        dev = device if device is not None else reid.get("device")
        # "auto" is this project's config vocabulary, not torch's -- torch.device
        # ("auto") raises. None means "let __init__ pick", which is the same thing.
        if isinstance(dev, str) and dev.strip().lower() in ("", "auto"):
            dev = None
        return cls(weights=reid["weights"], device=dev,
                   max_batch=int(reid.get("max_batch", 32)),
                   tap=reid.get("tap"),
                   model=reid.get("model"))

    # --- what the backend decided, surfaced for the rest of the pipeline ----

    @property
    def model(self):
        """The underlying torch module. Exposed because the calibration harness
        asserts eval mode on it and measure_score_separation.py reaches into
        OSNet's own layers to compare taps."""
        return self.backend.model

    @property
    def embedding_dim(self) -> int:
        """MEASURED output width of the loaded model. Pass this to
        PersonVectorStore(dim=...) so the collection can never be sized for a
        different backbone than the one actually running."""
        return self.backend.embedding_dim

    @property
    def input_size(self):
        """(H, W) this backend's checkpoint was trained at."""
        return self.backend.input_size

    def describe(self) -> str:
        """One line for a run banner."""
        return f"{self.backend.describe()} on {self.device}"

    # --- the contract -------------------------------------------------------

    @torch.no_grad()
    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        One BGR crop -> (D,) L2-normalized float32 embedding.

        Convenience wrapper over extract_batch for the common single-crop call.
        """
        return self.extract_batch([crop])[0]

    @torch.no_grad()
    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        A list of BGR crops -> (N, D) L2-normalized float32 embeddings.

        Batching is a REAL production need, not gratuitous abstraction: a single
        frame yields many person boxes, and running them through the network in
        one forward pass (instead of one call each) is a large GPU throughput
        win with hundreds of cameras. On CPU it still avoids Python-loop
        overhead. So the batch path is primary; extract() just calls it with N=1.

        @torch.no_grad() disables autograd -- we never train here, so we skip
        building the graph: less memory, faster.
        """
        if len(crops) == 0:
            # Explicit empty case: return a well-shaped (0, D) array so callers
            # can concatenate results without special-casing.
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # #56: the batch was unbounded in the number of PEOPLE. A crowded frame
        # (or a burst across four cameras) built one tensor sized by whatever
        # happened to be in view, which is an OOM on a smaller GPU at exactly the
        # moment the system is most useful. Chunking keeps peak memory flat and
        # costs nothing when the crowd is small -- one chunk is the old path.
        if self.max_batch and len(crops) > self.max_batch:
            outs = []
            for i in range(0, len(crops), self.max_batch):
                outs.append(self.extract_batch(crops[i:i + self.max_batch]))
            return np.concatenate(outs, axis=0)

        batch = torch.stack([self._preprocess(c) for c in crops])
        batch = batch.to(self.device)

        with self._fwd_lock:                              # #50: shared model only
            features = self.backend.forward(batch)        # (N, D), raw features

        # L2-normalize along the feature dim. eps guards the degenerate case of
        # an all-zero feature (never seen in practice) from producing NaNs.
        features = torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-12)

        # Back to CPU/NumPy for the rest of the (non-torch) pipeline.
        return features.cpu().numpy().astype(np.float32)

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """
        One BGR uint8 crop -> normalized (3, H, W) float32 tensor on CPU.

        The RECIPE belongs to the backend (it is checkpoint-specific and not
        shareable -- see backends.py). What stays here is the one policy that is
        the same for every model: a crop with no pixels is an error, not an
        input.
        """
        if crop is None or crop.size == 0:
            # A zero-area box (can happen at frame edges) has no pixels to
            # describe. Fail loud -- silently embedding a blank crop would inject
            # a meaningless vector into matching.
            raise ValueError("Empty crop passed to ReIDExtractor.")
        return self.backend.preprocess(crop)
