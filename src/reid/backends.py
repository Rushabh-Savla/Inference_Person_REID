"""
backends.py  --  the SWAPPABLE half of STAGE 5 (ReID feature extraction).

============================ THE BIG PICTURE ================================
`extractor.py` owns a contract: BGR person crop -> L2-normalized embedding, in
batches, deterministically, batch-invariantly, on a shared model. This file owns
the OTHER question -- WHICH network computes it.

The two are separated because they change for completely different reasons. The
contract is a property of the pipeline (Qdrant wants unit vectors; four cameras
share one model; a crowded frame must not OOM). The network is a property of an
EXPERIMENT: "is FastReID ResNet101-IBN better than OSNet-AIN on our footage?" --
which is the experiment this file was written for, and the swap that shipped.
Every such experiment used to mean editing the one file that also holds the
invariants the calibration harness asserts -- so each backbone trial risked
silently breaking batching or normalisation, which produce no error, just worse
matching.

WHAT IS RUNNING TODAY: `reid.model: fastreid_sbs_R101_ibn` -- FastReID SBS
ResNet-**101**-IBN, MSMT17, 2048-d at 384x128. R50 is registered and unused.
Nothing below changes that; see the note on OSNetBackend.__init__'s `arch`
default, which is the one line people misread as "we are still on OSNet".

------------------------------ THE SPLIT -----------------------------------
A backend OWNS everything model-specific:

  * building the architecture and loading the checkpoint (including which keys
    it is legitimate to drop);
  * the PREPROCESSING RECIPE -- input size, channel order, scale, per-channel
    statistics. This is NOT shareable across backends and must not be hoisted:
    torchreid normalises on the 0..1 scale with ImageNet mean/std, while
    FastReID normalises on the 0..255 scale. Feeding one recipe to the other
    model runs fine and returns garbage-quality embeddings.
  * WHERE in the network the feature is read from (the tap);
  * the embedding width it produces.

`ReIDExtractor` keeps everything backend-INVARIANT: batching, max_batch
chunking, the forward lock that serialises the shared model across camera
threads, L2-normalisation, the numpy hand-off, and rejecting empty crops. Those
are exactly the properties tests/calibration/verify_embedding_contract.py
asserts, and they must hold for EVERY backend rather than being re-implemented
(and re-broken) once per model.

--------------------------- ADDING A BACKEND -------------------------------
Subclass ReIDBackend, implement the four members, register it in BACKENDS. Then
it is reachable from config.yaml as `reid.model: <key>` with no other code
change -- which is the point: a backbone trial should be a config edit and a
harness re-run, not a refactor.

WHAT A NEW BACKEND STILL COSTS, unavoidably:
  * a different embedding width means the Qdrant collection is REBUILT (the
    store's dimension guard fails loud rather than mixing widths);
  * a different feature space VOIDS EVERY THRESHOLD IN config.yaml. Same rule
    as the tap flag (#39): it is not a better version of the same numbers, it is
    different numbers. Re-derive them and never compare score logs across
    backends.
============================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

import cv2
import numpy as np
import torch


class ReIDBackend(ABC):
    """One person-ReID network, loaded and ready for inference.

    Implementations must leave the model in eval() mode on the requested device
    and expose it as `.model` (the calibration harness asserts eval mode there --
    in train mode BatchNorm would use per-batch statistics, which breaks the
    batch-invariance the chunking in ReIDExtractor depends on).
    """

    #: short key this backend is registered under, echoed in run banners.
    name: str = "unnamed"

    def __init__(self, device: torch.device):
        self.device = device

    # --- what the pipeline downstream needs to know about the output --------

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Width of the vector `forward` returns. Propagates to the Qdrant
        collection's vector size, so it must be measured, not assumed."""

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """Model input as (Height, Width) -- the resolution this checkpoint was
        trained at. Person crops are tall and narrow, hence the 2:1 shapes."""

    # --- the two halves of a forward pass ----------------------------------

    @abstractmethod
    def preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """One BGR uint8 crop -> normalized (3, H, W) float32 CPU tensor.

        Receives BGR because that is what OpenCV hands us; converting to
        whatever the model wants is part of the recipe this backend owns.
        Callers guarantee the crop is non-empty.
        """

    @abstractmethod
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) on this backend's device -> (N, embedding_dim) RAW
        features. Do NOT L2-normalize here; ReIDExtractor does that once for
        every backend so the unit-norm guarantee cannot be forgotten."""

    # --- shared conveniences ----------------------------------------------

    def describe(self) -> str:
        """One line for the run banner."""
        return f"{self.name} ({self.embedding_dim}-d, {self.input_size[0]}x{self.input_size[1]})"

    def _probe_dim(self) -> int:
        """Measure the output width by running one dummy forward.

        Derived rather than declared on purpose. A hardcoded width that silently
        disagrees with the checkpoint is precisely the class of bug this module
        exists to prevent, and the cost is one forward pass of zeros at load.
        """
        h, w = self.input_size
        with torch.no_grad():
            out = self.forward(torch.zeros(1, 3, h, w, device=self.device))
        if out.ndim != 2 or out.shape[0] != 1:
            raise RuntimeError(
                f"{self.name}: forward() returned shape {tuple(out.shape)}; "
                f"expected (N, D) for a batch of N crops."
            )
        return int(out.shape[1])


# ============================ torchreid OSNet ==============================

# ImageNet train-set per-channel statistics, RGB order, on the 0..1 scale.
# Shaped (3,1,1) so they broadcast over a (3,H,W) tensor. ReID training started
# from ImageNet weights and never changed the normalization, so inference must
# reuse the exact same numbers -- change them and every activation drifts
# off-distribution. NOTE we keep ImageNet *normalization* but NOT ImageNet
# *weights*: the loaded checkpoint is fine-tuned for person ReID, which is what
# makes the embedding separate identities instead of object categories.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# OSNet's canonical person-ReID input as (H, W). Feeding a different size still
# runs (OSNet is fully convolutional + global pooling) but shifts the feature
# statistics away from training -> worse accuracy. Pin it.
OSNET_INPUT_SIZE = (256, 128)


class OSNetBackend(ReIDBackend):
    """The torchreid OSNet family: osnet_x1_0, osnet_ibn_x1_0, osnet_ain_x1_0.

    All three share one architecture skeleton (torchreid's `OSNet`), differing
    only in the normalization layers spliced into it -- so they share this
    backend, the same 256x128 preprocessing, the same 512-d width, the same
    `fc = Sequential(Linear, BatchNorm1d, ReLU)` head, and therefore the same
    tap logic. Swapping among them is a config edit (`reid.model`).

    We depend ONLY on torchreid's model DEFINITION, never its training
    machinery, so this stays a thin inference path that can later be re-pointed
    at a vendored osnet.py for TensorRT without touching the public API.

    THE TAP (#39). torchreid's OSNet ends in fc = Sequential(Linear, BN, ReLU)
    and eval-mode forward() returns that whole block, so the shipped embedding
    is POST-ReLU -- confined to the non-negative orthant (measured: 21.0% of
    dimensions exactly zero, 0% negative), which forces the cosine between ANY
    two embeddings to be >= 0 and squeezes the usable range upward. That is why
    different people here score as high as 0.78-0.94.

    Tapping POST-BN instead keeps the negative half of the space. Measured on
    register_file.avi, post-BN improved the separation margin in EVERY scoring
    mode at BOTH sample sizes (+0.055 -> +0.086 at 48 frames, +0.108 -> +0.157
    at 90) and lowered the different-person ceiling (0.845 -> 0.782 and
    0.892 -> 0.858). The DIRECTION is the stable result; the magnitudes are not.

    CHANGING THE TAP VOIDS EVERY THRESHOLD IN THE SYSTEM -- it is a different
    feature space, not a better version of the same one. Hence a flag that
    defaults to the shipped behaviour.
    """

    #: architectures this backend can build. Restricted to the family that
    #: genuinely shares the head/preprocessing above -- a wider torchreid model
    #: (resnet50_fc512, say) has a different fc block, so the tap logic would
    #: quietly fall back instead of doing what the config asked.
    ARCHS = ("osnet_x1_0", "osnet_ibn_x1_0", "osnet_ain_x1_0")

    # NOTE ON THIS `arch` DEFAULT: it selects which OSNet *this class* builds when
    # constructed without one. It is NOT the pipeline's ReID model, and it cannot
    # select FastReID -- that is a different class further down. What actually runs
    # is `reid.model` in config.yaml (today `fastreid_sbs_R101_ibn`) resolved through
    # the BACKENDS registry, which always passes `arch` explicitly for OSNet entries,
    # so this default is never consulted on the shipped path. Reading it as "we are
    # still on OSNet" is a mistake people have made; the run banner prints the model
    # actually loaded.
    def __init__(self, weights: str, device: torch.device,
                 arch: str = "osnet_ain_x1_0", tap: str = "post_relu"):
        super().__init__(device)
        import torchreid                    # heavy import, deferred to load time

        if arch not in self.ARCHS:
            raise ValueError(
                f"OSNetBackend does not build '{arch}'. Supported: {list(self.ARCHS)}. "
                f"Other torchreid models have a different fc head, so the feature "
                f"tap would not mean the same thing -- give them their own backend."
            )
        self.name = arch
        self.arch = arch
        self.tap = str(tap or "post_relu")
        self._tap_warned = False

        # Build the ARCHITECTURE only (pretrained=False): we supply our own
        # ReID-trained weights below. num_classes here is irrelevant -- it only
        # sizes the classifier head, which we deliberately discard.
        model = torchreid.models.build_model(
            name=arch, num_classes=1000, loss="softmax", pretrained=False,
        )

        # Load the checkpoint EXPLICITLY so the load is transparent and owned
        # here rather than hidden inside a framework helper.
        # weights_only=False: PyTorch >=2.6 defaults to weights_only=True, which
        # rejects the numpy scalars pickled into these legacy torchreid
        # checkpoints. Safe here -- we only ever point this at checkpoints we
        # downloaded ourselves from the trusted torchreid Model Zoo.
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        # Some Model Zoo checkpoints are the bare state_dict; others (like the
        # AIN one) are a full training checkpoint wrapping it under "state_dict",
        # with a "module." prefix left over from DataParallel training.
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model_sd = model.state_dict()
        to_load = {
            k: v for k, v in state_dict.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        dropped = [k for k in state_dict if k not in to_load]
        if dropped != ["classifier.weight", "classifier.bias"]:
            # Anything else missing means the checkpoint doesn't match `arch`
            # -- fail loud rather than emit garbage embeddings. The most likely
            # cause is a config pointing `reid.model` at one architecture and
            # `reid.weights` at another's file.
            raise RuntimeError(
                f"Unexpected checkpoint keys dropped: {dropped}. "
                f"Expected only the classifier head. Does '{weights}' really "
                f"contain {arch} weights?"
            )
        model.load_state_dict(to_load, strict=False)

        # eval() is REQUIRED for two reasons:
        #   1. it freezes BatchNorm to use running stats (not per-batch stats),
        #      so one crop and a batch of crops give identical embeddings;
        #   2. in eval mode OSNet's forward returns the 512-d FEATURE, whereas
        #      in train mode it returns classifier logits.
        model.eval()
        self.model = model.to(self.device)
        self._dim = self._probe_dim()

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def input_size(self) -> Tuple[int, int]:
        return OSNET_INPUT_SIZE

    def preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """The recipe, in order. Inference MUST match training preprocessing or
        embeddings silently degrade -- no error, just worse matching."""
        # 1. BGR -> RGB. OpenCV gives BGR; the model was trained on RGB.
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        # 2. Resize to (W, H) -- cv2 takes (width, height), our constant is
        #    (H, W), hence the reversed indexing. INTER_LINEAR matches the
        #    bilinear interpolation used during training.
        h, w = self.input_size
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        # 3. HWC uint8 -> CHW float32 in 0..1.
        tensor = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1)

        # 4. Per-channel ImageNet standardization.
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Run the model and take features at the configured TAP (#39)."""
        if self.tap != "post_bn":
            return self.model(batch)
        fc = getattr(self.model, "fc", None)
        if fc is None or not hasattr(fc, "__getitem__") or len(fc) < 2:
            # Not the Sequential(Linear, BN, ReLU) we expect -- do not guess at
            # a different architecture's internals, just use the shipped tap.
            if not self._tap_warned:
                print("[reid] tap 'post_bn' requested but this model's `fc` is not "
                      "the expected Sequential(Linear, BatchNorm1d, ReLU); using "
                      "the default post-ReLU tap.")
                self._tap_warned = True
            return self.model(batch)
        # featuremaps -> global pool -> fc[:2] (Linear, BatchNorm1d), stopping
        # before the ReLU. Mirrors torchreid's own eval-mode forward.
        v = self.model.featuremaps(batch)
        v = torch.nn.functional.adaptive_avg_pool2d(v, 1).view(v.size(0), -1)
        for layer in list(fc)[:2]:
            v = layer(v)
        return v

    def describe(self) -> str:
        return (f"{self.name} ({self.embedding_dim}-d, "
                f"{self.input_size[0]}x{self.input_size[1]}, tap={self.tap})")


# ============================ FastReID SBS =================================

# FastReID's SBS configs test at 384x128 (H, W) -- NOT the 256x128 the bagtricks
# base and OSNet use. Read off configs/Base-SBS.yml (`INPUT.SIZE_TEST`). Feeding
# it 256x128 runs and quietly costs accuracy, so this is pinned per-backend.
SBS_INPUT_SIZE = (384, 128)

# R101 block counts and non-local placement, read off upstream's
# build_resnet_backbone depth tables. Verified against the checkpoint: `NL_3` runs
# to index 8, i.e. 9 non-local blocks in stage 3.
_RESNET_SPEC = {
    # depth: (layers, non_local_layers)
    "50x":  ([3, 4, 6, 3],  [0, 2, 3, 0]),
    "101x": ([3, 4, 23, 3], [0, 2, 9, 0]),
}


class FastReIDBackend(ReIDBackend):
    """FastReID's SBS ResNet-IBN family, from the vendored definition.

    Inference path, verified against upstream source (not recalled):

        crop -> BGR->RGB -> resize 384x128 (INTER_CUBIC) -> keep 0..255 scale
             -> (x - mean)/std with 255-scaled ImageNet constants
             -> ResNet-IBN (+ non-local) -> GeM pool -> bnneck (BatchNorm1d)
             -> 2048-d feature

    WHAT THE CHECKPOINT CONTAINS (msmt_sbs_R101-ibn.pth, 836 tensors):
    `backbone.*` (827), `heads.pool_layer.p` (GeM's learnable exponent),
    `heads.bnneck.*` (the 2048-d BatchNorm1d) and `heads.classifier.weight`
    of shape (1041, 2048) -- 1041 being MSMT17's training identity count, which
    is how we know the domain. The classifier is the training-time identity head
    and is discarded, exactly as for OSNet.

    TWO THINGS THAT DIFFER FROM OSNet AND MATTER:

    1. INPUT SIZE is 384x128, not 256x128 (see SBS_INPUT_SIZE). 50% more pixels
       per crop on top of a heavier backbone.

    2. THERE IS NO TAP CHOICE. Upstream's EmbeddingHead.forward returns
       `neck_feat` unconditionally when `not self.training` -- the feature AFTER
       the bnneck BatchNorm. `MODEL.HEADS.NECK_FEAT` (`after` in the SBS config)
       only selects which feature the TRAINING losses see; at eval it is dead.
       So this backend is always post-BN, which is the side our own measurement
       preferred for OSNet (#39) -- but it is not selectable, and a `tap=` that
       silently did nothing would be worse than an error. Hence the raise below.

    Normalisation note: upstream keeps pixels on the 0..255 scale and uses
    PIXEL_MEAN = [0.485, 0.456, 0.406] * 255 with PIXEL_STD = [0.229, 0.224,
    0.225] * 255. That is ALGEBRAICALLY IDENTICAL to OSNet's divide-by-255 then
    standardise -- (x/255 - m)/s == (x - 255m)/(255s) -- so the two recipes agree
    on normalisation and differ only in size and interpolation. We keep upstream's
    formulation so the constants can be diffed against Base-bagtricks.yml directly.
    """

    #: (H, W) upstream tests SBS at.
    ARCHS = ("fastreid_sbs_R50_ibn", "fastreid_sbs_R101_ibn")

    #: RGB, 0..255 scale. Upstream: fastreid/config/defaults.py MODEL.PIXEL_MEAN/STD.
    PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) * 255.0
    PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) * 255.0

    def __init__(self, weights: str, device: torch.device,
                 depth: str = "101x", tap: Optional[str] = None):
        super().__init__(device)
        from torch import nn

        from reid.vendor.fastreid import (Bottleneck, GeneralizedMeanPoolingP,
                                          ResNet)

        if depth not in _RESNET_SPEC:
            raise ValueError(f"FastReIDBackend depth must be one of "
                             f"{sorted(_RESNET_SPEC)}, got {depth!r}.")
        # A tap is meaningless here (see class docstring). Refuse it rather than
        # accept a key that looks load-bearing and is not -- the pipeline passes
        # reid.tap through unconditionally, so silence would be a real trap.
        if tap not in (None, "", "n/a"):
            raise ValueError(
                f"reid.tap={tap!r} is not applicable to {self.__class__.__name__}. "
                f"FastReID's eval feature is always post-bnneck (its NECK_FEAT "
                f"setting affects training only), so there is nothing to select. "
                f"Set `reid.tap: n/a` when `reid.model` is a fastreid_* backend."
            )
        self.name = f"fastreid_sbs_R{depth.rstrip('x')}_ibn"
        self.depth = depth
        layers, non_layers = _RESNET_SPEC[depth]

        # Arguments mirror upstream's build_resnet_backbone under Base-SBS.yml:
        #   LAST_STRIDE 1, NORM "BN", WITH_IBN True, WITH_SE False, WITH_NL True.
        backbone = ResNet(last_stride=1, bn_norm="BN", with_ibn=True,
                          with_se=False, with_nl=True, block=Bottleneck,
                          layers=layers, non_layers=non_layers)
        pool = GeneralizedMeanPoolingP()
        # EMBEDDING_DIM defaults to 0 upstream, which means "no extra projection"
        # -- the bnneck operates directly on the backbone's 2048-d FEAT_DIM.
        bnneck = nn.BatchNorm1d(2048)

        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        # Released FastReID checkpoints are full training checkpoints: the weights
        # live under "model", alongside optimizer/scheduler/iteration state (which
        # is why the R101 file is ~537 MB for ~170 MB of parameters).
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

        got = backbone.load_state_dict(
            {k[len("backbone."):]: v for k, v in state.items()
             if k.startswith("backbone.")}, strict=False)
        if got.missing_keys or got.unexpected_keys:
            raise RuntimeError(
                f"{self.name}: backbone checkpoint mismatch -- "
                f"{len(got.missing_keys)} missing, {len(got.unexpected_keys)} "
                f"unexpected. First missing: {got.missing_keys[:3]}. Wrong depth "
                f"(this is {depth}) or wrong checkpoint file?")
        bnneck.load_state_dict({k[len("heads.bnneck."):]: v
                                for k, v in state.items()
                                if k.startswith("heads.bnneck.")})
        if "heads.pool_layer.p" in state:
            # GeM's exponent is LEARNED. Dropping it silently falls back to p=3
            # and changes every embedding, so it is loaded explicitly and its
            # absence is not tolerated below.
            pool.p.data = state["heads.pool_layer.p"].clone()
        else:
            raise RuntimeError(
                f"{self.name}: checkpoint has no 'heads.pool_layer.p'. The SBS "
                f"configs use a trainable-exponent GeM pool; without it every "
                f"embedding is wrong. Is this really an SBS checkpoint?")

        # Everything we deliberately ignore, stated so a future checkpoint that
        # carries MORE than the training head cannot slip through unnoticed.
        ignored = sorted(k for k in state
                         if not k.startswith(("backbone.", "heads.bnneck.",
                                              "heads.pool_layer.")))
        expected_ignored = ["heads.classifier.weight", "pixel_mean", "pixel_std"]
        if ignored != expected_ignored:
            raise RuntimeError(
                f"{self.name}: unexpected leftover checkpoint keys {ignored}; "
                f"expected only {expected_ignored} (the training-time identity "
                f"head and the normalisation buffers we apply ourselves).")

        for m in (backbone, pool, bnneck):
            m.eval()
            m.to(self.device)
        self.backbone_net, self.pool, self.bnneck = backbone, pool, bnneck
        # `.model` is the one torch module the harness introspects for eval mode.
        # Wrap the three stages so that check covers all of them.
        self.model = nn.Sequential(backbone, pool, bnneck).eval()
        self._mean = self.PIXEL_MEAN.to(self.device)
        self._std = self.PIXEL_STD.to(self.device)
        self._dim = self._probe_dim()

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def input_size(self) -> Tuple[int, int]:
        return SBS_INPUT_SIZE

    def preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        # 1. BGR -> RGB. Upstream's demo/predictor.py says so explicitly ("the
        #    model expects RGB inputs") and PIXEL_MEAN is in RGB order.
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        # 2. Resize to (W, H). INTER_CUBIC, not INTER_LINEAR -- that is what
        #    upstream's inference demo uses, and interpolation changes the
        #    feature statistics enough to be worth matching exactly.
        h, w = self.input_size
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
        # 3. CHW float32, still on the 0..255 scale (NO divide by 255 -- the
        #    constants below are 255-scaled to match).
        return torch.from_numpy(rgb).float().permute(2, 0, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Normalisation happens here rather than in preprocess because upstream
        # does it inside the model (Baseline.preprocess_image, as registered
        # buffers), and keeping it on-device costs nothing.
        x = (batch - self._mean) / self._std
        feat = self.pool(self.backbone_net(x))[..., 0, 0]
        return self.bnneck(feat)

    def describe(self) -> str:
        return (f"{self.name} ({self.embedding_dim}-d, "
                f"{self.input_size[0]}x{self.input_size[1]}, post-bnneck, "
                f"GeM p={float(self.pool.p.detach()):.3f})")


# ============================== the registry ===============================

# config `reid.model` value -> (backend class, extra kwargs). Every OSNet
# variant maps to the same class with a different `arch`, and every FastReID SBS
# variant to the same class with a different `depth`, so trying R50 against R101
# is one config line plus the matching checkpoint.
BACKENDS: Dict[str, Tuple[Type[ReIDBackend], dict]] = {
    "osnet_x1_0":            (OSNetBackend, {"arch": "osnet_x1_0"}),
    "osnet_ibn_x1_0":        (OSNetBackend, {"arch": "osnet_ibn_x1_0"}),
    "osnet_ain_x1_0":        (OSNetBackend, {"arch": "osnet_ain_x1_0"}),
    "fastreid_sbs_R50_ibn":  (FastReIDBackend, {"depth": "50x"}),
    "fastreid_sbs_R101_ibn": (FastReIDBackend, {"depth": "101x"}),
}

#: what `reid.model` defaults to -- the checkpoint every threshold in
#: config.yaml was derived against.
DEFAULT_BACKEND = "osnet_ain_x1_0"


def build_backend(name: Optional[str], weights: str, device: torch.device,
                  **kwargs) -> ReIDBackend:
    """Resolve a `reid.model` key to a loaded backend.

    Unknown keys fail loud with the list of valid ones: a typo'd model name must
    not fall back to the default, because the run banner and every threshold in
    play would then describe a model that isn't running.
    """
    key = str(name or DEFAULT_BACKEND)
    if key not in BACKENDS:
        raise ValueError(
            f"Unknown reid.model '{key}'. Registered backends: "
            f"{sorted(BACKENDS)}. Add one in src/reid/backends.py."
        )
    cls, preset = BACKENDS[key]
    return cls(weights=weights, device=device, **{**preset, **kwargs})
