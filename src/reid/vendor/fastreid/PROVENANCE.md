# Vendored FastReID (model definition only)

Upstream: <https://github.com/JDAI-CV/fast-reid> — Apache-2.0
Commit: `c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d` (master, fetched 2026-07-31)

## Why vendored rather than a dependency

FastReID **has no `setup.py`** and is not on PyPI as an official package, so
`pip install` is not an option — upstream's own instructions are "clone and add to
`PYTHONPATH`". Beyond that, importing it as a package is unusable here:
`fastreid/modeling/__init__.py` eagerly imports every backbone, including RegNet,
which needs `yacs` at module scope; `fastreid/utils` pulls `termcolor` and
`tabulate` for coloured logging. None of that has anything to do with running a
forward pass.

This mirrors the decision already documented for torchreid in
`src/reid/extractor.py`: **depend on the model DEFINITION, not the training
machinery.** The five files below need only `torch` — no yacs, no termcolor, no
config system, no dataset registry.

## Files copied verbatim

| File | Upstream path |
|---|---|
| `batch_norm.py` | `fastreid/layers/batch_norm.py` (`IBN`, `get_norm`) |
| `non_local.py` | `fastreid/layers/non_local.py` (`Non_local`) |
| `se_layer.py` | `fastreid/layers/se_layer.py` (`SELayer`) |
| `pooling.py` | `fastreid/layers/pooling.py` (`GeneralizedMeanPoolingP` = GeM) |
| `resnet.py` | `fastreid/modeling/backbones/resnet.py` (`ResNet`, `Bottleneck`) |

## Edits made

Only to `resnet.py`, and only to imports:

1. `from fastreid.layers import (IBN, SELayer, Non_local, get_norm)` → relative
   imports of the four vendored siblings.
2. Removed `from fastreid.utils.checkpoint import ...` and
   `from fastreid.utils import comm` — used only for logging messages while
   loading ImageNet pretrain weights.
3. Removed `from .build import BACKBONE_REGISTRY` and, with it, the
   `build_resnet_backbone(cfg)` function. It reads a yacs config and downloads
   ImageNet weights; `FastReIDBackend` constructs `ResNet(...)` directly with
   explicit arguments and loads our own MSMT17 checkpoint.

No changes to any layer or forward pass. `ResNet`, `Bottleneck`, `IBN`,
`Non_local`, `SELayer` and `GeneralizedMeanPoolingP` are byte-identical to
upstream.

## Refreshing

Re-copy the five files from the same paths, re-apply the three import edits, then
run `python tests/calibration/verify_embedding_contract.py`, which loads the real
checkpoint and asserts every parameter is accounted for.
