#!/usr/bin/env bash
set -euo pipefail

# InsightFace 0.7.3 declares the CPU `onnxruntime` package as a dependency.
# Keep dependency resolution isolated so it cannot replace the project's
# CUDA ONNX Runtime 1.28.x. The 0.7.3 release also requires albumentations,
# qudida, easydict and prettytable for the FaceAnalysis import path.
python -m pip install --upgrade --no-deps "insightface==0.7.3"
python -m pip install --upgrade --no-deps \
  "albumentations==1.3.1" \
  "qudida==0.0.4" \
  "easydict==1.13" \
  "prettytable==3.12.0"

# Restore the repository's numerics after any prior environment drift.
python -m pip install --force-reinstall --no-deps \
  "numpy==2.2.6" \
  "scipy==1.15.3"

# Explicitly keep CUDA ONNX Runtime as the only ORT distribution.
python -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
python -m pip install --force-reinstall --no-deps "onnxruntime-gpu==1.28.0"

python - <<'PY'
import numpy as np
import scipy
import onnxruntime as ort
import albumentations as A

print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("albumentations:", A.__version__)
print("onnxruntime:", ort.__version__)
print("providers:", ort.get_available_providers())
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit(
        "CUDAExecutionProvider is missing; keep onnxruntime-gpu 1.28.x installed."
    )
PY

python - <<'PY'
from insightface.app import FaceAnalysis
print("InsightFace import: OK")
print("FaceAnalysis buffalo_l: available")
PY