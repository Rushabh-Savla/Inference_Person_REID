#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade "insightface==0.7.3" --no-deps

python - <<'PY'
import onnxruntime as ort
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
