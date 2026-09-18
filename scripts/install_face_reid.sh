#!/usr/bin/env bash
set -euo pipefail

VENV_PY="$(command -v python)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate the project venv first."
  exit 1
fi

# Install InsightFace without dependency resolution. This preserves the CUDA
# ONNX Runtime already pinned by the project.
"$VENV_PY" -m pip install --upgrade --no-deps "insightface==2.0.0"

# FaceAnalysis Python dependencies, also without dependency resolution.
"$VENV_PY" -m pip install --upgrade --no-deps easydict prettytable requests tqdm scikit-learn onnx matplotlib

"$VENV_PY" -m pip install --force-reinstall --no-deps "numpy==1.26.4" "onnxruntime-gpu==1.28.0"

"$VENV_PY" - <<'PY'
import numpy as np
import onnxruntime as ort
import torch
from insightface.app import FaceAnalysis

if np.__version__ != "1.26.4":
    raise SystemExit(f"Expected numpy==1.26.4, got {np.__version__}")
if ort.__version__ != "1.28.0":
    raise SystemExit(f"Expected onnxruntime-gpu==1.28.0, got {ort.__version__}")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("CUDAExecutionProvider is unavailable")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA")

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=0, det_size=(640, 640))

session_providers = []
for face in getattr(app, "models", {}).values():
    session = getattr(face, "session", None)
    if session is not None:
        session_providers.extend(session.get_providers())

if session_providers and "CUDAExecutionProvider" not in session_providers:
    raise SystemExit(f"Face models are not using CUDA: {session_providers}")

print("[face] InsightFace 2.0.0: OK")
print("[face] buffalo_l: OK")
print("[face] CUDAExecutionProvider: OK")
print("[face] ORT:", ort.__version__)
print("[face] torch CUDA:", torch.version.cuda)
PY

echo "[face] Face runtime ready: SCRFD + ArcFace, visibility gate >= 0.68."
