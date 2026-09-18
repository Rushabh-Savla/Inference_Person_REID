#!/usr/bin/env bash
set -euo pipefail

# The active pipeline no longer uses InsightFace, ArcFace or SCRFD.
# This command is retained as a compatibility preflight for old workflows.
python - <<'PY'
import numpy as np
import scipy
import onnxruntime as ort

print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("onnxruntime:", ort.__version__)
print("providers:", ort.get_available_providers())

if np.__version__ != "2.2.6":
    raise SystemExit("Expected numpy==2.2.6")
if scipy.__version__ != "1.15.3":
    raise SystemExit("Expected scipy==1.15.3")
if ort.__version__ != "1.28.0":
    raise SystemExit("Expected onnxruntime-gpu==1.28.0")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("CUDAExecutionProvider is missing")
PY

echo "Face runtime: DISABLED"
echo "Identity runtime: NVIDIA ReID + NVIDIA Swin + SOLIDER + clothing + pose + Qdrant"
echo "No InsightFace/ArcFace/SCRFD packages are required."
