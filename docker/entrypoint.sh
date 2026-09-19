#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/workspace/inference_person_reid:/workspace/inference_person_reid/src:${PYTHONPATH:-}"
export NVDCF_DEEPSTREAM_ROOT="${NVDCF_DEEPSTREAM_ROOT:-/opt/nvidia/deepstream/deepstream}"
export NVDCF_TRACKER_LIBRARY="${NVDCF_TRACKER_LIBRARY:-/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so}"
export GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${GST_PLUGIN_PATH:-}"
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${LD_LIBRARY_PATH:-}"
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"

for attempt in $(seq 1 60); do
    if python3 - "$QDRANT_URL" <<'PY'
import socket
import sys
from urllib.parse import urlparse

value = urlparse(sys.argv[1])
host = value.hostname or "127.0.0.1"
port = value.port or 6333

try:
    with socket.create_connection((host, port), timeout=1.0):
        raise SystemExit(0)
except OSError:
    raise SystemExit(1)
PY
    then
        break
    fi
    if [[ "$attempt" == "60" ]]; then
        echo "[docker] ERROR: Qdrant is not reachable at $QDRANT_URL"
        exit 1
    fi
    sleep 1
done

test -f "${NVDCF_DEEPSTREAM_ROOT}/lib/libnvds_meta.so"
test -f "${NVDCF_TRACKER_LIBRARY}"

python3 - <<'PY'
import ctypes
import os
import subprocess
from pathlib import Path

root = Path(os.environ["NVDCF_DEEPSTREAM_ROOT"])
tracker = Path(os.environ["NVDCF_TRACKER_LIBRARY"])
mode = os.RTLD_NOW | os.RTLD_GLOBAL

ctypes.CDLL(str(root / "lib/libnvds_meta.so"), mode=mode)
ctypes.CDLL(str(tracker), mode=mode)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)

import pyds
if not getattr(pyds, "__file__", None):
    raise SystemExit("[docker] ERROR: PyDS is unavailable")

probe = subprocess.run(
    ["gst-inspect-1.0", "nvtracker"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    check=False,
)
if probe.returncode != 0:
    raise SystemExit("[docker] ERROR: nvtracker GStreamer plugin unavailable")

import onnxruntime as ort
import torch

if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("[docker] ERROR: ONNX Runtime CUDAExecutionProvider unavailable")
if not torch.cuda.is_available():
    raise SystemExit("[docker] ERROR: PyTorch CUDA unavailable")

print("[docker] DeepStream core: OK")
print("[docker] DeepStream root:", root)
print("[docker] NvDCF library: OK")
print("[docker] PyDS:", pyds.__file__)
print("[docker] GStreamer:", Gst.version_string())
print("[docker] nvtracker: OK")
print("[docker] ORT providers:", ort.get_available_providers())
print("[docker] Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("[docker] Qdrant:", os.environ["QDRANT_URL"])
print("[docker] Strict runtime: READY")
PY

exec "$@"
