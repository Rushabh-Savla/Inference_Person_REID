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
import sys
from urllib.parse import urlparse
import socket
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
        echo "[docker] ERROR: Qdrant did not become reachable at $QDRANT_URL"
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

mode = os.RTLD_NOW | os.RTLD_GLOBAL
ctypes.CDLL(os.environ["NVDCF_DEEPSTREAM_ROOT"] + "/lib/libnvds_meta.so", mode=mode)
ctypes.CDLL(os.environ["NVDCF_TRACKER_LIBRARY"], mode=mode)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)

import pyds
print("[docker] DeepStream:", os.environ["NVDCF_DEEPSTREAM_ROOT"])
print("[docker] PyDS:", pyds.__file__)
print("[docker] GStreamer:", Gst.version_string())

value = subprocess.run(
    ["gst-inspect-1.0", "nvtracker"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    check=False,
)
if value.returncode != 0:
    raise SystemExit("[docker] ERROR: nvtracker GStreamer plugin unavailable")
print("[docker] nvtracker: OK")

try:
    import torch
    print("[docker] Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("[docker] ERROR: CUDA torch is unavailable")
except ModuleNotFoundError:
    raise SystemExit("[docker] ERROR: PyTorch is missing from the DeepStream container")
PY

exec "$@"