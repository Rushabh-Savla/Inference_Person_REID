#!/usr/bin/env bash
set -euo pipefail

ROOT="${NVDCF_DEEPSTREAM_ROOT:-/opt/nvidia/deepstream/deepstream}"
TRACKER="${NVDCF_TRACKER_LIBRARY:-$ROOT/lib/libnvds_nvmultiobjecttracker.so}"
QDRANT="${QDRANT_URL:-http://127.0.0.1:6333}"
export PYTHONPATH="/workspace/inference_person_reid:/workspace/inference_person_reid/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/lib/gst-plugins:${LD_LIBRARY_PATH:-}"
export GST_PLUGIN_PATH="$ROOT/lib/gst-plugins:${GST_PLUGIN_PATH:-}"

test -f "$ROOT/lib/libnvds_meta.so"
test -f "$TRACKER"

python3 - "$ROOT" "$TRACKER" "$QDRANT" <<'PY'
import ctypes
import sys
from pathlib import Path

root = Path(sys.argv[1])
tracker = Path(sys.argv[2])
qurl = sys.argv[3]
mode = __import__("os").RTLD_NOW | __import__("os").RTLD_GLOBAL

ctypes.CDLL(str(root / "lib/libnvds_meta.so"), mode=mode)
ctypes.CDLL(str(tracker), mode=mode)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)

import pyds
from qdrant_client import QdrantClient
import onnxruntime as ort
import torch

assert Gst.ElementFactory.make("nvtracker", "verify_nvtracker") is not None
assert "CUDAExecutionProvider" in ort.get_available_providers()
assert torch.cuda.is_available()

client = QdrantClient(url=qurl, timeout=10.0)
client.get_collections()

print("DEEPSTREAM_CORE=OK")
print("NVDCF_LIBRARY=OK")
print("PYDS=OK")
print("GST_NVTRACKER=OK")
print("TORCH_CUDA=OK")
print("ORT_CUDA=OK")
print("QDRANT=OK")
print("RUNTIME=PASS")
PY

python3 -m pytest -q     rebuild/test_assignment_guard.py     rebuild/test_overlap_guard.py     rebuild/test_nvdcf_contract.py     rebuild/test_docker_contract.py
