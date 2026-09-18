#!/usr/bin/env bash
set -euo pipefail

# DeepStream's GObject/GStreamer Python bindings are normally installed
# system-wide, while this project uses an existing ML venv. Install the
# system GI packages and make a PyDS wheel available to that venv without
# changing its CUDA ONNX Runtime stack.

DS_ROOT="/opt/nvidia/deepstream/deepstream"
VENV_PY="$(command -v python)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate the project's venv before running this script."
  exit 1
fi

echo "[nvdcf] Python: $VENV_PY"
echo "[nvdcf] DeepStream root: $DS_ROOT"

sudo apt-get update
sudo apt-get install -y   python3-gi   python3-gst-1.0   gir1.2-gstreamer-1.0   gstreamer1.0-tools

wheel=""
for item in   "$DS_ROOT"/lib/pyds*.whl   "$DS_ROOT"/sources/deepstream_python_apps/bindings/dist/pyds*.whl   "$DS_ROOT"/sources/deepstream_python_apps/bindings/dist/*.whl
do
  if [[ -f "$item" ]]; then
    wheel="$item"
    break
  fi
done

if [[ -n "$wheel" ]]; then
  echo "[nvdcf] Installing PyDS wheel: $wheel"
  "$VENV_PY" -m pip install --no-deps --force-reinstall "$wheel"
else
  echo "[nvdcf] No PyDS wheel found under $DS_ROOT."
  echo "[nvdcf] NVIDIA's current DeepStream Python bindings are built from"
  echo "[nvdcf] deepstream_python_apps/bindings when a prebuilt wheel is absent."
fi

"$VENV_PY" - <<'PY'
import sys
from pathlib import Path

for path in (
    "/usr/lib/python3/dist-packages",
    "/usr/lib/python3.12/dist-packages",
    "/opt/nvidia/deepstream/deepstream/lib",
):
    if Path(path).exists() and path not in sys.path:
        sys.path.insert(0, path)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import pyds

Gst.init(None)
print("DeepStream bindings: OK")
print("Gst:", Gst.version_string())
print("pyds:", getattr(pyds, "__file__", "loaded"))
PY
