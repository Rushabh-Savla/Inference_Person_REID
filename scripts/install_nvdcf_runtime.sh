#!/usr/bin/env bash
set -euo pipefail

DS_ROOT="/opt/nvidia/deepstream/deepstream"
VENV_PY="$(command -v python)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate the project venv first."
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
  echo "[nvdcf] No PyDS wheel found; checking for a system-installed pyds."
fi

"$VENV_PY" -m pip install --force-reinstall --no-deps "numpy==1.26.4"

"$VENV_PY" - <<'PY'
import sys
from pathlib import Path

for path in (
    "/usr/lib/python3/dist-packages",
    "/usr/lib/python3.12/dist-packages",
    "/opt/nvidia/deepstream/deepstream/lib",
    "/opt/nvidia/deepstream/deepstream/lib/python",
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
