#!/usr/bin/env bash
set -euo pipefail

DS_ROOT="/opt/nvidia/deepstream/deepstream"
VENV_PY="$(command -v python)"
ROOT="${VIRTUAL_ENV}/.nvdcf_runtime"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate the project venv first."
  exit 1
fi

mkdir -p "$ROOT"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "[nvdcf] Python: $VENV_PY"
echo "[nvdcf] DeepStream root: $DS_ROOT"
echo "[nvdcf] Rootless runtime: $ROOT"

# No sudo/root is used. apt-get download only downloads .deb archives;
# dpkg-deb -x extracts them into the project venv.
cd "$tmp"
apt-get download python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0

for deb in "$tmp"/*.deb; do
  dpkg-deb -x "$deb" "$ROOT"
done

# Keep the project's CUDA ONNX Runtime and model stack intact.
"$VENV_PY" -m pip install --force-reinstall --no-deps "numpy==1.26.4"

# Locate a prebuilt PyDS wheel when the installed DeepStream release ships one.
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
  bind="$DS_ROOT/sources/deepstream_python_apps/bindings"
  if [[ -f "$bind/setup.py" || -f "$bind/pyproject.toml" ]]; then
    echo "[nvdcf] No PyDS wheel found; installing/building PyDS from: $bind"
    "$VENV_PY" -m pip install --no-deps --no-build-isolation "$bind"
  else
    echo "[nvdcf] PyDS source/wheel not found under $DS_ROOT."
  fi
fi

export NVDCF_RUNTIME="$ROOT"
export PYTHONPATH="$ROOT/usr/lib/python3/dist-packages:$ROOT/usr/lib/python3.12/dist-packages:${PYTHONPATH:-}"

"$VENV_PY" - <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["NVDCF_RUNTIME"])
for path in (
    root / "usr/lib/python3/dist-packages",
    root / "usr/lib/python3.12/dist-packages",
    root / "usr/lib/x86_64-linux-gnu/girepository-1.0",
):
    if path.exists():
        sys.path.insert(0, str(path))

typelib = root / "usr/lib/x86_64-linux-gnu/girepository-1.0"
libs = root / "usr/lib/x86_64-linux-gnu"
if typelib.exists():
    os.environ["GI_TYPELIB_PATH"] = str(typelib) + os.pathsep + os.environ.get("GI_TYPELIB_PATH", "")
if libs.exists():
    os.environ["LD_LIBRARY_PATH"] = str(libs) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import pyds

Gst.init(None)
print("DeepStream bindings: OK")
print("Gst:", Gst.version_string())
print("PyDS:", getattr(pyds, "__file__", "loaded"))
PY

echo "[nvdcf] Rootless DeepStream setup complete."
