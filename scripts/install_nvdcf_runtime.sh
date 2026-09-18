#!/usr/bin/env bash
set -euo pipefail

DS_ROOT="/opt/nvidia/deepstream/deepstream"
VENV_PY="$(command -v python)"
ROOT="${VIRTUAL_ENV}/.nvdcf_runtime"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate the project venv first."
  exit 1
fi

if [[ ! -d "$DS_ROOT" ]]; then
  echo "[nvdcf] DeepStream root not found: $DS_ROOT"
  exit 1
fi

mkdir -p "$ROOT"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "[nvdcf] Python: $VENV_PY"
echo "[nvdcf] DeepStream root: $DS_ROOT"
echo "[nvdcf] Rootless runtime: $ROOT"

# Read the installed DeepStream SDK version without requiring root.
major=""
minor=""
micro=""
header="$DS_ROOT/sources/includes/nvds_version.h"
if [[ -f "$header" ]]; then
  major="$(awk '/NVDS_VERSION_MAJOR/{print $3; exit}' "$header")"
  minor="$(awk '/NVDS_VERSION_MINOR/{print $3; exit}' "$header")"
  micro="$(awk '/NVDS_VERSION_MICRO/{print $3; exit}' "$header")"
fi
if [[ -z "$major" ]]; then
  pkg="$(dpkg-query -W -f='${Version}' 'deepstream-*' 2>/dev/null | head -n1 || true)"
  if [[ "$pkg" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
    major="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"
    micro="${BASH_REMATCH[3]}"
  fi
fi
major="${major:-unknown}"
minor="${minor:-unknown}"
micro="${micro:-unknown}"
echo "[nvdcf] Detected DeepStream: $major.$minor.$micro"

# The project runs Python 3.12 on x86_64. Install the matching Ubuntu GI/GStreamer
# runtime packages into the venv-local root without sudo/root.
if command -v apt-get >/dev/null 2>&1 && [[ -d "$ROOT/usr/lib" ]]; then
  cd "$tmp"
  apt-get download python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0 >/dev/null
  for deb in "$tmp"/*.deb; do
    [[ -f "$deb" ]] || continue
    dpkg-deb -x "$deb" "$ROOT"
  done
fi

# Keep the project's CUDA ONNX Runtime and model stack intact.
"$VENV_PY" -m pip install --force-reinstall --no-deps "numpy==1.26.4" >/dev/null

wheel=""

# DeepStream 8.0 has an official CPython 3.12 x86_64 PyDS wheel.
# NVIDIA released this as pyds 1.2.2 for DS 8.0.
if [[ "$major.$minor" == "8.0" ]] && [[ "$(uname -m)" == "x86_64" ]] && [[ "$("$VENV_PY" -c 'import sys; print(sys.version_info.minor)')" == "12" ]]; then
  wheel="$tmp/pyds-1.2.2-cp312-cp312-linux_x86_64.whl"
  url="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl"
  echo "[nvdcf] Downloading official DeepStream 8.0 PyDS wheel"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 "$url" -o "$wheel"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress "$url" -O "$wheel"
  else
    echo "[nvdcf] curl/wget is required to download the PyDS wheel."
    exit 1
  fi
fi

# Prefer a locally shipped PyDS wheel/source when one is already present.
if [[ -z "$wheel" ]]; then
  for item in \
    "$DS_ROOT/lib"/pyds*.whl \
    "$DS_ROOT/sources/deepstream_python_apps/bindings/dist"/pyds*.whl \
    "$DS_ROOT/sources/deepstream_python_apps/bindings/dist"/*.whl
  do
    if [[ -f "$item" ]]; then
      wheel="$item"
      break
    fi
  done
fi

if [[ -n "$wheel" ]]; then
  echo "[nvdcf] Installing PyDS wheel: $wheel"
  "$VENV_PY" -m pip install --no-deps --force-reinstall "$wheel"
else
  bind="$DS_ROOT/sources/deepstream_python_apps/bindings"
  if [[ ! -f "$bind/pyproject.toml" && ! -f "$bind/setup.py" ]]; then
    bind="$ROOT/deepstream_python_apps/bindings"
    if [[ ! -d "$bind" ]]; then
      echo "[nvdcf] No compatible PyDS wheel/source is installed."
      echo "[nvdcf] DeepStream $major.$minor requires a matching PyDS binding."
      echo "[nvdcf] For DS 9.x NVIDIA no longer publishes PyDS wheels; build the bindings from source."
      echo "[nvdcf] Required source location: $DS_ROOT/sources/deepstream_python_apps/bindings"
      echo "[nvdcf] This installer will not modify /opt and will never use sudo."
      exit 1
    fi
  fi
  echo "[nvdcf] Building/installing PyDS from: $bind"
  "$VENV_PY" -m pip install --no-deps --no-build-isolation "$bind"
fi

# Export both the local GI runtime and the real DeepStream libraries.
export NVDCF_RUNTIME="$ROOT"
export PYTHONPATH="$ROOT/usr/lib/python3/dist-packages:$ROOT/usr/lib/python3.12/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$DS_ROOT/lib:$DS_ROOT/lib/gst-plugins:$ROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
if [[ -d "$ROOT/usr/lib/x86_64-linux-gnu/girepository-1.0" ]]; then
  export GI_TYPELIB_PATH="$ROOT/usr/lib/x86_64-linux-gnu/girepository-1.0:${GI_TYPELIB_PATH:-}"
fi

"$VENV_PY" - <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["NVDCF_RUNTIME"])
for path in (
    root / "usr/lib/python3/dist-packages",
    root / "usr/lib/python3.12/dist-packages",
    root / "usr/lib/x86_64-linux-gnu/girepository-1.0",
    Path("/opt/nvidia/deepstream/deepstream/lib"),
):
    if path.exists():
        sys.path.insert(0, str(path))

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
