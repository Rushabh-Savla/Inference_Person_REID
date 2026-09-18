#!/usr/bin/env bash
set -euo pipefail

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
echo "[nvdcf] Rootless runtime: $ROOT"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

read -r DS_ROOT DS_LIB DS_CFG < <(
  "$VENV_PY" - <<'PY'
import os
from pathlib import Path
from rebuild.deepstream_runtime import DeepStreamRuntime

root, lib = DeepStreamRuntime.find()
cfg = DeepStreamRuntime.config(root)
if root and lib and cfg:
    print(root, lib, cfg)
    raise SystemExit(0)

print("", "", "")
PY
)

if [[ -z "$DS_ROOT" || -z "$DS_LIB" ]]; then
  echo "[nvdcf] Installed DeepStream was not found by library-name discovery."
  echo "[nvdcf] Running a targeted executable/library scan now..."
  while IFS= read -r item; do
    [[ -n "$item" ]] || continue
    read -r root cfg <<< "$("$VENV_PY" - "$item" <<'PY'
import sys
from pathlib import Path
from rebuild.deepstream_runtime import DeepStreamRuntime
lib = Path(sys.argv[1]).resolve()
sdk = DeepStreamRuntime.sdk_root(str(lib)) or lib.parent
print(sdk, DeepStreamRuntime.config(sdk) or "")
PY
)"
    if [[ -n "$cfg" ]]; then
      DS_ROOT="$root"
      DS_LIB="$item"
      DS_CFG="$cfg"
      break
    fi
  done < <(
    find /opt /usr/local /home -type f       \( -name 'libnvds_nvmultiobjecttracker.so' -o -name 'libnvds_nvmultiobjecttracker.so.*' \)       2>/dev/null | head -50
  )
fi

if [[ -z "$DS_ROOT" || -z "$DS_LIB" ]]; then
  echo "[nvdcf] ERROR: actual NvDCF library is not installed on this host."
  echo "[nvdcf] No DeepStream root/library was fabricated or substituted."
  exit 1
fi

DS_VERSION="$("$VENV_PY" - "$DS_ROOT" <<'PY'
import sys
from rebuild.deepstream_runtime import DeepStreamRuntime
from pathlib import Path
print(DeepStreamRuntime.version(Path(sys.argv[1])))
PY
)"
DS_MM="${DS_VERSION%.*}"
PY_MINOR="$("$VENV_PY" -c 'import sys; print(sys.version_info.minor)')"

echo "[nvdcf] DeepStream root: $DS_ROOT"
echo "[nvdcf] DeepStream version: $DS_VERSION"
echo "[nvdcf] NvDCF library: $DS_LIB"
echo "[nvdcf] NvDCF config: ${DS_CFG:-repo-config}"

# Rootless GI/GStreamer runtime. No sudo is used.
cd "$tmp"
apt-get download python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0 >/dev/null
for deb in "$tmp"/*.deb; do
  [[ -f "$deb" ]] || continue
  dpkg-deb -x "$deb" "$ROOT"
done

"$VENV_PY" -m pip install --force-reinstall --no-deps "numpy==1.26.4" >/dev/null

export NVDCF_RUNTIME="$ROOT"
export NVDCF_DEEPSTREAM_ROOT="$DS_ROOT"
export PYTHONPATH="$REPO_DIR:$ROOT/usr/lib/python3/dist-packages:$ROOT/usr/lib/python3.12/dist-packages:${PYTHONPATH:-}"
DS_LIB_DIR="$("$VENV_PY" - "$DS_LIB" <<'PY'
import sys
from rebuild.deepstream_runtime import DeepStreamRuntime
print(DeepStreamRuntime.library_dir(sys.argv[1]))
PY
)"
export LD_LIBRARY_PATH="$DS_LIB_DIR:$DS_ROOT/lib:$DS_ROOT/lib/gst-plugins:$ROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export GST_PLUGIN_PATH="$DS_LIB_DIR:$DS_ROOT/lib/gst-plugins:${GST_PLUGIN_PATH:-}"
if [[ -d "$ROOT/usr/lib/x86_64-linux-gnu/girepository-1.0" ]]; then
  export GI_TYPELIB_PATH="$ROOT/usr/lib/x86_64-linux-gnu/girepository-1.0:${GI_TYPELIB_PATH:-}"
fi

wheel=""

# Reuse an already-installed NVIDIA pyds binding when this host has one.
PYDS_HOST="$("$VENV_PY" - <<'PY'
import sys
from rebuild.deepstream_runtime import DeepStreamRuntime
item = DeepStreamRuntime.pyds()
print(item if item is not None and item.suffix == ".so" else "")
PY
)"
if [[ -n "$PYDS_HOST" ]]; then
  PYDS_DIR="$(dirname "$PYDS_HOST")"
  export PYTHONPATH="$PYDS_DIR:$PYTHONPATH"
fi
if "$VENV_PY" - <<'PY'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import pyds
print(pyds.__file__)
PY
then
  echo "[nvdcf] Existing PyDS import is usable; skipping rebuild."
  wheel="EXISTING"
fi

if [[ "$wheel" == "EXISTING" ]]; then
  :
elif [[ "$DS_VERSION" =~ ^8\.0\.[0-9]+$ ]] && [[ "$(uname -m)" == "x86_64" ]] && [[ "$PY_MINOR" == "12" ]]; then
  wheel="$tmp/pyds-1.2.2-cp312-cp312-linux_x86_64.whl"
  url="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl"
  echo "[nvdcf] Downloading NVIDIA PyDS 1.2.2 for DeepStream 8.0"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 "$url" -o "$wheel"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$wheel"
  else
    echo "[nvdcf] curl or wget is required."
    exit 1
  fi
fi

if [[ "$wheel" == "EXISTING" ]]; then
  :
elif [[ -z "$wheel" ]]; then
  for item in "$DS_ROOT/lib"/pyds*.whl "$DS_ROOT/sources/deepstream_python_apps/bindings/dist"/pyds*.whl "$DS_ROOT/sources/deepstream_python_apps/bindings/dist"/*.whl; do
    if [[ -f "$item" ]]; then
      wheel="$item"
      break
    fi
  done
fi

if [[ "$wheel" == "EXISTING" ]]; then
  :
elif [[ -z "$wheel" && -d "$DS_ROOT/sources/includes" ]]; then
  bind="$DS_ROOT/sources/deepstream_python_apps/bindings"
  if [[ ! -f "$bind/pyproject.toml" && ! -f "$bind/setup.py" ]]; then
    echo "[nvdcf] PyDS source not installed; cloning NVIDIA deepstream_python_apps rootlessly."
    bind="$ROOT/deepstream_python_apps/bindings"
    rm -rf "$ROOT/deepstream_python_apps"
    git clone --depth 1 https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git "$ROOT/deepstream_python_apps"
    (cd "$ROOT/deepstream_python_apps" && git submodule update --init --recursive)
  fi
  if [[ ! -f "$bind/pyproject.toml" && ! -f "$bind/setup.py" ]]; then
    echo "[nvdcf] ERROR: NVIDIA PyDS source checkout is incomplete."
    exit 1
  fi
  echo "[nvdcf] Building PyDS against $DS_ROOT"
  "$VENV_PY" -m pip install --upgrade --no-deps build pyproject-hooks >/dev/null
  export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
  export CMAKE_ARGS="-DDS_PATH=$DS_ROOT -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=$PY_MINOR"
  "$VENV_PY" -m build --wheel "$bind"
  wheel="$(find "$bind/dist" -maxdepth 1 -type f -name 'pyds-*.whl' -print -quit)"
  [[ -n "$wheel" ]] || { echo "[nvdcf] ERROR: PyDS wheel build produced no wheel."; exit 1; }
fi

if [[ -z "$wheel" ]]; then
  echo "[nvdcf] ERROR: no usable NVIDIA PyDS binding was found, and the discovered NvDCF library directory is not a complete DeepStream SDK."
  echo "[nvdcf] Found NvDCF: $DS_LIB"
  echo "[nvdcf] Expected either an importable pyds binding or a DeepStream SDK containing sources/includes."
  exit 1
fi

if [[ "$wheel" != "EXISTING" ]]; then
  echo "[nvdcf] Installing PyDS: $wheel"
  "$VENV_PY" -m pip install --no-deps --force-reinstall "$wheel"
fi

"$VENV_PY" - <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["NVDCF_RUNTIME"])
for path in (root / "usr/lib/python3/dist-packages", root / "usr/lib/python3.12/dist-packages"):
    if path.exists():
        sys.path.insert(0, str(path))

from rebuild.deepstream_runtime import DeepStreamRuntime
dsroot, library, config = DeepStreamRuntime.require()

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import pyds

Gst.init(None)
tracker = Gst.ElementFactory.make("nvtracker", "runtime_probe_tracker")
if tracker is None:
    raise RuntimeError("DeepStream nvtracker GStreamer element is unavailable")
tracker.set_property("ll-lib-file", library)
tracker.set_property("ll-config-file", config)
tracker.set_property("gpu-id", 0)

print("[nvdcf] DeepStream runtime: OK")
print("[nvdcf] root:", dsroot)
print("[nvdcf] version:", DeepStreamRuntime.version(dsroot))
print("[nvdcf] NvDCF library:", library)
print("[nvdcf] NvDCF config:", config)
print("[nvdcf] GStreamer:", Gst.version_string())
print("[nvdcf] nvtracker element: OK")
print("[nvdcf] PyDS:", pyds.__file__)
try:
    import onnxruntime as ort
    print("[nvdcf] ORT providers:", ort.get_available_providers())
except Exception as exc:
    print("[nvdcf] ORT providers: unavailable:", exc)
PY

echo "[nvdcf] Rootless NvDCF runtime setup complete."
