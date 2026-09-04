#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def main() -> int:
    print(f"python: {sys.executable}")
    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"torch cuda available: {torch.cuda.is_available()}")
        print(f"torch cuda version: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch check failed: {exc}")
        return 1

    try:
        import onnxruntime as ort
        print(f"onnxruntime: {ort.__version__}")
        print(f"onnxruntime package: {ort.__file__}")
        providers = ort.get_available_providers()
        print(f"providers: {providers}")
        if "CUDAExecutionProvider" not in providers:
            print("ERROR: CUDAExecutionProvider is unavailable.")
            print("Fix: uninstall CPU onnxruntime and reinstall onnxruntime-gpu==1.28.0.")
            return 2
    except Exception as exc:
        print(f"onnxruntime check failed: {exc}")
        return 3

    cuda13 = os.environ.get("CUDA13", "")
    cudnn = os.environ.get("CUDNN", "")
    print(f"CUDA13: {cuda13}")
    print(f"CUDNN: {cudnn}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
    print("runtime check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
