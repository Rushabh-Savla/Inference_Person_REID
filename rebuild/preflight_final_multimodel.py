from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from reid.nvidia_reid import NVIDIAReIDExtractor
from reid.nvidia_swin import NVIDIASwinReIDExtractor
from reid.solider_reid import SOLIDERReIDExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast final multimodel ReID preflight")
    parser.add_argument("--resnet", default="weights/reid/resnet50_market1501_aicity156.onnx")
    parser.add_argument("--swin", default="weights/reid/nvidia_swin_base_1024/export_55/swin_base_market1501_aicity156_featuredim1024.onnx")
    parser.add_argument("--solider", default="weights/solider_swin_base_msmt17.onnx")
    args = parser.parse_args()

    crop = np.zeros((256, 128, 3), dtype=np.uint8)
    models = [
        NVIDIAReIDExtractor(args.resnet, device="cuda", max_batch=2),
        NVIDIASwinReIDExtractor(args.swin, device="cuda", max_batch=2),
        SOLIDERReIDExtractor(args.solider, device="cuda", max_batch=2),
    ]
    for model in models:
        value = model.extract(crop)
        assert value.ndim == 1 and value.shape[0] == model.embedding_dim
        assert np.all(np.isfinite(value))
        norm = float(np.linalg.norm(value))
        assert 0.999 < norm < 1.001, (model.describe(), norm)
        print(f"[preflight] PASS {model.describe()} dim={value.shape[0]} norm={norm:.6f}")
    print("FINAL MULTIMODEL PREFLIGHT: OK")


if __name__ == "__main__":
    main()
