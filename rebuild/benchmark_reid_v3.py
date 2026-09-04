from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def unit(value):
    value = np.asarray(value, dtype=np.float32)
    return value / (np.linalg.norm(value, axis=1, keepdims=True) + 1e-12)


def imgs(root: Path, limit: int) -> List[Path]:
    values = sorted(root.glob("**/*.jpg"))
    return values[:limit]


def nvidia(path: str, images: List[np.ndarray], batch: int = 16):
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    out = session.get_outputs()[0].name
    values = []
    start = time.perf_counter()
    for i in range(0, len(images), batch):
        pack = []
        for image in images[i:i + batch]:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            rgb = (rgb - np.asarray([123.675, 116.280, 103.530], dtype=np.float32)) * 0.01735207
            pack.append(np.transpose(rgb, (2, 0, 1)))
        values.append(session.run([out], {name: np.stack(pack)})[0])
    value = unit(np.concatenate(values, axis=0))
    elapsed = time.perf_counter() - start
    return value, elapsed


def fast(cfg: str, images: List[np.ndarray]):
    import yaml
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from reid.extractor import ReIDExtractor
    with open(cfg, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    reid = data["reid"]
    dev = None if str(reid.get("device", "auto")).lower() == "auto" else reid.get("device")
    engine = ReIDExtractor(reid["weights"], dev, int(reid.get("max_batch", 32)), model=reid.get("model"))
    start = time.perf_counter()
    value = engine.extract_batch(images)
    elapsed = time.perf_counter() - start
    return value, elapsed, engine.describe()


def separation(value):
    if len(value) < 2:
        return {"count": 0}
    sim = value @ value.T
    upper = sim[np.triu_indices(len(sim), 1)]
    return {
        "count": int(len(upper)),
        "min": float(upper.min()),
        "mean": float(upper.mean()),
        "p50": float(np.percentile(upper, 50)),
        "p90": float(np.percentile(upper, 90)),
        "p95": float(np.percentile(upper, 95)),
        "max": float(upper.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Controlled ReID model benchmark on the same cached crops")
    parser.add_argument("--crops", default="rebuild_outputs/cache_v3/crops")
    parser.add_argument("--config", default="rebuild/config_v3.yaml")
    parser.add_argument("--nvidia", default="models/nvidia_reid/resnet50_market1501_aicity156.onnx")
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()

    paths = imgs(Path(args.crops), args.limit)
    if not paths:
        raise SystemExit("No V3 crops found. Run the V3 batch first.")
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is not None:
            images.append(image)
    print(f"crops={len(images)}")

    fast_value, fast_time, fast_name = fast(args.config, images)
    print(f"FASTREID: {fast_name}")
    print(f"  dim={fast_value.shape[1]} seconds={fast_time:.3f} images_per_sec={len(images)/max(fast_time,1e-6):.1f}")
    print("  pair-space:", json.dumps(separation(fast_value), sort_keys=True))

    nvidia_path = Path(args.nvidia)
    if not nvidia_path.exists():
        print("NVIDIA: model file not found")
        print("Download:")
        print("  wget 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/reidentificationnet/deployable_v1.2/files/resnet50_market1501_aicity156.onnx' -O models/nvidia_reid/resnet50_market1501_aicity156.onnx")
        return
    n_value, n_time = nvidia(str(nvidia_path), images)
    print("NVIDIA ReIdentificationNet: resnet50_market1501_aicity156")
    print(f"  dim={n_value.shape[1]} seconds={n_time:.3f} images_per_sec={len(images)/max(n_time,1e-6):.1f}")
    print("  pair-space:", json.dumps(separation(n_value), sort_keys=True))
    print("NOTE: pair-space is unlabeled. True same/different-person accuracy requires a manually verified ground-truth pair file.")


if __name__ == "__main__":
    main()
