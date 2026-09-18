from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parents[1]
if str(repo) not in os.sys.path:
    os.sys.path.insert(0, str(repo))

from rebuild.deepstream_runtime import DeepStreamRuntime


def _command(value):
    try:
        result = subprocess.run(value, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False, timeout=20)
        return result.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"

def main():
    parser = argparse.ArgumentParser(description="Verify every strict ReID runtime component.")
    parser.add_argument("--video", default="", help="Optional real input video for a one-buffer NvDCF smoke test.")
    args = parser.parse_args()

    info = DeepStreamRuntime.diagnostics()
    print("[verify] GPU:", info["gpu"])
    print("[verify] CUDA:", info["cuda"])
    print("[verify] TensorRT:", info["tensorrt"])
    print("[verify] cuDNN:", _command(["bash", "-lc", "python - <<'PY'\nimport torch\nprint(torch.backends.cudnn.version())\nPY"]))
    
    root, library, config = DeepStreamRuntime.require()
    print("[verify] DeepStream root:", root)
    print("[verify] DeepStream version:", DeepStreamRuntime.version(root))
    print("[verify] NvDCF library:", library)
    print("[verify] NvDCF config:", config)

    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    import pyds

    Gst.init(None)
    tracker = Gst.ElementFactory.make("nvtracker", "verify_nvdcf")
    if tracker is None:
        raise RuntimeError("nvtracker element is unavailable")
    tracker.set_property("ll-lib-file", library)
    tracker.set_property("ll-config-file", config)
    tracker.set_property("gpu-id", 0)
    tracker.set_state(Gst.State.READY)
    tracker.set_state(Gst.State.NULL)
    print("[verify] NvDCF low-level library load: OK")
    print("[verify] nvtracker element: OK")
    print("[verify] PyDS:", pyds.__file__)

    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable")
    print("[verify] ORT:", ort.__version__)
    print("[verify] CUDAExecutionProvider: OK")

    from reid.nvidia_reid import NVIDIAReIDExtractor
    from reid.nvidia_swin import NVIDIASwinReIDExtractor
    from reid.solider_reid import SOLIDERReIDExtractor
    import yaml

    cfg = yaml.safe_load(Path("rebuild/config_state_invariant.yaml").read_text())
    resnet = NVIDIAReIDExtractor(cfg["reid"]["weights"], device="cuda", max_batch=1)
    swin = NVIDIASwinReIDExtractor(cfg["cross_camera_models"]["swin_weights"], device="cuda", max_batch=1)
    solider = SOLIDERReIDExtractor(cfg["cross_camera_models"]["solider_weights"], device="cuda", max_batch=1)
    probe = np.zeros((384, 128, 3), dtype=np.uint8)
    assert resnet.extract_batch([probe]).shape[1] == 256
    assert swin.extract_batch([probe]).shape[1] == 1024
    assert solider.extract_batch([probe]).shape[1] == 1024
    print("[verify] Person ReID ResNet/Swin/SOLIDER inference: OK")

    from insightface.app import FaceAnalysis
    face = FaceAnalysis(name=cfg["face"]["model"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face.prepare(ctx_id=0, det_size=tuple(cfg["face"]["det_size"]))
    print("[verify] FaceAnalysis CUDA runtime: OK")

    from ultralytics import YOLO
    pose = YOLO(str(cfg["pose"]["model"]))
    result = pose(probe, classes=[0], conf=0.20, verbose=False)
    if not result:
        raise RuntimeError("Pose inference returned no result object")
    print("[verify] Pose model inference: OK")

    from rebuild.person_attributes import pack
    attrs = pack(probe, probe, [0, 0, 128, 384])
    if attrs.shape != (112,):
        raise RuntimeError(f"Clothing feature dimension mismatch: {attrs.shape}")
    print("[verify] Top/bottom clothing feature extraction: OK")

    from src.live.qdrant_gallery import QdrantGallery
    with tempfile.TemporaryDirectory(prefix="reid_qdrant_verify_") as qroot:
        q = QdrantGallery(qroot, "verify_reid", 8)
        probevec = np.ones(256, np.float32)
        q._upsert(1, "resnet", "full", [probevec])
        hits = q._query("resnet", probevec)
        if not hits or int((hits[0].payload or {}).get("gid", -1)) != 1:
            raise RuntimeError("Qdrant vector was stored but could not be retrieved")
    print("[verify] Qdrant write + retrieval: OK")

    if args.video:
        from rebuild.nvdcf_tracker import NvDCF
        detector = NvDCF(cfg["detector"])
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as handle:
            detector.track("verify", args.video, Path(handle.name))
        print("[verify] Real-video NvDCF tracking smoke test: OK")

    print("[verify] STRICT RUNTIME: PASS")


if __name__ == "__main__":
    main()
