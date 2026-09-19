from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from rebuild.deepstream_runtime import DeepStreamRuntime


def command(value):
    try:
        result = subprocess.run(
            value,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=20,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="")
    args = parser.parse_args()

    info = DeepStreamRuntime.diagnostics()
    print("[verify] GPU:", info["gpu"])
    print("[verify] CUDA:", info["cuda"])
    print("[verify] TensorRT:", info["tensorrt"])
    print("[verify] cuDNN:", command(["python3", "-c", "import torch; print(torch.backends.cudnn.version())"]))

    root, library, config = DeepStreamRuntime.require()
    print("[verify] DeepStream root:", root)
    print("[verify] DeepStream version:", DeepStreamRuntime.version(root, library))
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

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see CUDA")
    cudnn = torch.backends.cudnn.version()
    if cudnn != 92000:
        raise RuntimeError(f"PyTorch cuDNN mismatch: expected 92000, found {cudnn}")
    print("[verify] PyTorch CUDA: OK")
    print("[verify] PyTorch cuDNN:", cudnn)

    import yaml
    cfg = yaml.safe_load((ROOT / "rebuild/config_state_invariant.yaml").read_text())

    from reid.nvidia_reid import NVIDIAReIDExtractor
    from reid.nvidia_swin import NVIDIASwinReIDExtractor
    from reid.solider_reid import SOLIDERReIDExtractor

    probe = np.zeros((384, 128, 3), dtype=np.uint8)
    resnet = NVIDIAReIDExtractor(cfg["reid"]["weights"], device="cuda", max_batch=1)
    swin = NVIDIASwinReIDExtractor(cfg["cross_camera_models"]["swin_weights"], device="cuda", max_batch=1)
    solider = SOLIDERReIDExtractor(cfg["cross_camera_models"]["solider_weights"], device="cuda", max_batch=1)
    assert resnet.extract_batch([probe]).shape[1] == 256
    assert swin.extract_batch([probe]).shape[1] == 1024
    assert solider.extract_batch([probe]).shape[1] == 1024
    print("[verify] Person ReID ResNet/Swin/SOLIDER inference: OK")

    from insightface.app import FaceAnalysis
    face = FaceAnalysis(
        name=cfg["face"]["model"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face.prepare(ctx_id=0, det_size=tuple(cfg["face"]["det_size"]))
    if args.video:
        cap = cv2.VideoCapture(args.video)
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read first frame from {args.video}")
        face.get(image)
        face_probe = image
    else:
        face_probe = probe
        face.get(face_probe)
    print("[verify] FaceAnalysis + SCRFD/ArcFace inference: OK")

    from ultralytics import YOLO
    pose = YOLO(str(cfg["pose"]["model"]))
    result = pose(face_probe, classes=[0], conf=0.20, verbose=False)
    if not result:
        raise RuntimeError("Pose inference returned no result object")
    print("[verify] Pose inference: OK")

    from rebuild.person_attributes import pack
    attrs = pack(face_probe, face_probe, [0, 0, face_probe.shape[1], face_probe.shape[0]])
    if attrs.shape != (112,):
        raise RuntimeError(f"Clothing feature dimension mismatch: {attrs.shape}")
    if not np.isfinite(attrs[:40]).all() or float(np.linalg.norm(attrs[:20])) <= 0.0 or float(np.linalg.norm(attrs[20:40])) <= 0.0:
        raise RuntimeError("Top/bottom clothing descriptors are not populated")
    print("[verify] Top + bottom clothing extraction: OK")

    from src.live.qdrant_gallery import QdrantGallery
    qurl = os.environ.get("QDRANT_URL")
    if not qurl:
        raise RuntimeError("QDRANT_URL is required")
    prefix = "runtime_verify"
    q = QdrantGallery(prefix=prefix, limit=8, url=qurl)
    probevec = np.ones(256, np.float32)
    q._upsert(987654, "resnet", "full", [probevec])
    hits = q._query("resnet", probevec)
    if not hits or int((hits[0].payload or {}).get("gid", -1)) != 987654:
        raise RuntimeError("Qdrant remote write/retrieval failed")
    for collection in q.collections.values():
        try:
            q.client.delete_collection(collection)
        except Exception:
            pass
    print("[verify] Qdrant remote write + retrieval: OK")

    if args.video:
        from rebuild.nvdcf_tracker import NvDCF
        detector = NvDCF(cfg["detector"])
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as handle:
            target = Path(handle.name)
            fps, width, height = detector.track("verify", args.video, target)
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("NvDCF returned invalid video metadata")
        rows = target.read_text(encoding="utf-8").strip().splitlines()
        if not rows:
            raise RuntimeError("NvDCF produced no tracker rows for the real video")
        print("[verify] Real-video NvDCF tracking smoke test: OK")

    print("[verify] STRICT RUNTIME: PASS")


if __name__ == "__main__":
    main()
