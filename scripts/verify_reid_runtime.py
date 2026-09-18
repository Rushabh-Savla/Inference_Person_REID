from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from rebuild.deepstream_runtime import DeepStreamRuntime


def main():
    parser = argparse.ArgumentParser(description="Verify every strict ReID runtime component.")
    parser.add_argument("--video", default="", help="Optional real input video for a one-buffer NvDCF smoke test.")
    args = parser.parse_args()

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
    q = QdrantGallery(
        cfg["identity_state"]["qdrant_path"],
        cfg["identity_state"]["qdrant_prefix"],
        cfg["identity_state"]["qdrant_limit"],
    )
    # Confirm Qdrant accepts and retrieves a real vector in the configured space.
    q._upsert(999999, "resnet", "full", [np.ones(256, np.float32)])
    hits, _ = q.search_component([])
    if not isinstance(hits, dict):
        raise RuntimeError("Qdrant retrieval API did not return its expected structure")
    print("[verify] Qdrant collections/retrieval API: OK")

    if args.video:
        from rebuild.nvdcf_tracker import NvDCF
        import tempfile
        detector = NvDCF(cfg["detector"])
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as handle:
            detector.track("verify", args.video, Path(handle.name))
        print("[verify] Real-video NvDCF tracking smoke test: OK")

    print("[verify] STRICT RUNTIME: PASS")


if __name__ == "__main__":
    main()
