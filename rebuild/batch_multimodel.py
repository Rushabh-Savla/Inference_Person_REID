from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detector import PersonDetector  # noqa: E402
from rebuild.batch_v6 import BatchPipelineV6  # noqa: E402
from rebuild.identity_v2 import crop, illumination_variant, quality  # noqa: E402
from rebuild.multimodel_reid import MultiModelLocalGlobalResolver  # noqa: E402
from rebuild.v6_local_global import LocalNode  # noqa: E402
from reid.nvidia_swin import NVIDIASwinReIDExtractor  # noqa: E402
from reid.solider_reid import SOLIDERReIDExtractor  # noqa: E402


class BatchPipelineMultiModel(BatchPipelineV6):
    """Final cross-camera experiment: proven V6 + NVIDIA Swin + SOLIDER.

    Same-camera identity remains the protected V6 ResNet solution. Cross-camera
    identity is solved independently using three embedding spaces plus restrained
    clothing/shape/time evidence and a one-to-one camera matching constraint.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        models = self.cfg["cross_camera_models"]
        self.swin = NVIDIASwinReIDExtractor(
            models["swin_weights"], device="cuda", max_batch=int(models.get("swin_batch", 16))
        )
        self.solider = SOLIDERReIDExtractor(
            models["solider_weights"], device="cuda", max_batch=int(models.get("solider_batch", 16))
        )
        self.extra = {"swin": {}, "solider": {}}

    @staticmethod
    def colour_signature(image: np.ndarray) -> np.ndarray | None:
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        if h < 40 or w < 20:
            return None
        x1, x2 = int(0.16 * w), int(0.84 * w)
        y1, y2 = int(0.10 * h), int(0.72 * h)
        torso = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1].astype(np.float32) / 255.0
        val = hsv[..., 2].astype(np.float32) / 255.0
        hue = hsv[..., 0].astype(np.float32) / 180.0
        hue_hist, _ = np.histogram(hue, bins=8, range=(0.0, 1.0), weights=sat + 0.05)
        neutral = sat < 0.28
        value_hist, _ = np.histogram(val[neutral], bins=4, range=(0.0, 1.0))
        desc = np.concatenate([hue_hist, value_hist]).astype(np.float32)
        return desc / (np.linalg.norm(desc) + 1e-12)

    def add_body(self, key, camera, track_id, segment, bbox, stamp, score, image, feats, extra):
        super().add_body(key, camera, track_id, segment, bbox, stamp, score, image, feats)
        track = self.tracks[key]
        bank = getattr(track, "model_bank", None)
        if bank is None:
            bank = {"swin": [], "solider": []}
            track.model_bank = bank
        for model_name, values in extra.items():
            bank[model_name].extend([np.asarray(v, np.float32) for v in values])
            bank[model_name] = bank[model_name][-24:]
        signature = self.colour_signature(image)
        if signature is not None:
            old = getattr(track, "colour_signature", None)
            if old is None:
                track.colour_signature = signature
            else:
                mixed = 0.80 * np.asarray(old, np.float32) + 0.20 * signature
                track.colour_signature = mixed / (np.linalg.norm(mixed) + 1e-12)

    def collect(self, camera: str, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.meta[camera] = {"source": path, "fps": fps, "width": width, "height": height, "frames": total}
        detector_cfg = self.detector
        detector = PersonDetector(
            model_path=detector_cfg["model"], confidence_threshold=float(detector_cfg["conf"]),
            person_class_id=0, tracker_config=detector_cfg["tracker"], pose_ensemble=None, iou=float(detector_cfg["iou"]),
        )
        rows = (self.cache / f"{camera}.detections.jsonl").open("w", encoding="utf-8")
        last_body: Dict[str, int] = {}; segments: Dict[int, int] = {}; seen: Dict[int, int] = {}
        frame = 0; samples = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                frame += 1
                for item in detector.track(image):
                    if item.track_id is None:
                        continue
                    tid = int(item.track_id); prev = seen.get(tid)
                    if prev is None or frame - prev > int(self.gap * fps):
                        segments[tid] = segments.get(tid, 0) + 1
                    seg = segments[tid]; key = f"{camera}:{tid}:{seg}"; seen[tid] = frame
                    box = (item.x1, item.y1, item.x2, item.y2)
                    rows.write(json.dumps({"camera": camera, "frame": frame, "timestamp": frame / fps, "track_id": tid, "segment": seg, "tracklet_key": key, "bbox": list(box), "detection_score": float(item.confidence)}) + "\n")
                    if frame - last_body.get(key, -10**9) < self.interval:
                        continue
                    person = crop(image, box); q = quality(person) if person is not None else 0.0
                    if person is None or q < self.min_quality:
                        continue
                    crops = [person]; names = ["full"]
                    if self.light and frame - last_body.get(key + ":light", -10**9) >= self.part_interval:
                        crops.append(illumination_variant(person)); names.append("light"); last_body[key + ":light"] = frame
                    if frame - last_body.get(key + ":part", -10**9) >= self.part_interval:
                        for name, part in self.parts(person).items():
                            crops.append(part); names.append(name)
                        last_body[key + ":part"] = frame
                    last_body[key] = frame

                    resnet_values = self.extractor.extract_batch(crops)
                    extra_crops = [crops[i] for i, n in enumerate(names) if n in {"full", "light"}]
                    swin_values = self.swin.extract_batch(extra_crops)
                    solider_values = self.solider.extract_batch(extra_crops)
                    extras = {"swin": swin_values, "solider": solider_values}
                    feat_map = {n: v for n, v in zip(names, resnet_values)}
                    self.add_body(key, camera, tid, seg, box, frame / fps, float(item.confidence), person, feat_map, extras)
                    samples += 1
        finally:
            cap.release(); rows.close()
        print(f"[final] {camera}: frames={frame} tracklets={sum(1 for k in self.tracks if k.startswith(camera + ':'))} multimodel_samples={samples} total={total}")

    def run(self, values):
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")
        cameras = [x[0] for x in sources]
        self.out.mkdir(parents=True, exist_ok=True)
        print(f"[final] ResNet: {self.extractor.describe()}")
        print(f"[final] Swin:   {self.swin.describe()}")
        print(f"[final] SOLIDER:{self.solider.describe()}")
        print("[final] FACE: OFF")
        print(f"[final] cameras: {', '.join(cameras)}")
        print("[final] pass 1: detect + track + three independent body embeddings")
        for camera, path in sources:
            self.collect(camera, path)
        self.save_cache()
        print("[final] pass 2: independent per-camera protected V6 identity solving")
        local_mapping, _ = self.local_assign(self.tracks, cameras)
        print("[final] pass 3: three-model cross-camera one-to-one reconciliation")
        resolver = MultiModelLocalGlobalResolver(self.cfg["identity_v6"])
        global_mapping, components, edges = resolver.resolve(local_mapping, self.tracks, cameras)
        payload = {"local_mapping": local_mapping, "global_mapping": global_mapping, "components": components, "edges": edges}
        (self.out / "final_multimodel_debug.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[final] accepted cross-camera links: {len(edges)}")
        print(f"[final] final global IDs: {len(set(global_mapping.values()))}")
        print("[final] pass 4: render")
        self.render(global_mapping)
        multi = {gid: sorted({k.split("::", 1)[0] for k in members}) for gid, members in components.items() if len({k.split("::", 1)[0] for k in members}) > 1}
        print("MULTI-CAMERA IDS:")
        for gid, cams in sorted(multi.items()):
            print(f"  {gid}: {', '.join(cams)}")
        print(f"outputs: {self.out}")
        return global_mapping
