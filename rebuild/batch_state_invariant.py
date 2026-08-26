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
from rebuild.multimodel_state_invariant import StateInvariantResolver  # noqa: E402
from rebuild.identity_body_v6 import GlobalIdentityBodyV6  # noqa: E402
from rebuild.identity_v2 import crop, illumination_variant, quality  # noqa: E402
from live.persistent_multimodel import PersistentMultimodelRegistry  # noqa: E402
from reid.nvidia_swin import NVIDIASwinReIDExtractor  # noqa: E402
from reid.solider_reid import SOLIDERReIDExtractor  # noqa: E402


class BatchPipelineStateInvariant(BatchPipelineV6):
    """Final state-invariant MTMC inference pipeline.

    Same-camera association remains the proven NVIDIA V6 body matcher. Every
    accepted observation is additionally represented by ResNet, NVIDIA Swin and
    SOLIDER features over several body views, allowing a seated person to match a
    standing observation without relying on lower-body visibility.
    """

    VARIANTS = ("full", "light", "upper", "torso", "lower")
    MULTIMODEL_VARIANTS = ("full", "light", "upper", "torso", "lower")

    def __init__(self, config_path: str):
        super().__init__(config_path)
        models = self.cfg["cross_camera_models"]
        self.swin = NVIDIASwinReIDExtractor(
            models["swin_weights"], device="cuda", max_batch=int(models.get("swin_batch", 16))
        )
        self.solider = SOLIDERReIDExtractor(
            models["solider_weights"], device="cuda", max_batch=int(models.get("solider_batch", 16))
        )
        state = self.cfg.get("identity_state", {})
        self.registry = PersistentMultimodelRegistry(
            state.get("path", "identity_state/reid_multimodel.sqlite3"),
            model_id=str(state.get("model_id", "final-state-invariant-v1")),
            bank_size=int(state.get("bank_size", 48)),
        )

    @staticmethod
    def parts(image: np.ndarray) -> Dict[str, np.ndarray]:
        h, w = image.shape[:2]
        if h < 40 or w < 20:
            return {}
        return {
            "upper": image[: max(1, int(h * 0.68))],
            "torso": image[int(h * 0.08): max(int(h * 0.74), int(h * 0.08) + 1)],
            "lower": image[int(h * 0.32):],
        }

    @staticmethod
    def colour_signature(image: np.ndarray) -> np.ndarray | None:
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        if h < 40 or w < 20:
            return None
        # Clothing descriptor is deliberately torso/upper weighted and invariant
        # to the visibility of legs when a person is seated.
        x1, x2 = int(0.16 * w), int(0.84 * w)
        y1, y2 = int(0.08 * h), int(0.72 * h)
        torso = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1].astype(np.float32) / 255.0
        val = hsv[..., 2].astype(np.float32) / 255.0
        hue = hsv[..., 0].astype(np.float32) / 180.0
        hue_hist, _ = np.histogram(hue, bins=12, range=(0.0, 1.0), weights=sat + 0.05)
        neutral = sat < 0.28
        value_hist, _ = np.histogram(val[neutral], bins=5, range=(0.0, 1.0))
        desc = np.concatenate([hue_hist, value_hist]).astype(np.float32)
        return desc / (np.linalg.norm(desc) + 1e-12)

    def _init_state_bank(self, track) -> None:
        if not hasattr(track, "state_bank"):
            track.state_bank = {
                "resnet": {kind: [] for kind in self.VARIANTS},
                "swin": {kind: [] for kind in self.VARIANTS},
                "solider": {kind: [] for kind in self.VARIANTS},
            }
        if not hasattr(track, "colour_bank"):
            track.colour_bank = []

    def add_body(self, key, camera, track_id, segment, bbox, stamp, score, image, feats, multi):
        super().add_body(key, camera, track_id, segment, bbox, stamp, score, image, feats)
        track = self.tracks[key]
        self._init_state_bank(track)
        for kind, vector in feats.items():
            if kind in track.state_bank["resnet"]:
                track.state_bank["resnet"][kind].append(np.asarray(vector, np.float32))
                track.state_bank["resnet"][kind] = track.state_bank["resnet"][kind][-48:]
        for model in ("swin", "solider"):
            values = multi.get(model, {})
            for kind, vectors in values.items():
                if kind not in track.state_bank[model]:
                    track.state_bank[model][kind] = []
                track.state_bank[model][kind].extend(np.asarray(v, np.float32) for v in vectors)
                track.state_bank[model][kind] = track.state_bank[model][kind][-48:]
        signature = self.colour_signature(image)
        if signature is not None:
            track.colour_bank.append(signature)
            track.colour_bank = track.colour_bank[-24:]
            mixed = np.mean(np.stack(track.colour_bank), axis=0)
            track.colour_signature = mixed / (np.linalg.norm(mixed) + 1e-12)

    def collect(self, camera: str, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.meta[camera] = {"source": path, "fps": fps, "width": width, "height": height, "frames": total}
        detector = PersonDetector(
            model_path=self.detector["model"],
            confidence_threshold=float(self.detector["conf"]),
            person_class_id=0,
            tracker_config=self.detector["tracker"],
            pose_ensemble=None,
            iou=float(self.detector["iou"]),
        )
        rows = (self.cache / f"{camera}.detections.jsonl").open("w", encoding="utf-8")
        last: Dict[str, int] = {}
        segments: Dict[int, int] = {}
        seen: Dict[int, int] = {}
        frame = 0
        samples = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                frame += 1
                for item in detector.track(image):
                    if item.track_id is None:
                        continue
                    tid = int(item.track_id)
                    previous = seen.get(tid)
                    if previous is None or frame - previous > int(self.gap * fps):
                        segments[tid] = segments.get(tid, 0) + 1
                    seg = segments[tid]
                    key = f"{camera}:{tid}:{seg}"
                    seen[tid] = frame
                    box = (item.x1, item.y1, item.x2, item.y2)
                    rows.write(json.dumps({
                        "camera": camera,
                        "frame": frame,
                        "timestamp": frame / fps,
                        "track_id": tid,
                        "segment": seg,
                        "tracklet_key": key,
                        "bbox": list(box),
                        "detection_score": float(item.confidence),
                    }) + "\n")
                    if frame - last.get(key, -10**9) < self.interval:
                        continue
                    person = crop(image, box)
                    q = quality(person) if person is not None else 0.0
                    if person is None or q < self.min_quality:
                        continue

                    variants: Dict[str, np.ndarray] = {"full": person}
                    if self.light and frame - last.get(key + ":light", -10**9) >= self.part_interval:
                        variants["light"] = illumination_variant(person)
                        last[key + ":light"] = frame
                    if frame - last.get(key + ":parts", -10**9) >= self.part_interval:
                        variants.update(self.parts(person))
                        last[key + ":parts"] = frame
                    last[key] = frame

                    ordered_names = list(variants.keys())
                    crops = [variants[name] for name in ordered_names]
                    resnet = self.extractor.extract_batch(crops)
                    swin = self.swin.extract_batch(crops)
                    solider = self.solider.extract_batch(crops)

                    resnet_feats = {name: value for name, value in zip(ordered_names, resnet)}
                    state_multi = {
                        "swin": {name: [value] for name, value in zip(ordered_names, swin)},
                        "solider": {name: [value] for name, value in zip(ordered_names, solider)},
                    }
                    self.add_body(
                        key,
                        camera,
                        tid,
                        seg,
                        box,
                        frame / fps,
                        float(item.confidence),
                        person,
                        resnet_feats,
                        state_multi,
                    )
                    samples += 1
        finally:
            cap.release()
            rows.close()
        count = sum(1 for key in self.tracks if key.startswith(camera + ":"))
        print(f"[state-final] {camera}: frames={frame} tracklets={count} multiview_samples={samples} total={total}")

    def run(self, values):
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")
        cameras = [item[0] for item in sources]
        try:
            print(f"[state-final] ResNet: {self.extractor.describe()}")
            print(f"[state-final] Swin:   {self.swin.describe()}")
            print(f"[state-final] SOLIDER:{self.solider.describe()}")
            print("[state-final] FACE: OFF")
            print(f"[state-final] cameras: {', '.join(cameras)}")
            print("[state-final] pass 1: detect + track + full/upper/torso/lower multimodel embeddings")
            for camera, path in sources:
                self.collect(camera, path)
            self.save_cache()

            print("[state-final] pass 2: independent protected V6 identity solving per camera")
            local_mapping: Dict[str, str] = {}
            for camera in cameras:
                subset = {key: track for key, track in self.tracks.items() if track.camera == camera}
                local_engine = GlobalIdentityBodyV6(self.cfg["identity_v6"])
                mapping, _ = local_engine.run(subset)
                local_mapping.update(mapping)
                print(f"[state-final] local {camera}: tracklets={len(subset)} local_ids={len(set(mapping.values()))}")

            print("[state-final] pass 3: state-invariant cross-camera reconciliation")
            resolver_cfg = dict(self.cfg["identity_v6"])
            resolver = StateInvariantResolver(resolver_cfg, registry=self.registry)
            global_mapping, components, edges = resolver.resolve(local_mapping, self.tracks, cameras)
            debug = {
                "local_mapping": local_mapping,
                "global_mapping": global_mapping,
                "components": components,
                "edges": edges,
            }
            (self.out / "state_invariant_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
            print(f"[state-final] accepted cross-camera links: {len(edges)}")
            print(f"[state-final] final global IDs: {len(set(global_mapping.values()))}")
            print(f"[state-final] persistent global IDs: {self.registry.gids()}")

            print("[state-final] pass 4: render")
            self.render(global_mapping)
            multi = {
                gid: sorted({key.split(":", 1)[0] for key in members})
                for gid, members in components.items()
                if len({key.split(":", 1)[0] for key in members}) > 1
            }
            print("MULTI-CAMERA IDS:")
            for gid, cams in sorted(multi.items()):
                print(f"  {gid}: {', '.join(cams)}")
            print(f"outputs: {self.out}")
            return global_mapping
        finally:
            self.registry.close()
