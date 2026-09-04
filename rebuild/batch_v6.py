from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detector import PersonDetector  # noqa: E402
from rebuild.identity_body_v6 import GlobalIdentityBodyV6  # noqa: E402
from rebuild.identity_v2 import crop, illumination_variant, quality  # noqa: E402
from rebuild.identity_v3 import Feature, Tracklet  # noqa: E402
from reid.nvidia_reid import NVIDIAReIDExtractor  # noqa: E402


class BatchPipelineV6:
    """V6 body-only pipeline: NVIDIA embeddings + persistent global gallery.

    Face recognition is deliberately absent from this controlled experiment.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle) or {}
        self.out = Path(self.cfg["input"].get("output_dir", "rebuild_outputs"))
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out / "cache_v6"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.crops = self.cache / "crops"
        self.crops.mkdir(parents=True, exist_ok=True)

        det = self.cfg["detector"]
        reid = self.cfg["reid"]
        self.extractor = NVIDIAReIDExtractor(
            weights=reid["weights"],
            device=str(reid.get("device", "cuda")),
            max_batch=int(reid.get("max_batch", 32)),
        )
        self.detector = det
        self.interval = max(1, int(reid.get("interval", 5)))
        self.part_interval = max(self.interval, int(reid.get("part_interval", 15)))
        self.min_quality = float(reid.get("min_quality", 0.20))
        self.light = bool(reid.get("illumination_variant", True))
        self.novelty = float(self.cfg["identity_v6"].get("novelty", 0.985))
        self.bank = int(self.cfg["identity_v6"].get("gallery", 32))
        self.crop_bank = 8
        self.gap = float(self.cfg.get("tracking", {}).get("fragment_gap_sec", 2.0))
        self.engine = GlobalIdentityBodyV6(self.cfg["identity_v6"])
        self.tracks: Dict[str, Tracklet] = {}
        self.meta: Dict[str, dict] = {}
        self.saved: Dict[str, int] = {}

    def sources(self, values: List[str]):
        if values:
            return [(Path(v).stem, v) for v in values]
        out = []
        for item in self.cfg.get("input", {}).get("videos", []):
            if isinstance(item, dict):
                out.append((item.get("name") or Path(item["path"]).stem, item["path"]))
            else:
                out.append((Path(item).stem, item))
        return out

    @staticmethod
    def parts(image: np.ndarray):
        h, w = image.shape[:2]
        if h < 40 or w < 20:
            return {}
        return {"upper": image[:max(1, int(h * 0.68))], "lower": image[int(h * 0.32):]}

    def store_crop(self, key: str, kind: str, image: np.ndarray) -> None:
        count = self.saved.get(key, 0)
        if count >= self.crop_bank or image is None or image.size == 0:
            return
        folder = self.crops / key.replace(":", "_")
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{count:02d}_{kind}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        self.saved[key] = count + 1

    def add_body(self, key, camera, track_id, segment, bbox, stamp, score, image, feats):
        track = self.tracks.get(key)
        if track is None:
            track = Tracklet(camera, track_id, segment, self.meta[camera]["fps"])
            self.tracks[key] = track
        base = {
            "camera": camera,
            "frame": int(stamp * self.meta[camera]["fps"]),
            "timestamp": stamp,
            "track_id": track_id,
            "bbox": list(bbox),
            "detection_score": float(score),
            "quality": float(score),
        }
        for kind, vector in feats.items():
            value = base.copy(); value["kind"] = kind
            if track.add(vector, kind, float(score), value, self.novelty, self.bank):
                self.store_crop(key, kind, image)
        if track.observations:
            boxes = np.asarray([x["bbox"] for x in track.observations], dtype=np.float32)
            hh = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
            ww = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
            track.shape = float(np.median(hh / ww))
            track.start = min(x["timestamp"] for x in track.observations)
            track.end = max(x["timestamp"] for x in track.observations)

    def collect(self, camera: str, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.meta[camera] = {"source": path, "fps": fps, "width": width, "height": height, "frames": total}
        detector = PersonDetector(
            model_path=self.detector["model"], confidence_threshold=float(self.detector["conf"]),
            person_class_id=0, tracker_config=self.detector["tracker"], pose_ensemble=None, iou=float(self.detector["iou"]),
        )
        rows = (self.cache / f"{camera}.detections.jsonl").open("w", encoding="utf-8")
        last_body: Dict[str, int] = {}; segments: Dict[int, int] = {}; seen: Dict[int, int] = {}
        frame = 0; samples = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok: break
                frame += 1
                for item in detector.track(image):
                    if item.track_id is None: continue
                    tid = int(item.track_id); prev = seen.get(tid)
                    if prev is None or frame - prev > int(self.gap * fps): segments[tid] = segments.get(tid, 0) + 1
                    seg = segments[tid]; key = f"{camera}:{tid}:{seg}"; seen[tid] = frame
                    box = (item.x1, item.y1, item.x2, item.y2)
                    rows.write(json.dumps({"camera": camera, "frame": frame, "timestamp": frame / fps, "track_id": tid, "segment": seg, "tracklet_key": key, "bbox": list(box), "detection_score": float(item.confidence)}) + "\n")
                    if frame - last_body.get(key, -10**9) < self.interval: continue
                    person = crop(image, box); q = quality(person) if person is not None else 0.0
                    if person is None or q < self.min_quality: continue
                    crops = [person]; names = ["full"]
                    if self.light and frame - last_body.get(key + ":light", -10**9) >= self.part_interval:
                        crops.append(illumination_variant(person)); names.append("light"); last_body[key + ":light"] = frame
                    if frame - last_body.get(key + ":part", -10**9) >= self.part_interval:
                        for name, part in self.parts(person).items(): crops.append(part); names.append(name)
                        last_body[key + ":part"] = frame
                    last_body[key] = frame
                    features = self.extractor.extract_batch(crops)
                    self.add_body(key, camera, tid, seg, box, frame / fps, float(item.confidence), person, {n: v for n, v in zip(names, features)})
                    samples += 1
        finally:
            cap.release(); rows.close()
        print(f"[v6] {camera}: frames={frame} tracklets={sum(1 for k in self.tracks if k.startswith(camera + ':'))} body_samples={samples} total={total}")

    def save_cache(self):
        info = []; packed = {}
        for key, track in sorted(self.tracks.items()):
            features = [{"kind": x.kind, "quality": x.quality, "camera": x.camera, "timestamp": x.stamp, "meta": x.meta} for x in track.features]
            info.append({"key": key, "camera": track.camera, "track_id": track.track_id, "segment": track.segment, "start": track.start, "end": track.end, "shape": track.shape, "count": track.count(), "features": features})
            if track.features: packed[key] = np.stack([x.vector for x in track.features]).astype(np.float32)
        (self.cache / "tracklets_v6.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        np.savez_compressed(self.cache / "tracklets_v6.npz", **packed)
        (self.cache / "video_meta_v6.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def save_debug(self, decisions):
        with (self.out / "identity_debug_v6.jsonl").open("w", encoding="utf-8") as handle:
            for item in decisions: handle.write(json.dumps(item.__dict__) + "\n")

    def save_gallery(self):
        body = {}; meta = {}
        for gid, identity in self.engine.identities.items():
            if identity.trusted: body[gid] = np.stack([x.vector for x in identity.trusted]).astype("float32")
            meta[gid] = {"tracks": identity.tracks, "cameras": sorted(identity.cameras), "trusted_body": len(identity.trusted), "candidate_body": len(identity.candidate)}
        np.savez_compressed(self.out / "global_body_gallery_v6.npz", **body)
        (self.out / "global_gallery_v6.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def render(self, mapping):
        for camera, meta in self.meta.items():
            cap = cv2.VideoCapture(meta["source"]); out = self.out / f"{camera}_v6.mp4"
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"]))
            rows = {}
            with (self.cache / f"{camera}.detections.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line); rows.setdefault(int(item["frame"]), []).append(item)
            frame = 0
            try:
                while True:
                    ok, image = cap.read()
                    if not ok: break
                    frame += 1
                    for item in rows.get(frame, []):
                        gid = mapping.get(item["tracklet_key"], "UNKNOWN"); x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, gid, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, camera, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    writer.write(image)
            finally:
                cap.release(); writer.release()
            print(f"[v6] wrote {out}")

    def print_summary(self):
        data = self.engine.summary(self.tracks)
        print("\n===== V6 IDENTITY RESULT =====")
        for name in ("tracklets","global_ids","new_identities","reidentified","same_camera_reassociations","recent_lost_track_reassociations","cross_camera_reidentifications","identity_merges","provisional_identities","fragmented_identity_count","face_assisted","body_assisted","temporal_assisted"):
            print(f"{name}: {data.get(name, 0)}")
        print(f"reasons: {json.dumps(data['reasons'], sort_keys=True)}")
        if data["multi_camera"]:
            print("MULTI-CAMERA IDS:")
            for gid, cams in sorted(data["multi_camera"].items()): print(f"  {gid}: {', '.join(cams)}")
        if data["fragmented_identities"]:
            print("IDENTITY TRACK GROUPS:")
            for gid, tracks in sorted(data["fragmented_identities"].items()): print(f"  {gid}: {', '.join(tracks)}")
        print(f"outputs: {self.out}")

    def run(self, values):
        sources = self.sources(values)
        if not sources: raise SystemExit("No videos supplied")
        print(f"[v6] {self.extractor.describe()}")
        print("[v6] FACE: OFF (controlled body-only experiment)")
        print(f"[v6] cameras: {len(sources)}")
        print("[v6] pass 1: detect + track + NVIDIA body embeddings")
        for camera, path in sources: self.collect(camera, path)
        self.save_cache()
        print("[v6] pass 2: persistent global-gallery association + same-camera reassociation")
        mapping, decisions = self.engine.run(self.tracks)
        self.save_debug(decisions); self.save_gallery(); self.print_summary()
        print("[v6] pass 3: render from saved detections")
        self.render(mapping)
        return mapping
