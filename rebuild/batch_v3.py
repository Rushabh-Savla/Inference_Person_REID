from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detector import PersonDetector  # noqa: E402
from reid.extractor import ReIDExtractor  # noqa: E402
from rebuild.identity_v2 import crop, illumination_variant, quality  # noqa: E402
from rebuild.identity_v3 import GlobalIdentityV3, Tracklet  # noqa: E402


class BatchPipelineV3:
    """V3 batch pipeline with a persistent identity gallery.

    The key architectural change is not a new threshold. Tracklets are treated
    as observations of persistent identities. Every new track searches the
    complete trusted gallery before a new identity is created, and confirmed
    observations enrich that gallery.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle) or {}
        self.out = Path(self.cfg["input"].get("output_dir", "rebuild_outputs"))
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out / "cache_v3"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.crops = self.cache / "crops"
        self.crops.mkdir(parents=True, exist_ok=True)

        det = self.cfg["detector"]
        reid = self.cfg["reid"]
        ident = self.cfg["identity_v3"]
        track = self.cfg["tracking"]
        dev = None if str(reid.get("device", "auto")).lower() == "auto" else reid.get("device")
        self.extractor = ReIDExtractor(
            weights=reid["weights"],
            device=dev,
            max_batch=int(reid.get("max_batch", 32)),
            model=reid.get("model"),
        )
        self.detector = det
        self.interval = max(1, int(reid.get("interval", 5)))
        self.part_interval = max(self.interval, int(reid.get("part_interval", 15)))
        self.min_quality = float(reid.get("min_quality", 0.20))
        self.light = bool(reid.get("illumination_variant", True))
        self.novelty = float(ident.get("novelty", 0.985))
        self.bank = int(ident.get("track_bank", 24))
        self.crop_bank = int(ident.get("crop_bank", 8))
        self.gap = float(track.get("fragment_gap_sec", 2.0))
        self.engine = GlobalIdentityV3(
            threshold=float(ident.get("match_threshold", 0.61)),
            margin=float(ident.get("match_margin", 0.035)),
            strong=float(ident.get("strong_threshold", 0.74)),
            support=int(ident.get("support", 2)),
            gallery=int(ident.get("gallery", 24)),
            novelty=self.novelty,
            promote=float(ident.get("promote_quality", 0.70)),
            new_count=int(ident.get("new_count", 3)),
        )
        self.tracks: Dict[str, Tracklet] = {}
        self.meta: Dict[str, dict] = {}
        self.saved: Dict[str, int] = {}

    def sources(self, values: List[str]) -> List[Tuple[str, str]]:
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
    def parts(image: np.ndarray) -> Dict[str, np.ndarray]:
        h, w = image.shape[:2]
        if h < 40 or w < 20:
            return {}
        # Regions are secondary evidence. Full-body remains the primary feature.
        upper = image[: max(1, int(h * 0.68))]
        lower = image[int(h * 0.32):]
        return {"upper": upper, "lower": lower}

    def store(self, key: str, kind: str, image: np.ndarray, meta: dict) -> None:
        if not image.size:
            return
        count = self.saved.get(key, 0)
        if count >= self.crop_bank:
            return
        safe = key.replace(":", "_")
        folder = self.crops / safe
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{count:02d}_{kind}.jpg"
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        meta["crop_path"] = str(path)
        self.saved[key] = count + 1

    def add(self, key: str, camera: str, track_id: int, segment: int, bbox: tuple, stamp: float,
            score: float, image: np.ndarray, feats: Dict[str, np.ndarray]) -> None:
        track = self.tracks.get(key)
        if track is None:
            track = Tracklet(camera, track_id, segment, self.meta[camera]["fps"])
            self.tracks[key] = track
        base = {"camera": camera, "frame": int(stamp * self.meta[camera]["fps"]),
                "timestamp": stamp, "track_id": track_id, "bbox": list(bbox),
                "detection_score": float(score), "quality": 0.0}
        for kind, vector in feats.items():
            value = base.copy()
            value["quality"] = float(score)
            value["kind"] = kind
            if track.add(vector, kind, float(score), value, self.novelty, self.bank):
                self.store(key, kind, image, value)
        if track.observations:
            boxes = np.asarray([x["bbox"] for x in track.observations], dtype=np.float32)
            hh = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
            ww = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
            track.shape = float(np.median(hh / ww))
            track.start = min(x["timestamp"] for x in track.observations)
            track.end = max(x["timestamp"] for x in track.observations)

    def collect(self, camera: str, path: str) -> None:
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
                detections = detector.track(image)
                for item in detections:
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
                        "camera": camera, "frame": frame, "timestamp": frame / fps,
                        "track_id": tid, "segment": seg, "tracklet_key": key,
                        "bbox": list(box), "detection_score": float(item.confidence),
                    }) + "\n")
                    if frame - last.get(key, -10**9) < self.interval:
                        continue
                    person = crop(image, box)
                    q = quality(person) if person is not None else 0.0
                    if person is None or q < self.min_quality:
                        continue
                    crops = [person]
                    names = ["full"]
                    if self.light and frame - last.get(key + ":light", -10**9) >= self.part_interval:
                        crops.append(illumination_variant(person))
                        names.append("light")
                    if frame - last.get(key + ":part", -10**9) >= self.part_interval:
                        for name, part in self.parts(person).items():
                            crops.append(part)
                            names.append(name)
                        last[key + ":part"] = frame
                    last[key] = frame
                    features = self.extractor.extract_batch(crops)
                    pack = {name: value for name, value in zip(names, features)}
                    self.add(key, camera, tid, seg, box, frame / fps, float(item.confidence), person, pack)
                    samples += 1
        finally:
            cap.release()
            rows.close()
        print(f"[v3] {camera}: frames={frame} tracklets={sum(1 for k in self.tracks if k.startswith(camera + ':'))} sampled={samples} total={total}")

    def save_cache(self) -> None:
        info = []
        packed = {}
        for key, track in sorted(self.tracks.items()):
            rows = []
            for feature in track.features:
                rows.append({"kind": feature.kind, "quality": feature.quality, "camera": feature.camera,
                             "timestamp": feature.stamp, "meta": feature.meta})
            info.append({"key": key, "camera": track.camera, "track_id": track.track_id,
                         "segment": track.segment, "start": track.start, "end": track.end,
                         "shape": track.shape, "count": track.count(), "features": rows})
            if track.features:
                packed[key] = np.stack([x.vector for x in track.features]).astype(np.float32)
        (self.cache / "tracklets_v3.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        np.savez_compressed(self.cache / "tracklets_v3.npz", **packed)
        (self.cache / "video_meta_v3.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def save_debug(self, decisions) -> None:
        with (self.out / "identity_debug_v3.jsonl").open("w", encoding="utf-8") as handle:
            for item in decisions:
                handle.write(json.dumps(item.__dict__) + "\n")

    def save_gallery(self, mapping: Dict[str, str]) -> None:
        gallery = self.engine.gallery_map()
        np.savez_compressed(self.out / "global_gallery_v3.npz", **gallery)
        meta = {}
        for gid, identity in self.engine.identities.items():
            meta[gid] = {"tracks": identity.tracks, "cameras": sorted(identity.cameras),
                         "trusted": len(identity.trusted), "candidate": len(identity.candidate)}
        (self.out / "global_gallery_v3.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def render(self, mapping: Dict[str, str]) -> None:
        for camera, meta in self.meta.items():
            cap = cv2.VideoCapture(meta["source"])
            out = self.out / f"{camera}_v3.mp4"
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"]))
            rows: Dict[int, list] = {}
            with (self.cache / f"{camera}.detections.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    rows.setdefault(int(item["frame"]), []).append(item)
            frame = 0
            try:
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    frame += 1
                    for item in rows.get(frame, []):
                        gid = mapping.get(item["tracklet_key"], "UNKNOWN")
                        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, gid, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, camera, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    writer.write(image)
            finally:
                cap.release()
                writer.release()
            print(f"[v3] wrote {out}")

    def run(self, values: List[str]) -> Dict[str, str]:
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")
        print(f"[v3] ReID: {self.extractor.describe()}")
        print(f"[v3] cameras: {len(sources)}")
        print("[v3] pass 1: detect + track + dynamic multi-part feature collection")
        for camera, path in sources:
            self.collect(camera, path)
        self.save_cache()
        print("[v3] pass 2: persistent global identity search")
        mapping, decisions = self.engine.run(self.tracks)
        self.save_debug(decisions)
        self.save_gallery(mapping)
        self.print_summary(mapping)
        print("[v3] pass 3: render from saved detections")
        self.render(mapping)
        return mapping

    def print_summary(self, mapping: Dict[str, str]) -> None:
        data = self.engine.summary(self.tracks)
        print("\n===== V3 REID RESULT =====")
        print(f"tracklets: {data['tracklets']}")
        print(f"global IDs: {data['global_ids']}")
        print(f"multi-camera IDs: {data['multi_camera_count']}")
        print(f"reasons: {json.dumps(data['reasons'], sort_keys=True)}")
        print(f"trusted gallery features: {data['gallery_features']}")
        print(f"candidate gallery features: {data['candidate_features']}")
        for gid, cams in sorted(data["multi_camera"].items()):
            print(f"  {gid}: {', '.join(cams)}")
        print(f"outputs: {self.out}")
