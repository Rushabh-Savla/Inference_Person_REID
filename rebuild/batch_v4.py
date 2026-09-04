from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

from detector import PersonDetector
from reid.extractor import ReIDExtractor
from rebuild.identity_v2 import crop, illumination_variant, quality
from rebuild.identity_v3 import Feature, Tracklet
from rebuild.identity_v4 import GlobalIdentityV4
from rebuild.face_v4 import FaceExtractorV4


class BatchPipelineV4:
    """V4: V3 body pipeline plus persistent face evidence."""

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle) or {}
        self.out = Path(self.cfg["input"].get("output_dir", "rebuild_outputs"))
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out / "cache_v4"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.crops = self.cache / "crops"
        self.faces_dir = self.cache / "faces"
        self.crops.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)

        det = self.cfg["detector"]
        reid = self.cfg["reid"]
        ident = self.cfg["identity_v4"]
        face = self.cfg["face"]
        track = self.cfg["tracking"]

        dev = None if str(reid.get("device", "auto")).lower() == "auto" else reid.get("device")
        self.extractor = ReIDExtractor(
            weights=reid["weights"],
            device=dev,
            max_batch=int(reid.get("max_batch", 32)),
            model=reid.get("model"),
        )
        self.face = FaceExtractorV4(
            model=face.get("model", "buffalo_l"),
            det_size=tuple(face.get("det_size", [640, 640])),
            min_detection=float(face.get("min_detection", 0.55)),
            min_size=int(face.get("min_size", 32)),
            min_quality=float(face.get("min_quality", 0.42)),
            device=str(face.get("device", "auto")),
        )
        self.detector = det
        self.interval = max(1, int(reid.get("interval", 5)))
        self.part_interval = max(self.interval, int(reid.get("part_interval", 15)))
        self.face_interval = max(self.interval, int(face.get("interval", 15)))
        self.min_quality = float(reid.get("min_quality", 0.20))
        self.light = bool(reid.get("illumination_variant", True))
        self.novelty = float(ident.get("novelty", 0.985))
        self.bank = int(ident.get("track_bank", 24))
        self.crop_bank = int(ident.get("crop_bank", 8))
        self.face_bank = int(face.get("bank", 8))
        self.gap = float(track.get("fragment_gap_sec", 2.0))
        self.engine = GlobalIdentityV4(ident)
        self.tracks: Dict[str, Tracklet] = {}
        self.faces: Dict[str, List[Feature]] = {}
        self.meta: Dict[str, dict] = {}
        self.saved: Dict[str, int] = {}
        self.saved_faces: Dict[str, int] = {}

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
        return {"upper": image[:max(1, int(h * 0.68))], "lower": image[int(h * 0.32):]}

    def store_crop(self, key: str, kind: str, image: np.ndarray) -> None:
        count = self.saved.get(key, 0)
        if count >= self.crop_bank or image is None or image.size == 0:
            return
        folder = self.crops / key.replace(":", "_")
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{count:02d}_{kind}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        self.saved[key] = count + 1

    def store_face(self, key: str, image: np.ndarray) -> None:
        count = self.saved_faces.get(key, 0)
        if count >= self.face_bank or image is None or image.size == 0:
            return
        folder = self.faces_dir / key.replace(":", "_")
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{count:02d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        self.saved_faces[key] = count + 1

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
            value = base.copy()
            value["kind"] = kind
            if track.add(vector, kind, float(score), value, self.novelty, self.bank):
                self.store_crop(key, kind, image)
        if track.observations:
            boxes = np.asarray([x["bbox"] for x in track.observations], dtype=np.float32)
            hh = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
            ww = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
            track.shape = float(np.median(hh / ww))
            track.start = min(x["timestamp"] for x in track.observations)
            track.end = max(x["timestamp"] for x in track.observations)

    def add_face(self, key, camera, track_id, stamp, obs):
        if obs is None:
            return
        items = self.faces.setdefault(key, [])
        feature = Feature(
            obs.vector,
            "face",
            float(obs.quality),
            camera,
            float(stamp),
            {"camera": camera, "track_id": track_id, "timestamp": stamp,
             "quality": float(obs.quality), "det": float(obs.detection),
             "width": float(obs.width), "height": float(obs.height),
             "area": float(obs.area), "roll": float(obs.roll)},
        )
        if items:
            best = float(np.max(np.stack([x.vector for x in items]) @ feature.vector))
            if best >= self.novelty and feature.quality <= max(x.quality for x in items) + 0.02:
                return
        items.append(feature)
        items.sort(key=lambda x: x.quality, reverse=True)
        del items[self.face_bank:]
        self.store_face(key, np.asarray(getattr(obs, "crop", np.empty((0, 0), dtype=np.uint8))))

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
        last_body: Dict[str, int] = {}
        last_face: Dict[str, int] = {}
        segments: Dict[int, int] = {}
        seen: Dict[int, int] = {}
        frame = 0
        samples = 0
        face_samples = 0
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
                    person = crop(image, box)
                    q = quality(person) if person is not None else 0.0
                    if person is None or q < self.min_quality:
                        continue

                    if frame - last_body.get(key, -10**9) >= self.interval:
                        crops = [person]
                        names = ["full"]
                        if self.light and frame - last_body.get(key + ":light", -10**9) >= self.part_interval:
                            crops.append(illumination_variant(person))
                            names.append("light")
                            last_body[key + ":light"] = frame
                        if frame - last_body.get(key + ":part", -10**9) >= self.part_interval:
                            for name, part in self.parts(person).items():
                                crops.append(part)
                                names.append(name)
                            last_body[key + ":part"] = frame
                        last_body[key] = frame
                        features = self.extractor.extract_batch(crops)
                        self.add_body(key, camera, tid, seg, box, frame / fps, float(item.confidence), person,
                                      {name: value for name, value in zip(names, features)})
                        samples += 1

                    if frame - last_face.get(key, -10**9) >= self.face_interval:
                        obs = self.face.extract(person)
                        last_face[key] = frame
                        if obs is not None:
                            self.add_face(key, camera, tid, frame / fps, obs)
                            face_samples += 1
        finally:
            cap.release()
            rows.close()
        print(f"[v4] {camera}: frames={frame} tracklets={sum(1 for k in self.tracks if k.startswith(camera + ':'))} body_samples={samples} face_samples={face_samples} total={total}")

    def save_cache(self):
        info = []
        packed = {}
        face_info = {}
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
            face_info[key] = [{"quality": x.quality, "timestamp": x.stamp, "meta": x.meta}
                              for x in self.faces.get(key, [])]
        (self.cache / "tracklets_v4.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        np.savez_compressed(self.cache / "tracklets_v4.npz", **packed)
        (self.cache / "faces_v4.json").write_text(json.dumps(face_info, indent=2), encoding="utf-8")
        face_pack = {key: np.stack([x.vector for x in values]).astype(np.float32)
                     for key, values in self.faces.items() if values}
        np.savez_compressed(self.cache / "faces_v4.npz", **face_pack)
        (self.cache / "video_meta_v4.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def save_debug(self, decisions):
        with (self.out / "identity_debug_v4.jsonl").open("w", encoding="utf-8") as handle:
            for item in decisions:
                handle.write(json.dumps(item.__dict__) + "\n")

    def save_gallery(self):
        body = {}
        face = {}
        meta = {}
        for gid, identity in self.engine.identities.items():
            if identity.trusted:
                body[gid] = np.stack([x.vector for x in identity.trusted]).astype(np.float32)
            values = self.engine.face_trusted.get(gid, [])
            if values:
                face[gid] = np.stack([x.vector for x in values]).astype(np.float32)
            meta[gid] = {
                "tracks": identity.tracks,
                "cameras": sorted(identity.cameras),
                "trusted_body": len(identity.trusted),
                "candidate_body": len(identity.candidate),
                "trusted_face": len(self.engine.face_trusted.get(gid, [])),
                "candidate_face": len(self.engine.face_candidate.get(gid, [])),
            }
        np.savez_compressed(self.out / "global_body_gallery_v4.npz", **body)
        np.savez_compressed(self.out / "global_face_gallery_v4.npz", **face)
        (self.out / "global_gallery_v4.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def render(self, mapping):
        for camera, meta in self.meta.items():
            cap = cv2.VideoCapture(meta["source"])
            out = self.out / f"{camera}_v4.mp4"
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"]))
            rows = {}
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
            print(f"[v4] wrote {out}")

    def run(self, values):
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")
        print(f"[v4] Body ReID: {self.extractor.describe()}")
        print(f"[v4] Face ReID: {self.face.describe()}")
        print(f"[v4] cameras: {len(sources)}")
        print("[v4] pass 1: detect + track + continuous body/face feature collection")
        for camera, path in sources:
            self.collect(camera, path)
        self.save_cache()
        print("[v4] pass 2: persistent multimodal global identity search")
        mapping, decisions = self.engine.run(self.tracks, self.faces)
        self.save_debug(decisions)
        self.save_gallery()
        self.print_summary()
        print("[v4] pass 3: render from saved detections")
        self.render(mapping)
        return mapping

    def print_summary(self):
        data = self.engine.summary(self.tracks)
        print("\n===== V4 REID RESULT =====")
        print(f"tracklets: {data['tracklets']}")
        print(f"global IDs: {data['global_ids']}")
        print(f"multi-camera IDs: {data['multi_camera_count']}")
        print(f"reasons: {json.dumps(data['reasons'], sort_keys=True)}")
        print(f"trusted body features: {data['trusted_body']}")
        print(f"candidate body features: {data['candidate_body']}")
        print(f"trusted face features: {data['trusted_face']}")
        print(f"candidate face features: {data['candidate_face']}")
        print(f"face-assisted matches: {data['face_assisted']}")
        print(f"body-only matches: {data['body_only']}")
        print(f"conflicts: {data['conflicts']}")
        for gid, cams in sorted(data["multi_camera"].items()):
            print(f"  {gid}: {', '.join(cams)}")
        print(f"outputs: {self.out}")
