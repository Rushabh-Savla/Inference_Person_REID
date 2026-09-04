from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detector import PersonDetector  # noqa: E402
from reid.extractor import ReIDExtractor  # noqa: E402
from rebuild.identity_v2 import (  # noqa: E402
    GlobalIdentityEngine,
    OnlineGlobalGallery,
    Tracklet,
    crop,
    illumination_variant,
    quality,
    unit,
)


class LiveCameraV2(threading.Thread):
    def __init__(self, camera, source, cfg, extractor, gallery, output_dir=None, show=False):
        super().__init__(daemon=True)
        self.camera = camera
        self.source = source
        self.cfg = cfg
        self.extractor = extractor
        self.gallery = gallery
        self.output_dir = Path(output_dir) if output_dir else None
        self.show = show
        self.error = None
        self.stop_flag = threading.Event()
        self.tracks: Dict[int, Tracklet] = {}
        self.last_embed: Dict[int, int] = {}

    def stop(self):
        self.stop_flag.set()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = RuntimeError(f"Cannot open stream: {self.source}")
            return
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        writer = None
        events = None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(self.output_dir / f"{self.camera}_v2.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            events = (self.output_dir / f"{self.camera}_v2.events.jsonl").open("w", encoding="utf-8")

        det_cfg = self.cfg["detector"]
        reid_cfg = self.cfg["reid"]
        detector = PersonDetector(
            model_path=det_cfg["model"],
            confidence_threshold=float(det_cfg["conf"]),
            person_class_id=0,
            tracker_config=det_cfg["tracker"],
            pose_ensemble=None,
            iou=float(det_cfg["iou"]),
        )
        interval = max(1, int(reid_cfg.get("interval", 5)))
        use_light = bool(reid_cfg.get("illumination_variant", True))
        min_quality = float(reid_cfg.get("min_quality", 0.20))
        frame_index = 0

        try:
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                frame_index += 1
                detections = detector.track(frame)
                crops = []
                meta = []

                for item in detections:
                    if item.track_id is None:
                        continue
                    track_id = int(item.track_id)
                    if frame_index - self.last_embed.get(track_id, -10**9) < interval:
                        continue
                    bbox = (item.x1, item.y1, item.x2, item.y2)
                    person = crop(frame, bbox)
                    q = quality(person) if person is not None else 0.0
                    if person is None or q < min_quality:
                        continue
                    crops.append(person)
                    if use_light:
                        crops.append(illumination_variant(person))
                    meta.append((track_id, bbox, q, float(item.confidence)))
                    self.last_embed[track_id] = frame_index

                if crops:
                    vectors = self.extractor.extract_batch(crops)
                    pos = 0
                    for track_id, bbox, q, det_score in meta:
                        track = self.tracks.get(track_id)
                        if track is None:
                            track = Tracklet(camera=self.camera, track_id=track_id, segment=1, fps=fps)
                            self.tracks[track_id] = track
                        row = {
                            "camera": self.camera,
                            "frame": frame_index,
                            "timestamp": frame_index / fps,
                            "track_id": track_id,
                            "bbox": list(bbox),
                            "detection_score": det_score,
                            "quality": q,
                        }
                        track.add(vectors[pos], q, row)
                        pos += 1
                        if use_light:
                            light = vectors[pos]
                            pos += 1
                            if float(unit(track.embeddings[-1]) @ unit(light)) < 0.995:
                                track.add(light, max(0.10, q * 0.92), {**row, "variant": "illumination"})

                        gid, reason, score = self.gallery.update(track)
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{gid} T{track_id}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                        if events:
                            events.write(json.dumps({
                                "camera": self.camera,
                                "frame": frame_index,
                                "timestamp": frame_index / fps,
                                "track_id": track_id,
                                "bbox": list(bbox),
                                "global_id": None if gid == "PENDING" else gid,
                                "reason": reason,
                                "score": score,
                            }) + "\n")

                cv2.putText(frame, self.camera, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                if writer:
                    writer.write(frame)
                if self.show:
                    cv2.imshow(self.camera, frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.stop_flag.set()
                        break
        except Exception as exc:
            self.error = exc
        finally:
            cap.release()
            if writer:
                writer.release()
            if events:
                events.close()


def run_live_v2(cfg_path: str, sources: Dict[str, str], output_dir: str | None, show: bool):
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    reid = cfg["reid"]
    ident = cfg["identity"]
    device = None if str(reid.get("device", "auto")).lower() == "auto" else reid.get("device")
    extractor = ReIDExtractor(
        weights=reid["weights"],
        device=device,
        max_batch=int(reid.get("max_batch", 32)),
        model=reid.get("model"),
    )
    engine = GlobalIdentityEngine(
        threshold=float(ident["match_threshold"]),
        margin=float(ident["match_margin"]),
        strong=float(ident["strong_threshold"]),
        bank_size=int(ident.get("bank_size", 12)),
        passes=int(ident.get("global_passes", 3)),
    )
    gallery = OnlineGlobalGallery(
        engine,
        min_embeddings=int(ident.get("min_embeddings", 6)),
        new_id_embeddings=int(ident.get("new_id_embeddings", 8)),
    )
    workers = [LiveCameraV2(name, source, cfg, extractor, gallery, output_dir, show) for name, source in sources.items()]
    for worker in workers:
        worker.start()
    try:
        while any(worker.is_alive() for worker in workers):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=5)
        if show:
            cv2.destroyAllWindows()
    errors = [worker.error for worker in workers if worker.error]
    if errors:
        raise RuntimeError("Live camera failure: " + "; ".join(map(str, errors)))
