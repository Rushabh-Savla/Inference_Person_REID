from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detector import PersonDetector  # noqa: E402
from reid.extractor import ReIDExtractor  # noqa: E402

from core import Observation, Tracklet, OfflineReconciler, crop, quality  # noqa: E402


class LiveGallery:
    def __init__(self, cfg: dict):
        ident = cfg["identity"]
        self.engine = OfflineReconciler(
            same_threshold=float(ident["same_threshold"]),
            cross_threshold=float(ident["cross_threshold"]),
            min_margin=float(ident["min_margin"]),
            bank_size=int(ident["bank_size"]),
            max_same_camera_gap_sec=float(ident["max_same_camera_gap_sec"]),
        )
        self.min_embeddings = int(ident["min_embeddings"])
        self.tracks: Dict[str, Tracklet] = {}
        self.gids: Dict[str, str] = {}
        self.next_id = 1
        self.lock = threading.Lock()

    def update(
        self,
        camera: str,
        track_id: int,
        fps: float,
        frame: int,
        bbox: Tuple[int, int, int, int],
        det_score: float,
        emb: np.ndarray,
        q: float,
    ) -> str:
        key = f"{camera}:{track_id}"
        with self.lock:
            track = self.tracks.get(key)
            if track is None:
                track = Tracklet(camera=camera, track_id=track_id, fps=fps)
                self.tracks[key] = track

            obs = Observation(
                camera=camera,
                frame=frame,
                timestamp=frame / fps,
                track_id=track_id,
                bbox=bbox,
                detection_score=det_score,
                quality=q,
            )
            track.observations.append(obs)
            track.add_embedding(emb, q)

            if key in self.gids:
                return self.gids[key]
            if track.count < self.min_embeddings:
                return "PENDING"

            candidates = []
            for other_key, other in self.tracks.items():
                if other_key == key or other.count < self.min_embeddings:
                    continue
                if other.camera == camera and self.engine.overlap(other, track):
                    continue
                value = self.engine.score(track, other, self.engine.bank_size)
                candidates.append((value, other_key))

            candidates.sort(reverse=True)
            if candidates:
                best, other_key = candidates[0]
                second = candidates[1][0] if len(candidates) > 1 else 0.0
                other_camera = self.tracks[other_key].camera
                threshold = (
                    self.engine.same_threshold
                    if other_camera == camera
                    else self.engine.cross_threshold
                )
                if best >= threshold and best - second >= self.engine.min_margin:
                    gid = self.gids.get(other_key)
                    if gid is not None:
                        self.gids[key] = gid
                        return gid

            gid = f"G{self.next_id:06d}"
            self.next_id += 1
            self.gids[key] = gid
            return gid


class LiveCamera(threading.Thread):
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
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"{self.camera}.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        det = self.cfg["detector"]
        reid_cfg = self.cfg["reid"]
        detector = PersonDetector(
            model_path=det["model"],
            confidence_threshold=float(det["conf"]),
            person_class_id=0,
            tracker_config=det["tracker"],
            pose_ensemble=None,
            iou=float(det["iou"]),
        )
        interval = max(1, int(reid_cfg.get("interval", 5)))
        last_embed = {}
        frame_index = 0
        events = None
        if self.output_dir:
            events = (self.output_dir / f"{self.camera}.events.jsonl").open("w", encoding="utf-8")

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
                for det_box in detections:
                    if det_box.track_id is None:
                        continue
                    bbox = (det_box.x1, det_box.y1, det_box.x2, det_box.y2)
                    person = crop(frame, bbox)
                    if person is None:
                        continue
                    key = int(det_box.track_id)
                    if frame_index - last_embed.get(key, -10**9) < interval:
                        continue
                    q = quality(person, frame.shape[:2])
                    if q < 0.20:
                        continue
                    crops.append(person)
                    meta.append((key, bbox, q, float(det_box.confidence)))
                    last_embed[key] = frame_index

                if crops:
                    vectors = self.extractor.extract_batch(crops)
                    for vector, item in zip(vectors, meta):
                        track_id, bbox, q, det_score = item
                        gid = self.gallery.update(
                            self.camera,
                            track_id,
                            fps,
                            frame_index,
                            bbox,
                            det_score,
                            vector,
                            q,
                        )
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{gid} T{track_id}",
                            (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        if events:
                            events.write(
                                json.dumps(
                                    {
                                        "camera": self.camera,
                                        "frame": frame_index,
                                        "timestamp": frame_index / fps,
                                        "track_id": track_id,
                                        "bbox": list(bbox),
                                        "global_id": None if gid == "PENDING" else gid,
                                        "reason": "pending" if gid == "PENDING" else "assigned",
                                    }
                                )
                                + "\n"
                            )

                cv2.putText(
                    frame,
                    self.camera,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if writer:
                    writer.write(frame)
                if self.show:
                    cv2.imshow(self.camera, frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.stop_flag.set()
                        break
        except Exception as exc:  # noqa: BLE001
            self.error = exc
        finally:
            cap.release()
            if writer:
                writer.release()
            if events:
                events.close()


def run_live(cfg_path: str, sources: Dict[str, str], output_dir: str | None, show: bool):
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    reid = cfg["reid"]
    device = None if str(reid.get("device", "auto")).lower() == "auto" else reid.get("device")
    extractor = ReIDExtractor(
        weights=reid["weights"],
        device=device,
        max_batch=int(reid.get("max_batch", 32)),
        model=reid.get("model"),
    )
    gallery = LiveGallery(cfg)

    workers = [
        LiveCamera(name, source, cfg, extractor, gallery, output_dir, show)
        for name, source in sources.items()
    ]
    for worker in workers:
        worker.start()

    try:
        while True:
            alive = any(worker.is_alive() for worker in workers)
            if not alive:
                break
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
