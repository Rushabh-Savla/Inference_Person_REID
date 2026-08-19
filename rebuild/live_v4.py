from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict

import cv2
import yaml

from detector import PersonDetector
from reid.extractor import ReIDExtractor
from rebuild.batch_v4 import BatchPipelineV4
from rebuild.face_v4 import FaceExtractorV4
from rebuild.identity_v2 import crop, illumination_variant, quality
from rebuild.identity_v3 import Feature, Tracklet
from rebuild.identity_v4 import GlobalIdentityV4


class LiveCameraV4(threading.Thread):
    def __init__(self, camera, source, cfg, extractor, face, engine, lock, output_dir=None, show=False):
        super().__init__(daemon=True)
        self.camera = camera
        self.source = source
        self.cfg = cfg
        self.extractor = extractor
        self.face = face
        self.engine = engine
        self.lock = lock
        self.output_dir = Path(output_dir) if output_dir else None
        self.show = show
        self.error = None
        self.stop_flag = threading.Event()
        self.tracks: Dict[str, Tracklet] = {}
        self.faces: Dict[str, list[Feature]] = {}
        self.track_gid: Dict[str, str] = {}
        self.last_body: Dict[str, int] = {}
        self.last_face: Dict[str, int] = {}
        self.segment: Dict[int, int] = {}
        self.last_seen: Dict[int, int] = {}

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
            writer = cv2.VideoWriter(str(self.output_dir / f"{self.camera}_v4.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            events = (self.output_dir / f"{self.camera}_v4.events.jsonl").open("w", encoding="utf-8")

        det = self.cfg["detector"]
        reid = self.cfg["reid"]
        face_cfg = self.cfg["face"]
        detector = PersonDetector(model_path=det["model"], confidence_threshold=float(det["conf"]),
                                  person_class_id=0, tracker_config=det["tracker"], pose_ensemble=None,
                                  iou=float(det["iou"]))
        interval = max(1, int(reid.get("interval", 5)))
        face_interval = max(interval, int(face_cfg.get("interval", 15)))
        min_quality = float(reid.get("min_quality", 0.20))
        use_light = bool(reid.get("illumination_variant", True))
        frame = 0

        try:
            while not self.stop_flag.is_set():
                ok, image = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                frame += 1
                detections = detector.track(image)
                body_jobs = []
                for item in detections:
                    if item.track_id is None:
                        continue
                    tid = int(item.track_id)
                    previous = self.last_seen.get(tid)
                    if previous is None or frame - previous > int(float(self.cfg["tracking"].get("fragment_gap_sec", 2.0)) * fps):
                        self.segment[tid] = self.segment.get(tid, 0) + 1
                    self.last_seen[tid] = frame
                    seg = self.segment[tid]
                    key = f"{self.camera}:{tid}:{seg}"
                    bbox = (item.x1, item.y1, item.x2, item.y2)
                    person = crop(image, bbox)
                    q = quality(person) if person is not None else 0.0
                    if person is None or q < min_quality:
                        continue
                    if frame - self.last_body.get(key, -10**9) >= interval:
                        crops = [person]
                        names = ["full"]
                        if use_light:
                            crops.append(illumination_variant(person))
                            names.append("light")
                        body_jobs.append((key, tid, seg, bbox, person, q, float(item.confidence), crops, names))
                        self.last_body[key] = frame

                if body_jobs:
                    all_crops = [crop for job in body_jobs for crop in job[7]]
                    vectors = self.extractor.extract_batch(all_crops)
                    pos = 0
                    for key, tid, seg, bbox, person, q, det_score, crops, names in body_jobs:
                        track = self.tracks.get(key)
                        if track is None:
                            track = Tracklet(self.camera, tid, seg, fps)
                            self.tracks[key] = track
                        for name in names:
                            row = {"camera": self.camera, "frame": frame, "timestamp": frame / fps,
                                   "track_id": tid, "bbox": list(bbox), "quality": q, "kind": name}
                            track.add(vectors[pos], name, q, row, float(self.engine.novelty), self.engine.gallery)
                            pos += 1
                        track.end = frame / fps
                        if track.start == 0.0:
                            track.start = frame / fps

                        if frame - self.last_face.get(key, -10**9) >= face_interval:
                            self.last_face[key] = frame
                            face_obs = self.face.extract(person)
                            if face_obs is not None:
                                feat = Feature(face_obs.vector, "face", face_obs.quality, self.camera, frame / fps,
                                                {"camera": self.camera, "frame": frame, "track_id": tid,
                                                 "quality": face_obs.quality})
                                self.faces.setdefault(key, []).append(feat)
                                self.faces[key] = sorted(self.faces[key], key=lambda x: x.quality, reverse=True)[:8]

                        with self.lock:
                            gid = self.track_gid.get(key)
                            if gid is None:
                                decision = self.engine.assign(track, self.faces.get(key, []), self.tracks)
                                if decision.gid != "PENDING":
                                    gid = decision.gid
                                    self.track_gid[key] = gid
                            else:
                                identity = self.engine.identities.get(gid)
                                if identity is not None:
                                    identity.add(track, False, self.engine.gallery, self.engine.promote)
                                    self.engine.add_face(gid, self.faces.get(key, []), False)
                                decision = None

                        if gid is not None:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(image, gid, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)
                            if events:
                                events.write(json.dumps({"camera": self.camera, "frame": frame, "timestamp": frame / fps,
                                                         "track_id": tid, "global_id": gid,
                                                         "decision": decision.reason if decision else "track_update"}) + "\n")

                cv2.putText(image, self.camera, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                if writer:
                    writer.write(image)
                if self.show:
                    cv2.imshow(self.camera, image)
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


def run_live_v4(cfg_path: str, sources: Dict[str, str], output_dir: str | None, show: bool):
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    reid = cfg["reid"]
    device = None if str(reid.get("device", "auto")).lower() == "auto" else reid.get("device")
    extractor = ReIDExtractor(weights=reid["weights"], device=device, max_batch=int(reid.get("max_batch", 32)), model=reid.get("model"))
    face = FaceExtractorV4(model=cfg["face"].get("model", "buffalo_l"), det_size=tuple(cfg["face"].get("det_size", [640, 640])),
                           min_detection=float(cfg["face"].get("min_detection", 0.55)), min_size=int(cfg["face"].get("min_size", 32)),
                           min_quality=float(cfg["face"].get("min_quality", 0.42)), device=str(cfg["face"].get("device", "auto")))
    engine = GlobalIdentityV4(cfg["identity_v4"])
    lock = threading.Lock()
    workers = [LiveCameraV4(name, source, cfg, extractor, face, engine, lock, output_dir, show) for name, source in sources.items()]
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
