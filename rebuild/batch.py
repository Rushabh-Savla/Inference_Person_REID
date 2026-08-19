from __future__ import annotations

import json
import os
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

from core import Observation, OfflineReconciler, Tracklet, crop, quality  # noqa: E402


class BatchPipeline:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle) or {}

        self.out = Path(self.cfg.get("input", {}).get("output_dir", "rebuild_outputs"))
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)

        det = self.cfg["detector"]
        self.detector_cfg = det

        reid = self.cfg["reid"]
        device = None if str(reid.get("device", "auto")).lower() == "auto" else reid.get("device")
        self.extractor = ReIDExtractor(
            weights=reid["weights"],
            device=device,
            max_batch=int(reid.get("max_batch", 32)),
            model=reid.get("model"),
        )

        ident = self.cfg["identity"]
        self.reconciler = OfflineReconciler(
            same_threshold=float(ident["same_threshold"]),
            cross_threshold=float(ident["cross_threshold"]),
            min_margin=float(ident["min_margin"]),
            bank_size=int(ident["bank_size"]),
            max_same_camera_gap_sec=float(ident["max_same_camera_gap_sec"]),
        )

        self.tracklets: Dict[str, Tracklet] = {}
        self.embedding_store: List[np.ndarray] = []
        self.observations: List[Observation] = []
        self.video_meta: Dict[str, dict] = {}

    def sources(self, values: List[str]) -> List[Tuple[str, str]]:
        if values:
            result = []
            for value in values:
                path = Path(value)
                result.append((path.stem, str(path)))
            return result
        configured = self.cfg.get("input", {}).get("videos", [])
        result = []
        for entry in configured:
            if isinstance(entry, dict):
                result.append((entry.get("name") or Path(entry["path"]).stem, entry["path"]))
            else:
                result.append((Path(entry).stem, entry))
        return result

    def run(self, sources: List[str]) -> Dict[str, str]:
        camera_sources = self.sources(sources)
        if not camera_sources:
            raise SystemExit("No videos supplied. Use: python rebuild/run.py batch --videos ...")

        print(f"[clean] ReID: {self.extractor.describe()}")
        print(f"[clean] cameras: {len(camera_sources)}")

        for camera, path in camera_sources:
            self.collect_camera(camera, path)

        self.save_cache()
        mapping = self.reconcile()
        self.save_mapping(mapping)
        self.render(mapping)
        self.summary(mapping)
        return mapping

    def collect_camera(self, camera: str, path: str) -> None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_meta[camera] = {
            "source": path,
            "fps": fps,
            "width": width,
            "height": height,
            "frames": total,
        }

        detector = PersonDetector(
            model_path=self.detector_cfg["model"],
            confidence_threshold=float(self.detector_cfg["conf"]),
            person_class_id=0,
            tracker_config=self.detector_cfg["tracker"],
            pose_ensemble=None,
            iou=float(self.detector_cfg["iou"]),
        )

        interval = max(1, int(self.cfg["reid"]["interval"]))
        last_embed: Dict[int, int] = {}
        detections_path = self.cache / f"{camera}.detections.jsonl"
        embeddings_path = self.cache / f"{camera}.embeddings.npy"
        events = detections_path.open("w", encoding="utf-8")
        local_vectors: List[np.ndarray] = []

        frame_index = 0
        pending: List[Tuple[int, int, Tuple[int, int, int, int], float, float, np.ndarray]] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            detections = detector.track(frame)
            batch_crops = []
            batch_meta = []

            for det in detections:
                if det.track_id is None:
                    continue
                bbox = (det.x1, det.y1, det.x2, det.y2)
                events.write(
                    json.dumps(
                        {
                            "camera": camera,
                            "frame": frame_index,
                            "timestamp": frame_index / fps,
                            "track_id": int(det.track_id),
                            "bbox": list(bbox),
                            "detection_score": float(det.confidence),
                        }
                    )
                    + "\n"
                )

                if frame_index - last_embed.get(int(det.track_id), -10**9) < interval:
                    continue

                person = crop(frame, bbox)
                if person is None:
                    continue
                q = quality(person, frame.shape[:2])
                if q < 0.20:
                    continue

                batch_crops.append(person)
                batch_meta.append((int(det.track_id), bbox, q, float(det.confidence)))
                last_embed[int(det.track_id)] = frame_index

            if batch_crops:
                vectors = self.extractor.extract_batch(batch_crops)
                for vector, meta in zip(vectors, batch_meta):
                    track_id, bbox, q, det_score = meta
                    key = f"{camera}:{track_id}"
                    tracklet = self.tracklets.get(key)
                    if tracklet is None:
                        tracklet = Tracklet(camera=camera, track_id=track_id, fps=fps)
                        self.tracklets[key] = tracklet
                    obs = Observation(
                        camera=camera,
                        frame=frame_index,
                        timestamp=frame_index / fps,
                        track_id=track_id,
                        bbox=bbox,
                        detection_score=det_score,
                        quality=q,
                    )
                    index = tracklet.add_embedding(vector, q)
                    obs.embedding_index = index
                    tracklet.observations.append(obs)
                    self.observations.append(obs)
                    local_vectors.append(vector.astype(np.float16))

        cap.release()
        events.close()
        np.save(embeddings_path, np.stack(local_vectors) if local_vectors else np.empty((0, self.extractor.embedding_dim), dtype=np.float16))

        print(
            f"[clean] {camera}: frames={frame_index} "
            f"tracklets={sum(1 for key in self.tracklets if key.startswith(camera + ':'))} "
            f"embeddings={len(local_vectors)}"
        )

    def save_cache(self) -> None:
        data = []
        for key, tracklet in sorted(self.tracklets.items()):
            data.append(
                {
                    "key": key,
                    "camera": tracklet.camera,
                    "track_id": tracklet.track_id,
                    "fps": tracklet.fps,
                    "start": tracklet.start,
                    "end": tracklet.end,
                    "embedding_count": tracklet.count,
                    "quality": tracklet.embedding_quality,
                }
            )
        with (self.cache / "tracklets.json").open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def reconcile(self) -> Dict[str, str]:
        usable = {
            key: tracklet
            for key, tracklet in self.tracklets.items()
            if tracklet.count >= int(self.cfg["identity"]["min_embeddings"])
        }
        mapping = self.reconciler.reconcile(usable)

        for key in self.tracklets:
            if key not in mapping:
                # Very short tracklets are not allowed to contaminate global IDs.
                # They receive their own identity only for visualization.
                next_number = 1 + max(
                    [int(value[1:]) for value in mapping.values()] or [0]
                )
                mapping[key] = f"G{next_number:06d}"
        return mapping

    def save_mapping(self, mapping: Dict[str, str]) -> None:
        with (self.out / "track_to_global.json").open("w", encoding="utf-8") as handle:
            json.dump(mapping, handle, indent=2, sort_keys=True)

    def render(self, mapping: Dict[str, str]) -> None:
        interval_lookup = {}
        for camera, meta in self.video_meta.items():
            src = meta["source"]
            cap = cv2.VideoCapture(src)
            output = self.out / f"{camera}.mp4"
            writer = cv2.VideoWriter(
                str(output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                meta["fps"],
                (meta["width"], meta["height"]),
            )
            records: Dict[int, List[dict]] = {}
            cache_path = self.cache / f"{camera}.detections.jsonl"
            with cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    records.setdefault(int(record["frame"]), []).append(record)

            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                for record in records.get(frame_index, []):
                    box = tuple(record["bbox"])
                    key = f"{camera}:{int(record['track_id'])}"
                    gid = mapping[key]
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{gid} T{record['track_id']}",
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    frame,
                    camera,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)

            cap.release()
            writer.release()
            print(f"[clean] wrote {output}")

    def summary(self, mapping: Dict[str, str]) -> None:
        cameras: Dict[str, set] = {}
        for key, gid in mapping.items():
            camera = key.split(":", 1)[0]
            cameras.setdefault(gid, set()).add(camera)

        multi = {gid: values for gid, values in cameras.items() if len(values) > 1}
        print("\n===== CLEAN BATCH RESULT =====")
        print(f"tracklets: {len(mapping)}")
        print(f"global IDs: {len(set(mapping.values()))}")
        print(f"multi-camera IDs: {len(multi)}")
        for gid, values in sorted(multi.items()):
            print(f"  {gid}: {', '.join(sorted(values))}")
        print(f"outputs: {self.out}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rebuild/config.yaml")
    parser.add_argument("--videos", nargs="*", default=[])
    args = parser.parse_args()
    BatchPipeline(args.config).run(args.videos)


if __name__ == "__main__":
    main()
