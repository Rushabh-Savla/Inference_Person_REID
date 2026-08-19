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
from core import GlobalIdentityEngine, Observation, Tracklet, crop, quality  # noqa: E402


class BatchPipeline:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle) or {}

        self.out = Path(self.cfg["input"].get("output_dir", "rebuild_outputs"))
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)

        det = self.cfg["detector"]
        reid = self.cfg["reid"]
        ident = self.cfg["identity"]
        tracking = self.cfg["tracking"]

        device = None if str(reid.get("device", "auto")).lower() == "auto" else reid.get("device")
        self.extractor = ReIDExtractor(
            weights=reid["weights"],
            device=device,
            max_batch=int(reid.get("max_batch", 32)),
            model=reid.get("model"),
        )
        self.detector_cfg = det
        self.interval = max(1, int(reid.get("interval", 5)))
        self.fragment_gap_sec = float(tracking.get("fragment_gap_sec", 2.0))
        self.min_embeddings = int(ident.get("min_embeddings", 4))
        self.engine = GlobalIdentityEngine(
            threshold=float(ident["match_threshold"]),
            margin=float(ident["match_margin"]),
            strong=float(ident["strong_threshold"]),
            bank_size=int(ident.get("bank_size", 8)),
        )
        self.tracklets: Dict[str, Tracklet] = {}
        self.video_meta: Dict[str, dict] = {}

    def sources(self, values: List[str]) -> List[Tuple[str, str]]:
        if values:
            return [(Path(v).stem, v) for v in values]
        configured = self.cfg.get("input", {}).get("videos", [])
        out = []
        for item in configured:
            if isinstance(item, dict):
                out.append((item.get("name") or Path(item["path"]).stem, item["path"]))
            else:
                out.append((Path(item).stem, item))
        return out

    def run(self, sources: List[str]) -> Dict[str, str]:
        camera_sources = self.sources(sources)
        if not camera_sources:
            raise SystemExit("No videos supplied. Use --videos camera1.mp4 camera2.mp4 ...")

        print(f"[clean] ReID: {self.extractor.describe()}")
        print(f"[clean] cameras: {len(camera_sources)}")
        print("[clean] pass 1: detect + track + collect high-quality embeddings")
        for camera, path in camera_sources:
            self.collect_camera(camera, path)

        self.save_cache()
        print("[clean] pass 2: global identity reconciliation")
        mapping, matches = self.engine.reconcile(self.tracklets)
        self.add_unmatched(mapping)
        self.save_mapping(mapping, matches)
        self.save_gallery(mapping)
        self.print_diagnostics(mapping, matches)

        print("[clean] pass 3: render saved detections (no detector or ReID rerun)")
        self.render(mapping)
        self.summary(mapping, matches)
        return mapping

    def collect_camera(self, camera: str, path: str) -> None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_meta[camera] = {"source": path, "fps": fps, "width": width, "height": height, "frames": total}

        detector = PersonDetector(
            model_path=self.detector_cfg["model"],
            confidence_threshold=float(self.detector_cfg["conf"]),
            person_class_id=0,
            tracker_config=self.detector_cfg["tracker"],
            pose_ensemble=None,
            iou=float(self.detector_cfg["iou"]),
        )

        detections_path = self.cache / f"{camera}.detections.jsonl"
        embeddings_path = self.cache / f"{camera}.embeddings.npy"
        events = detections_path.open("w", encoding="utf-8")
        vectors: List[np.ndarray] = []
        last_embed: Dict[str, int] = {}
        last_seen: Dict[int, int] = {}
        segments: Dict[int, int] = {}
        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                detections = detector.track(frame)
                crops = []
                meta = []

                for item in detections:
                    if item.track_id is None:
                        continue
                    track_id = int(item.track_id)
                    previous = last_seen.get(track_id)
                    if previous is None or frame_index - previous > int(self.fragment_gap_sec * fps):
                        segments[track_id] = segments.get(track_id, 0) + 1
                    segment = segments[track_id]
                    key = f"{camera}:{track_id}:{segment}"
                    last_seen[track_id] = frame_index
                    bbox = (item.x1, item.y1, item.x2, item.y2)
                    events.write(json.dumps({
                        "camera": camera,
                        "frame": frame_index,
                        "timestamp": frame_index / fps,
                        "track_id": track_id,
                        "tracklet_key": key,
                        "bbox": list(bbox),
                        "detection_score": float(item.confidence),
                    }) + "\n")

                    if frame_index - last_embed.get(key, -10**9) < self.interval:
                        continue
                    person = crop(frame, bbox)
                    if person is None:
                        continue
                    q = quality(person, frame.shape[:2])
                    if q < 0.20:
                        continue
                    crops.append(person)
                    meta.append((key, track_id, segment, bbox, q, float(item.confidence)))
                    last_embed[key] = frame_index

                if crops:
                    features = self.extractor.extract_batch(crops)
                    for feature, (key, track_id, segment, bbox, q, det_score) in zip(features, meta):
                        tracklet = self.tracklets.get(key)
                        if tracklet is None:
                            tracklet = Tracklet(camera=camera, track_id=track_id, segment=segment, fps=fps)
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
                        obs.embedding_index = tracklet.add_embedding(feature, q)
                        tracklet.observations.append(obs)
                        vectors.append(feature.astype(np.float16))
        finally:
            cap.release()
            events.close()

        matrix = np.stack(vectors) if vectors else np.empty((0, self.extractor.embedding_dim), dtype=np.float16)
        np.save(embeddings_path, matrix)
        camera_count = sum(1 for k in self.tracklets if k.startswith(camera + ":"))
        camera_embeds = sum(v.count for k, v in self.tracklets.items() if k.startswith(camera + ":"))
        print(f"[clean] {camera}: frames={frame_index} tracklets={camera_count} embeddings={camera_embeds}")

    def add_unmatched(self, mapping: Dict[str, str]) -> None:
        next_id = 1 + max([int(gid[1:]) for gid in mapping.values()] or [0])
        for key in sorted(self.tracklets):
            if key not in mapping:
                mapping[key] = f"G{next_id:06d}"
                next_id += 1

    def save_cache(self) -> None:
        rows = []
        packed = {}
        for key, track in sorted(self.tracklets.items()):
            rows.append({
                "key": key,
                "camera": track.camera,
                "track_id": track.track_id,
                "segment": track.segment,
                "fps": track.fps,
                "start": track.start,
                "end": track.end,
                "embedding_count": track.count,
                "quality": track.embedding_quality,
            })
            if track.embeddings:
                packed[key] = np.stack(track.embeddings).astype(np.float32)
        (self.cache / "tracklets.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        np.savez_compressed(self.cache / "tracklet_embeddings.npz", **packed)
        (self.cache / "video_meta.json").write_text(json.dumps(self.video_meta, indent=2), encoding="utf-8")

    def save_mapping(self, mapping: Dict[str, str], matches) -> None:
        (self.out / "track_to_global.json").write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
        rows = []
        for match in matches:
            rows.append({
                "left": match.left,
                "right": match.right,
                "left_global": mapping[match.left],
                "right_global": mapping[match.right],
                "score": match.score,
                "margin_left": match.margin_left,
                "margin_right": match.margin_right,
                "reciprocal": match.reciprocal,
                "camera_relation": "same" if self.tracklets[match.left].camera == self.tracklets[match.right].camera else "cross",
                "decision": "MERGE",
            })
        with (self.out / "identity_matches.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def save_gallery(self, mapping: Dict[str, str]) -> None:
        groups: Dict[str, List[str]] = {}
        for key, gid in mapping.items():
            groups.setdefault(gid, []).append(key)
        gallery = {}
        for gid, keys in groups.items():
            valid = [self.tracklets[k] for k in keys if self.tracklets[k].count]
            if not valid:
                continue
            reps = np.stack([track.prototype(self.engine.bank_size) for track in valid])
            gallery[gid] = unit(reps.mean(axis=0)).astype(np.float32)
        np.savez_compressed(self.out / "global_gallery.npz", **gallery)

    def print_diagnostics(self, mapping: Dict[str, str], matches) -> None:
        score_stats = self.engine.summarize_scores(self.tracklets)
        print("[clean] score space:", json.dumps(score_stats, sort_keys=True))
        print(f"[clean] accepted associations: {len(matches)}")
        same = sum(self.tracklets[m.left].camera == self.tracklets[m.right].camera for m in matches)
        cross = len(matches) - same
        print(f"[clean] same-camera associations: {same}")
        print(f"[clean] cross-camera associations: {cross}")
        for match in matches[:30]:
            relation = "same" if self.tracklets[match.left].camera == self.tracklets[match.right].camera else "cross"
            print(f"  {relation}: {match.left} <-> {match.right} score={match.score:.4f} margins={match.margin_left:.4f}/{match.margin_right:.4f}")

    def render(self, mapping: Dict[str, str]) -> None:
        for camera, meta in self.video_meta.items():
            cap = cv2.VideoCapture(meta["source"])
            output = self.out / f"{camera}.mp4"
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"])
            )
            frame_rows: Dict[int, List[dict]] = {}
            with (self.cache / f"{camera}.detections.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    frame_rows.setdefault(int(record["frame"]), []).append(record)
            frame_index = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_index += 1
                    for record in frame_rows.get(frame_index, []):
                        gid = mapping[record["tracklet_key"]]
                        x1, y1, x2, y2 = [int(v) for v in record["bbox"]]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{gid} T{record['track_id']}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, camera, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    writer.write(frame)
            finally:
                cap.release()
                writer.release()
            print(f"[clean] wrote {output}")

    def summary(self, mapping: Dict[str, str], matches) -> None:
        groups: Dict[str, set] = {}
        for key, gid in mapping.items():
            groups.setdefault(gid, set()).add(self.tracklets[key].camera)
        multi = {gid: cams for gid, cams in groups.items() if len(cams) > 1}
        same = sum(self.tracklets[m.left].camera == self.tracklets[m.right].camera for m in matches)
        cross = len(matches) - same
        print("\n===== CLEAN REID RESULT =====")
        print(f"tracklets: {len(mapping)}")
        print(f"global IDs: {len(set(mapping.values()))}")
        print(f"same-camera merges: {same}")
        print(f"cross-camera merges: {cross}")
        print(f"multi-camera IDs: {len(multi)}")
        for gid, cams in sorted(multi.items()):
            print(f"  {gid}: {', '.join(sorted(cams))}")
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
