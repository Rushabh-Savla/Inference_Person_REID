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
from rebuild.identity_v2 import (  # noqa: E402
    GlobalIdentityEngine,
    Tracklet,
    crop,
    illumination_variant,
    quality,
    unit,
)


class BatchPipelineV2:
    """Second-stage clean batch pipeline.

    Important design change: every sampled high-quality observation contributes
    to a persistent multi-view gallery. We keep diverse embeddings instead of
    collapsing a person to a single centroid. The same global gallery is then
    used for same-camera and cross-camera association.
    """

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
        self.use_light = bool(reid.get("illumination_variant", True))
        self.min_quality = float(reid.get("min_quality", 0.20))
        self.fragment_gap_sec = float(tracking.get("fragment_gap_sec", 2.0))
        self.bank_size = int(ident.get("bank_size", 12))
        self.engine = GlobalIdentityEngine(
            threshold=float(ident["match_threshold"]),
            margin=float(ident["match_margin"]),
            strong=float(ident["strong_threshold"]),
            bank_size=self.bank_size,
            passes=int(ident.get("global_passes", 3)),
        )
        self.tracklets: Dict[str, Tracklet] = {}
        self.video_meta: Dict[str, dict] = {}
        self.matched_camera_by_tracklet: Dict[str, str] = {}

    def sources(self, values: List[str]) -> List[Tuple[str, str]]:
        if values:
            return [(Path(v).stem, v) for v in values]
        configured = self.cfg.get("input", {}).get("videos", [])
        result = []
        for item in configured:
            if isinstance(item, dict):
                result.append((item.get("name") or Path(item["path"]).stem, item["path"]))
            else:
                result.append((Path(item).stem, item))
        return result

    def run(self, values: List[str]) -> Dict[str, str]:
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied. Use --videos camera1.mp4 camera2.mp4 ...")

        print(f"[v2] ReID: {self.extractor.describe()}")
        print(f"[v2] cameras: {len(sources)}")
        print("[v2] pass 1: detect + track + continuously collect diverse embeddings")
        for camera, path in sources:
            self.collect_camera(camera, path)

        self.save_cache()
        print("[v2] pass 2: shared global multi-view reconciliation")
        mapping, matches = self.engine.reconcile(self.tracklets)
        self.matched_camera_by_tracklet = self.build_matched_camera_map(matches)
        self.save_mapping(mapping, matches)
        self.save_gallery(mapping)
        self.print_diagnostics(mapping, matches)

        print("[v2] pass 3: render from saved detections (no detector/ReID rerun)")
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
        embed_observations = 0

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
                    q = quality(person) if person is not None else 0.0
                    if person is None or q < self.min_quality:
                        continue
                    crops.append(person)
                    if self.use_light:
                        crops.append(illumination_variant(person))
                    meta.append((key, track_id, segment, bbox, q, float(item.confidence)))
                    last_embed[key] = frame_index

                if crops:
                    features = self.extractor.extract_batch(crops)
                    pos = 0
                    for key, track_id, segment, bbox, q, det_score in meta:
                        original = features[pos]
                        pos += 1
                        tracklet = self.tracklets.get(key)
                        if tracklet is None:
                            tracklet = Tracklet(camera=camera, track_id=track_id, segment=segment, fps=fps)
                            self.tracklets[key] = tracklet
                        meta_row = {
                            "camera": camera,
                            "frame": frame_index,
                            "timestamp": frame_index / fps,
                            "track_id": track_id,
                            "bbox": list(bbox),
                            "detection_score": det_score,
                            "quality": q,
                        }
                        tracklet.add(original, q, meta_row)
                        vectors.append(original.astype(np.float16))
                        embed_observations += 1

                        if self.use_light:
                            light = features[pos]
                            pos += 1
                            agreement = float(unit(original) @ unit(light))
                            if agreement < 0.995:
                                tracklet.add(light, max(0.10, q * 0.92), {**meta_row, "variant": "illumination"})
                                vectors.append(light.astype(np.float16))
                                embed_observations += 1
        finally:
            cap.release()
            events.close()

        matrix = np.stack(vectors) if vectors else np.empty((0, self.extractor.embedding_dim), dtype=np.float16)
        np.save(embeddings_path, matrix)
        camera_count = sum(1 for key in self.tracklets if key.startswith(camera + ":"))
        camera_embeds = sum(track.count for key, track in self.tracklets.items() if key.startswith(camera + ":"))
        print(f"[v2] {camera}: frames={frame_index} tracklets={camera_count} embeddings={camera_embeds} sampled={embed_observations}")

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
                "bbox_aspect": track.shape,
            })
            if track.embeddings:
                packed[key] = np.stack(track.embeddings).astype(np.float32)
        (self.cache / "tracklets_v2.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        np.savez_compressed(self.cache / "tracklet_embeddings_v2.npz", **packed)
        (self.cache / "video_meta_v2.json").write_text(json.dumps(self.video_meta, indent=2), encoding="utf-8")

    def build_matched_camera_map(self, matches) -> Dict[str, str]:
        """Map only confirmed cross-camera matches to their supporting camera.

        A tracklet gets a label only when the reconciler produced an explicit
        cross-camera MERGE for it. This prevents newly created identities or
        same-camera-only associations from being rendered as MATCHED CAM.
        """
        links: Dict[str, List[Tuple[float, float, str]]] = {}
        for match in matches:
            if match.relation != "cross":
                continue
            if match.left not in self.tracklets or match.right not in self.tracklets:
                continue
            left = self.tracklets[match.left]
            right = self.tracklets[match.right]
            links.setdefault(match.left, []).append((float(match.score), right.end, right.camera))
            links.setdefault(match.right, []).append((float(match.score), left.end, left.camera))

        result: Dict[str, str] = {}
        for key, values in links.items():
            # Prefer the strongest confirmed cross-camera association; break ties
            # using the most recent supporting observation.
            values.sort(key=lambda item: (item[0], item[1]), reverse=True)
            result[key] = values[0][2]
        return result

    def save_mapping(self, mapping: Dict[str, str], matches) -> None:
        (self.out / "track_to_global_v2.json").write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
        rows = []
        for match in matches:
            rows.append({
                "left": match.left,
                "right": match.right,
                "left_global": mapping[match.left],
                "right_global": mapping[match.right],
                "left_camera": self.tracklets[match.left].camera,
                "right_camera": self.tracklets[match.right].camera,
                "score": match.score,
                "margin_left": match.margin_left,
                "margin_right": match.margin_right,
                "reciprocal": match.reciprocal,
                "relation": match.relation,
                "decision": "MERGE",
            })
        with (self.out / "identity_matches_v2.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

        matched_meta = {
            key: {
                "global_id": mapping[key],
                "matched_camera": camera,
            }
            for key, camera in self.matched_camera_by_tracklet.items()
            if key in mapping
        }
        (self.out / "matched_cameras_v2.json").write_text(json.dumps(matched_meta, indent=2, sort_keys=True), encoding="utf-8")

    def save_gallery(self, mapping: Dict[str, str]) -> None:
        galleries = self.engine.gallery_for_mapping(self.tracklets, mapping, self.bank_size)
        np.savez_compressed(self.out / "global_gallery_v2.npz", **galleries)
        meta = {
            gid: {
                "members": [key for key, value in mapping.items() if value == gid],
                "prototype_count": int(len(gallery)),
                "embedding_dim": int(gallery.shape[1]),
            }
            for gid, gallery in galleries.items()
        }
        (self.out / "global_gallery_v2.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def print_diagnostics(self, mapping: Dict[str, str], matches) -> None:
        stats = self.engine.summarize_scores(self.tracklets)
        same = sum(match.relation == "same" for match in matches)
        cross = len(matches) - same
        print("[v2] score space:", json.dumps(stats, sort_keys=True))
        print(f"[v2] accepted associations: {len(matches)}")
        print(f"[v2] same-camera merges: {same}")
        print(f"[v2] cross-camera merges: {cross}")
        print(f"[v2] matched-camera labels: {len(self.matched_camera_by_tracklet)}")
        for match in matches[:30]:
            print(f"  {match.relation}: {match.left} <-> {match.right} score={match.score:.4f} margins={match.margin_left:.4f}/{match.margin_right:.4f}")

    def render(self, mapping: Dict[str, str]) -> None:
        for camera, meta in self.video_meta.items():
            cap = cv2.VideoCapture(meta["source"])
            output = self.out / f"{camera}_v2.mp4"
            writer = cv2.VideoWriter(
                str(output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                meta["fps"],
                (meta["width"], meta["height"]),
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
                        key = record["tracklet_key"]
                        gid = mapping[key]
                        x1, y1, x2, y2 = [int(v) for v in record["bbox"]]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        label_y = max(25, y1 - 8)
                        cv2.putText(
                            frame,
                            gid,
                            (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                        matched_camera = self.matched_camera_by_tracklet.get(key)
                        if matched_camera:
                            display_camera = matched_camera[4:] if matched_camera.startswith("cam_") else matched_camera
                            cv2.putText(
                                frame,
                                f"MATCHED CAM {display_camera}",
                                (x1, label_y + 24),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.58,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                    cv2.putText(frame, camera, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    writer.write(frame)
            finally:
                cap.release()
                writer.release()
            print(f"[v2] wrote {output}")

    def summary(self, mapping: Dict[str, str], matches) -> None:
        groups: Dict[str, set] = {}
        for key, gid in mapping.items():
            groups.setdefault(gid, set()).add(self.tracklets[key].camera)
        multi = {gid: cameras for gid, cameras in groups.items() if len(cameras) > 1}
        same = sum(match.relation == "same" for match in matches)
        cross = len(matches) - same
        print("\n===== V2 REID RESULT =====")
        print(f"tracklets: {len(mapping)}")
        print(f"global IDs: {len(set(mapping.values()))}")
        print(f"same-camera merges: {same}")
        print(f"cross-camera merges: {cross}")
        print(f"multi-camera IDs: {len(multi)}")
        for gid, cameras in sorted(multi.items()):
            print(f"  {gid}: {', '.join(sorted(cameras))}")
        print(f"outputs: {self.out}")
