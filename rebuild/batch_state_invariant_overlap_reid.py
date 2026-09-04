from __future__ import annotations

import json
from typing import Dict

import numpy as np

from rebuild.batch_state_invariant_accurate import BatchPipelineStateInvariantAccurate
from rebuild.face_v4 import FaceExtractorV4
from rebuild.identity_v2 import crop, illumination_variant, quality
from rebuild.overlap_recovery import OverlapEpisode, bbox_overlap_metrics, is_severe_overlap, participant_anchor, recovery_sources


class BatchPipelineStateInvariantOverlapReid(BatchPipelineStateInvariantAccurate):
    """Overlap-aware Safe055/V6 pipeline with tracker-ID-independent recovery."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        guard = self.cfg.get("overlap_guard", {}) or {}
        self.overlap_iou = float(guard.get("iou_min", 0.80))
        self.overlap_intersection = float(guard.get("intersection_min", 0.85))
        self.overlap_clear_grace = max(1, int(guard.get("clear_grace_frames", 2)))
        self.recovery_samples = max(4, int(guard.get("recovery_samples", 6)))
        self.recovery_search_sec = float(guard.get("recovery_search_sec", 1.75))
        self.recovery_spatial_scale = float(guard.get("recovery_spatial_scale", 4.5))
        self.post_overlap_interval = max(1, int(self.cfg.get("post_overlap_interval_frames", 1)))
        self.trajectory_history = max(8, int(self.cfg.get("trajectory_history_frames", 30)))
        self._trajectory: Dict[str, list[dict]] = {}
        self._episodes: Dict[str, list[OverlapEpisode]] = {}
        self._episode_active: Dict[str, OverlapEpisode | None] = {}
        self._pair_metrics: Dict[tuple[str, str], dict] = {}
        self._recovery_meta: Dict[str, dict] = {}
        self.face = None
        face_cfg = self.cfg.get("face", {}) or {}
        if bool(face_cfg.get("enabled", True)):
            try:
                self.face = FaceExtractorV4(
                    model=str(face_cfg.get("model", "buffalo_l")),
                    det_size=tuple(face_cfg.get("det_size", [640, 640])),
                    min_detection=float(face_cfg.get("min_detection", 0.55)),
                    min_size=int(face_cfg.get("min_size", 32)),
                    min_quality=float(face_cfg.get("min_quality", 0.50)),
                    min_visibility=float(face_cfg.get("min_visibility", 0.60)),
                    device=str(face_cfg.get("device", "cuda")),
                )
                print(f"[state-joint] FACE: {self.face.describe()} | visibility gate >= {self.face.min_visibility:.2f}")
            except Exception as exc:
                if bool(face_cfg.get("required", False)):
                    raise
                print(f"[state-joint] FACE: unavailable, body-only mode ({exc})")

    @staticmethod
    def _unit(value):
        arr = np.asarray(value, np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if arr.size == 0 or not np.isfinite(norm) or norm <= 0.0:
            return None
        return arr / norm

    def _remember_position(self, key: str, frame: int, fps: float, box) -> None:
        x1, y1, x2, y2 = [float(v) for v in box]
        row = {
            "frame": int(frame),
            "timestamp": float(frame / fps),
            "bbox": [x1, y1, x2, y2],
            "center": [0.5 * (x1 + x2), 0.5 * (y1 + y2)],
            "height": max(1.0, y2 - y1),
        }
        history = self._trajectory.setdefault(key, [])
        history.append(row)
        if len(history) > self.trajectory_history:
            del history[:-self.trajectory_history]

    def _attach_position_history(self, key: str) -> None:
        track = self.tracks.get(key)
        history = self._trajectory.get(key)
        if track is not None and history:
            setattr(track, "trajectory", list(history))

    def _touch_track(self, key: str, frame: int, fps: float) -> None:
        track = self.tracks.get(key)
        history = self._trajectory.get(key)
        if track is None or not history:
            return
        stamp = float(frame / fps)
        track.start = min(float(getattr(track, "start", stamp)), stamp)
        track.end = max(float(getattr(track, "end", stamp)), stamp)
        setattr(track, "trajectory", list(history))

    def _overlaps(self, items):
        blocked: set[str] = set()
        partners: dict[str, list[str]] = {}
        self._pair_metrics = {}
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                left = items[i]
                right = items[j]
                metrics = bbox_overlap_metrics(left["bbox"], right["bbox"])
                a = left["key"]
                b = right["key"]
                self._pair_metrics[tuple(sorted((a, b)))] = dict(metrics)
                if not is_severe_overlap(metrics, self.overlap_iou, self.overlap_intersection):
                    continue
                blocked.add(a)
                blocked.add(b)
                partners.setdefault(a, []).append(b)
                partners.setdefault(b, []).append(a)
        return blocked, partners

    def _anchor_for(self, key: str) -> bool:
        return participant_anchor(self.tracks.get(key))

    def _update_overlap_episode(self, camera: str, frame: int, fps: float, prepared, blocked) -> OverlapEpisode | None:
        episodes = self._episodes.setdefault(camera, [])
        active = self._episode_active.get(camera)
        if blocked:
            if active is None or active.closed:
                active = OverlapEpisode(camera, frame, frame)
                episodes.append(active)
                self._episode_active[camera] = active
            active.block(frame)
            for item in prepared:
                if item["key"] not in blocked:
                    continue
                key = item["key"]
                active.touch(key, item["bbox"], frame, self._anchor_for(key))
            return None

        if active is not None and not active.closed:
            if active.clear(frame, self.overlap_clear_grace):
                self._episode_active[camera] = None
                return active
        return None

    def _find_recovery_sources(self, camera: str, box, frame: int, fps: float) -> list[str]:
        episodes = self._episodes.get(camera, [])
        sources = recovery_sources(
            box,
            frame,
            fps,
            episodes,
            self.recovery_spatial_scale,
            self.recovery_search_sec,
        )
        cutoff = float(frame / max(fps, 1.0)) - max(self.recovery_search_sec, 0.5) * 2.0
        self._episodes[camera] = [
            episode for episode in episodes
            if episode.exit_frame is None or episode.exit_frame / max(fps, 1.0) >= cutoff
        ]
        return sources

    def _set_recovery_meta(self, key: str, sources: list[str], camera: str) -> None:
        if not sources:
            return
        self._recovery_meta[key] = {
            "camera": camera,
            "sources": list(sources),
            "recovery": True,
        }
        track = self.tracks.get(key)
        if track is not None:
            setattr(track, "overlap_recovery", True)
            setattr(track, "recovery_sources", list(sources))

    def _store_face(self, key: str, image) -> None:
        if self.face is None:
            return
        observation = self.face.extract(image)
        if observation is None:
            return
        track = self.tracks.get(key)
        if track is None:
            return
        bank = list(getattr(track, "face_bank", []) or [])
        bank.append({
            "vector": np.asarray(observation.vector, np.float32),
            "quality": float(observation.quality),
            "visibility": float(observation.visibility),
            "detection": float(observation.detection),
            "width": float(observation.width),
            "height": float(observation.height),
            "area": float(observation.area),
            "roll": float(observation.roll),
            "valid": bool(observation.valid),
        })
        setattr(track, "face_bank", bank[-32:])

    def add_body(self, key, camera, track_id, segment, bbox, stamp, score, image, feats, multi):
        measured = float(quality(image))
        super().add_body(key, camera, track_id, segment, bbox, stamp, measured, image, feats, multi)
        self._store_face(key, image)
        meta = self._recovery_meta.get(key)
        track = self.tracks.get(key)
        if track is not None and meta:
            setattr(track, "overlap_recovery", True)
            setattr(track, "recovery_sources", list(meta["sources"]))
            setattr(track, "recovery_camera", meta["camera"])

    def _extract_one(self, camera, frame, fps, image, item, key, seg, recovery, info, rows):
        box = item["bbox"]
        detection = item["item"]
        person = crop(image, box)
        q = quality(person) if person is not None else 0.0
        if person is None or q < self.min_quality:
            return False

        dense = bool(recovery)
        variants = {"full": person}
        if self.light and (dense or frame - info["last"].get(key + ":light", -10**9) >= self.part_interval):
            variants["light"] = illumination_variant(person)
            info["last"][key + ":light"] = frame
        if dense or frame - info["last"].get(key + ":parts", -10**9) >= self.part_interval:
            variants.update(self.parts(person))
            info["last"][key + ":parts"] = frame

        ordered = list(variants)
        crops = [variants[name] for name in ordered]
        resnet = self.extractor.extract_batch(crops)
        swin = self.swin.extract_batch(crops)
        solider = self.solider.extract_batch(crops)
        self._check(resnet, "NVIDIA ResNet", len(crops))
        self._check(swin, "NVIDIA Swin", len(crops))
        self._check(solider, "SOLIDER", len(crops))

        resnet_map = {name: value for name, value in zip(ordered, resnet)}
        swin_map = {name: value for name, value in zip(ordered, swin)}
        solider_map = {name: value for name, value in zip(ordered, solider)}
        self._frame_image = image
        multi = {
            "swin": {name: [value] for name, value in swin_map.items()},
            "solider": {name: [value] for name, value in solider_map.items()},
        }
        self.add_body(
            key,
            camera,
            item["tid"],
            seg,
            box,
            frame / fps,
            float(detection.confidence),
            person,
            resnet_map,
            multi,
        )
        self._attach_position_history(key)
        info["samples"] += 1
        info["feature_batches"] += 1
        info["last"][key] = frame
        if recovery:
            info["recovery_samples"] += 1
        return True

    def _extract(self, camera, frame, fps, image, prepared, blocked, partners, info, rows):
        self._update_overlap_episode(camera, frame, fps, prepared, blocked)
        for item in prepared:
            tid = item["tid"]
            old_key = item["key"]
            box = item["bbox"]
            seg = item["seg"]
            self._remember_position(old_key, frame, fps, box)
            self._touch_track(old_key, frame, fps)

            active = old_key in blocked
            reason = "normal"
            sources: list[str] = []
            if active:
                reason = "severe_overlap_feature_pause"
            else:
                sources = self._find_recovery_sources(camera, box, frame, fps)
                if sources:
                    self._set_recovery_meta(old_key, sources, camera)
                    info["recovery_left"][old_key] = max(
                        info["recovery_left"].get(old_key, 0), self.recovery_samples
                    )
                    info["last"][old_key + ":recovery"] = -10**9
                    reason = "post_overlap_feature_recovery"

            info["was_overlap"][tid] = active
            rows.write(json.dumps({
                "camera": camera,
                "frame": frame,
                "timestamp": frame / fps,
                "track_id": tid,
                "segment": seg,
                "tracklet_key": old_key,
                "bbox": list(box),
                "detection_score": float(item["item"].confidence),
                "overlap_blocked": bool(active),
                "overlap_partners": partners.get(old_key, []) if active else [],
                "recovery_after_overlap": bool(sources),
                "recovery_sources": list(sources),
                "segment_reason": reason,
                "position_tracked_every_frame": True,
            }) + "\n")

            # Severe overlap is a tracking-only period. No body, attribute or
            # face features from the occluded mixture enter any identity bank.
            if active:
                continue

            recovery = info["recovery_left"].get(old_key, 0)
            normal_due = frame - info["last"].get(old_key, -10**9) >= self.interval
            recovery_due = recovery > 0 and frame - info["last"].get(old_key + ":recovery", -10**9) >= self.post_overlap_interval
            if not normal_due and not recovery_due:
                continue

            ok = self._extract_one(
                camera,
                frame,
                fps,
                image,
                item,
                old_key,
                seg,
                bool(recovery_due),
                info,
                rows,
            )
            if not ok:
                continue
            if recovery_due:
                info["last"][old_key + ":recovery"] = frame
                info["recovery_left"][old_key] = max(0, recovery - 1)
            self._attach_position_history(old_key)

    __all__ = ["BatchPipelineStateInvariantOverlapReid"]
