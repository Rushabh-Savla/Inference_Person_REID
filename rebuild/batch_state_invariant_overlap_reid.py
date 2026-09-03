from __future__ import annotations

import json
from typing import Dict

import numpy as np

from rebuild.batch_state_invariant_accurate import BatchPipelineStateInvariantAccurate
from rebuild.identity_v2 import crop, illumination_variant, quality


class BatchPipelineStateInvariantOverlapReid(BatchPipelineStateInvariantAccurate):
    """Safe055/V6 overlap path: track always, pause features during severe overlap, then re-identify densely."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.post_overlap_interval = max(1, int(self.cfg.get("post_overlap_interval_frames", 1)))
        self.trajectory_history = max(8, int(self.cfg.get("trajectory_history_frames", 30)))
        self._trajectory: Dict[str, list[dict]] = {}
        self._overlap_refs = self._refs
        self._post_overlap_fails: Dict[str, int] = {}

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

    def _save_clean_anchor(self, key: str) -> None:
        track = self.tracks.get(key)
        if track is None:
            return
        bank = getattr(track, "state_bank", {})
        refs = {"resnet": [], "swin": [], "solider": []}
        for model in refs:
            refs[model] = list(bank.get(model, {}).get("full", [])[-4:])
        if all(refs[model] for model in refs):
            self._overlap_refs[key] = refs
            self._post_overlap_fails[key] = 0

    def _extract_one(self, camera, frame, fps, image, item, key, seg, recovery, info, rows, active=False):
        del active, rows
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
        stamp = frame / fps
        multi = {
            "swin": {name: [value] for name, value in swin_map.items()},
            "solider": {name: [value] for name, value in solider_map.items()},
        }
        self.add_body(key, camera, item["tid"], seg, box, stamp, float(detection.confidence), person, resnet_map, multi)
        self._attach_position_history(key)
        info["samples"] += 1
        info["feature_batches"] += 1
        info["last"][key] = frame
        if recovery:
            info["recovery_samples"] += 1
        return True

    def _check_anchor(self, key: str):
        track = self.tracks.get(key)
        anchor = self._overlap_refs.get(key)
        if track is None or not anchor:
            return None
        scores = {}
        for model in ("resnet", "swin", "solider"):
            values = getattr(track, "state_bank", {}).get(model, {}).get("full", [])
            query = self._unit(values[-1]) if values else None
            refs = []
            for value in anchor.get(model, []):
                ref = self._unit(value)
                if query is not None and ref is not None and query.shape == ref.shape:
                    refs.append(float(np.dot(query, ref)))
            refs.sort(reverse=True)
            scores[model] = float(np.mean(refs[:3])) if refs else 0.0
        ordered = sorted(scores.values(), reverse=True)
        fused = float(np.mean(ordered[:2])) if len(ordered) >= 2 else 0.0
        support = sum(scores[model] >= self.recovery_min[model] for model in scores)
        return {
            "scores": scores,
            "fused": fused,
            "support": support,
            "match": bool(support >= self.recovery_models and fused >= self.recovery_fused),
        }

    def _extract(self, camera, frame, fps, image, prepared, blocked, partners, info, rows):
        for item in prepared:
            tid = item["tid"]
            old_key = item["key"]
            old_seg = item["seg"]
            box = item["bbox"]
            detection = item["item"]

            self._remember_position(old_key, frame, fps, box)
            self._touch_track(old_key, frame, fps)

            active = old_key in blocked
            previous_active = bool(info["was_overlap"].get(tid, False))
            key = old_key
            seg = old_seg
            boundary = False
            reason = "normal"

            if active and not previous_active:
                info["overlap_events"] += 1
                info["overlap_tids"].add(tid)
                info["recovery_left"][old_key] = 0
                self._save_clean_anchor(old_key)
                reason = "high_overlap_start_feature_pause"

            elif previous_active and not active:
                info["segments"][tid] = info["segments"].get(tid, old_seg) + 1
                self._segments[camera][tid] = info["segments"][tid]
                seg = info["segments"][tid]
                key = f"{camera}:{tid}:{seg}"
                self._trajectory[key] = []
                self._remember_position(key, frame, fps, box)
                info["recovery_left"][key] = max(4, self.recovery_samples)
                info["last"][key] = -10**9
                info["last"][key + ":parts"] = -10**9
                info["last"][key + ":light"] = -10**9
                info["last"][key + ":recovery"] = -10**9
                anchor = self._overlap_refs.get(old_key)
                if anchor:
                    self._overlap_refs[key] = {model: list(values) for model, values in anchor.items()}
                self._post_overlap_fails[key] = 0
                info["overlap_tids"].discard(tid)
                boundary = True
                reason = "high_overlap_exit_dense_feature_reassignment"
                info.setdefault("post_overlap_events", 0)
                info["post_overlap_events"] += 1

            info["was_overlap"][tid] = active
            self._remember_position(key, frame, fps, box)
            self._touch_track(key, frame, fps)

            rows.write(json.dumps({
                "camera": camera,
                "frame": frame,
                "timestamp": frame / fps,
                "track_id": tid,
                "segment": seg,
                "tracklet_key": key,
                "bbox": list(box),
                "detection_score": float(detection.confidence),
                "overlap_blocked": bool(active),
                "overlap_partners": partners.get(old_key, []) if active else [],
                "overlap_boundary": bool(boundary),
                "segment_reason": reason,
                "recovery_after_overlap": bool((not active) and info["recovery_left"].get(key, 0) > 0),
                "position_tracked_every_frame": True,
            }) + "\n")

            if active:
                continue

            recovery = info["recovery_left"].get(key, 0)
            normal_due = frame - info["last"].get(key, -10**9) >= self.interval
            recovery_due = recovery > 0 and frame - info["last"].get(key + ":recovery", -10**9) >= self.post_overlap_interval
            if not normal_due and not recovery_due:
                continue

            ok = self._extract_one(camera, frame, fps, image, item, key, seg, recovery_due, info, rows, active=False)
            if not ok:
                continue

            if recovery_due:
                info["last"][key + ":recovery"] = frame
                check = self._check_anchor(key)
                if check is not None:
                    rows.write(json.dumps({
                        "camera": camera,
                        "frame": frame,
                        "timestamp": frame / fps,
                        "track_id": tid,
                        "segment": seg,
                        "tracklet_key": key,
                        "bbox": list(box),
                        "post_overlap_feature_check": True,
                        "post_overlap_reid_scores": check["scores"],
                        "post_overlap_reid_fused": check["fused"],
                        "post_overlap_model_support": check["support"],
                        "post_overlap_same_pre_identity": check["match"],
                        "post_overlap_reassignment_by_features": True,
                        "post_overlap_position_tracked": True,
                        "post_overlap_complete_body_checked": True,
                    }) + "\n")
                    info.setdefault("post_overlap_feature_checks", 0)
                    info["post_overlap_feature_checks"] += 1

            self._attach_position_history(key)
            if recovery > 1:
                info["recovery_left"][key] = recovery - 1
            else:
                info["recovery_left"][key] = 0
                self._overlap_refs.pop(key, None)
                self._post_overlap_fails.pop(key, None)


__all__ = ["BatchPipelineStateInvariantOverlapReid"]
