from __future__ import annotations

import json

import numpy as np

from rebuild.batch_state_invariant_accurate import BatchPipelineStateInvariantAccurate


class BatchPipelineStateInvariantOverlapReid(BatchPipelineStateInvariantAccurate):
    """V6 overlap handling: track continuously, pause ReID, recover densely."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.post_overlap_interval = max(1, int(self.cfg.get("post_overlap_interval_frames", 1)))
        self.trajectory_history = max(8, int(self.cfg.get("trajectory_history_frames", 30)))

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
        support = sum(
            scores[model] >= threshold
            for model, threshold in {
                "resnet": self.recovery_min["resnet"],
                "swin": self.recovery_min["swin"],
                "solider": self.recovery_min["solider"],
            }.items()
        )
        return {
            "scores": scores,
            "fused": fused,
            "support": support,
            "match": bool(support >= 2 and fused >= self.recovery_fused),
        }

    def _extract(self, camera, frame, fps, image, prepared, blocked, partners, info, rows):
        for item in prepared:
            tid = item["tid"]
            old_key = item["key"]
            old_seg = item["seg"]
            box = item["bbox"]
            detection = item["item"]

            # ByteTrack stalking/trajectory is updated on every detector result,
            # including the ambiguity window. Feature extraction is separate.
            self._remember_position(old_key, frame, fps, box)
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
                # Keep ByteTrack continuity, but create a new appearance segment.
                # GID reassignment is decided only from the clean post-overlap
                # observations, never from overlap state itself.
                info["segments"][tid] = info["segments"].get(tid, old_seg) + 1
                self._segments[camera][tid] = info["segments"][tid]
                seg = info["segments"][tid]
                key = f"{camera}:{tid}:{seg}"
                self._trajectory[key] = []
                info["recovery_left"][key] = max(4, self.recovery_samples)
                info["last"][key] = -10**9
                info["last"][key + ":parts"] = -10**9
                info["last"][key + ":light"] = -10**9
                info["last"][key + ":recovery"] = -10**9
                anchor = self._overlap_refs.get(old_key)
                if anchor:
                    self._overlap_refs[key] = anchor
                info["overlap_tids"].discard(tid)
                boundary = True
                reason = "high_overlap_exit_dense_feature_reassignment"
                info.setdefault("post_overlap_events", 0)
                info["post_overlap_events"] += 1

            info["was_overlap"][tid] = active
            if key != old_key:
                self._remember_position(key, frame, fps, box)

            rows.write(
                json.dumps(
                    {
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
                    }
                )
                + "\n"
            )

            # HARD RULE: severe overlap => tracking only, ZERO feature extraction.
            if active:
                continue

            recovery = info["recovery_left"].get(key, 0)
            normal_due = frame - info["last"].get(key, -10**9) >= self.interval
            recovery_due = (
                recovery > 0
                and frame - info["last"].get(key + ":recovery", -10**9) >= self.post_overlap_interval
            )
            if not normal_due and not recovery_due:
                continue

            # Four clean frames after overlap are sampled densely. Each sample
            # gets full + available parts/light through all three ReID models.
            ok = self._extract_one(
                camera,
                frame,
                fps,
                image,
                item,
                key,
                seg,
                recovery_due,
                info,
                rows,
                active=False,
            )

            if recovery_due:
                info["last"][key + ":recovery"] = frame
                if ok:
                    check = self._check_anchor(key)
                    if check is not None:
                        rows.write(
                            json.dumps(
                                {
                                    "camera": camera,
                                    "frame": frame,
                                    "timestamp": frame / fps,
                                    "track_id": tid,
                                    "segment": seg,
                                    "tracklet_key": key,
                                    "post_overlap_feature_check": True,
                                    "post_overlap_reid_scores": check["scores"],
                                    "post_overlap_reid_fused": check["fused"],
                                    "post_overlap_model_support": check["support"],
                                    "post_overlap_same_pre_identity": check["match"],
                                    "post_overlap_reassignment_by_features": True,
                                    "post_overlap_position_tracked": True,
                                }
                            )
                            + "\n"
                        )
                        info.setdefault("post_overlap_feature_checks", 0)
                        info["post_overlap_feature_checks"] += 1

            self._attach_position_history(key)
            if recovery <= 1 and info["recovery_left"].get(key, 0) == 0:
                self._overlap_refs.pop(key, None)


__all__ = ["BatchPipelineStateInvariantOverlapReid"]
