from __future__ import annotations

from typing import Dict

import numpy as np

from rebuild.batch_state_invariant_joint import BatchPipelineStateInvariantJoint
from rebuild.identity_v2 import crop, illumination_variant, quality


class BatchPipelineStateInvariantJointGuarded(BatchPipelineStateInvariantJoint):
    """Joint MTMC V6 with conservative high-overlap and recovery guards.

    The established V6 architecture and joint multi-camera scheduling remain
    unchanged. This layer only changes when feature extraction is suppressed
    and adds a strict post-overlap consistency gate so a tracker swap cannot
    contaminate the new segment's state bank.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        guard = self.cfg.get("overlap_guard", {}) or {}
        self.overlap_iou = float(guard.get("iou_min", 0.80))
        self.overlap_intersection = float(guard.get("intersection_min", 0.85))
        self.recovery_samples = max(1, int(guard.get("recovery_samples", 3)))
        recovery = self.cfg.get("recovery_guard", {}) or {}
        self.recovery_models = max(2, int(recovery.get("required_models", 2)))
        self.recovery_fused = float(recovery.get("fused_min", 0.48))
        self.recovery_model_min = {
            "resnet": float(recovery.get("resnet_min", 0.44)),
            "swin": float(recovery.get("swin_min", 0.44)),
            "solider": float(recovery.get("solider_min", 0.42)),
        }
        self._recovery_refs: Dict[str, Dict[str, list[np.ndarray]]] = {}

    def _overlaps(self, items):
        """Block only severe physical overlap, not ordinary close walking.

        `intersection / smaller_box_area` captures containment, so a person
        nearly completely inside another detection is still treated as an
        ambiguous overlap. Small edge intersections are deliberately allowed.
        """
        blocked: set[str] = set()
        partners: dict[str, list[str]] = {}
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                left = items[i]
                right = items[j]
                if not self._is_overlap(left["bbox"], right["bbox"]):
                    continue
                a = left["key"]
                b = right["key"]
                blocked.add(a)
                blocked.add(b)
                partners.setdefault(a, []).append(b)
                partners.setdefault(b, []).append(a)
        return blocked, partners

    @staticmethod
    def _unit(value):
        arr = np.asarray(value, np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if arr.size == 0 or not np.isfinite(norm) or norm <= 0.0:
            return None
        return arr / norm

    def _recovery_ok(self, key: str, vectors: Dict[str, np.ndarray]) -> tuple[bool, Dict[str, float]]:
        refs = self._recovery_refs.get(key, {})
        scores: Dict[str, float] = {}
        for model in ("resnet", "swin", "solider"):
            query = self._unit(vectors[model])
            bank = refs.get(model, [])
            values = []
            if query is not None:
                for value in bank[-4:]:
                    ref = self._unit(value)
                    if ref is not None and ref.shape == query.shape:
                        values.append(float(np.dot(query, ref)))
            scores[model] = max(values) if values else 0.0

        support = sum(
            scores[model] >= self.recovery_model_min[model]
            for model in scores
        )
        ordered = sorted(scores.values(), reverse=True)
        fused = float(np.mean(ordered[:2])) if len(ordered) >= 2 else 0.0
        return support >= self.recovery_models and fused >= self.recovery_fused, scores

    def _remember_recovery(self, key: str, vectors: Dict[str, np.ndarray]) -> None:
        refs = self._recovery_refs.setdefault(
            key, {"resnet": [], "swin": [], "solider": []}
        )
        for model in refs:
            refs[model].append(np.asarray(vectors[model], np.float32))
            refs[model] = refs[model][-4:]

    def _extract(
        self,
        camera: str,
        frame: int,
        fps: float,
        image: np.ndarray,
        prepared,
        blocked,
        partners,
        last,
        was_overlap,
        recovery_left,
        stats,
        rows,
    ):
        for item in prepared:
            tid = item["tid"]
            box = item["bbox"]
            detection = item["item"]
            old_key = item["key"]
            active = old_key in blocked
            previous_active = bool(was_overlap.get(tid, False))
            key = old_key
            seg = item["seg"]
            boundary = False
            reason = "normal"

            if active and not previous_active:
                stats["overlap_events"] += 1
                stats["overlap_tids"].add(tid)
                recovery_left[key] = 0
                reason = "high_overlap_start"
            elif previous_active and not active:
                old_track = self.tracks.get(old_key)
                refs = {"resnet": [], "swin": [], "solider": []}
                if old_track is not None:
                    bank = getattr(old_track, "state_bank", {})
                    for model in refs:
                        refs[model] = list(bank.get(model, {}).get("full", [])[-4:])
                self._overlap_exit_segment(tid, camera, old_key, seg, recovery_left, last)
                seg = self._segments[camera][tid]
                key = f"{camera}:{tid}:{seg}"
                self._recovery_refs[key] = refs
                recovery_left[key] = self.recovery_samples
                last[key] = -10**9
                boundary = True
                reason = "high_overlap_exit_new_segment"
                stats["overlap_tids"].discard(tid)

            was_overlap[tid] = active
            partner_keys = partners.get(old_key, []) if active else []
            rows.write(
                __import__("json").dumps(
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
                        "overlap_partners": partner_keys,
                        "overlap_boundary": bool(boundary),
                        "segment_reason": reason,
                        "recovery_after_overlap": bool(
                            (not active) and recovery_left.get(key, 0) > 0
                        ),
                    }
                )
                + "\n"
            )

            if active:
                continue

            recovery = recovery_left.get(key, 0)
            due = recovery > 0
            if not due and frame - last.get(key, -10**9) < self.interval:
                continue

            person = crop(image, box)
            q = quality(person) if person is not None else 0.0
            if person is None or q < self.min_quality:
                continue

            variants: Dict[str, np.ndarray] = {"full": person}
            if self.light and (
                due or frame - last.get(key + ":light", -10**9) >= self.part_interval
            ):
                variants["light"] = illumination_variant(person)
                last[key + ":light"] = frame
            if due or frame - last.get(key + ":parts", -10**9) >= self.part_interval:
                variants.update(self.parts(person))
                last[key + ":parts"] = frame

            ordered = list(variants.keys())
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

            if due and self._recovery_refs.get(key):
                vectors = {
                    "resnet": np.asarray(resnet_map["full"], np.float32),
                    "swin": np.asarray(swin_map["full"], np.float32),
                    "solider": np.asarray(solider_map["full"], np.float32),
                }
                accepted, scores = self._recovery_ok(key, vectors)
                if not accepted:
                    stats.setdefault("recovery_rejected", 0)
                    stats["recovery_rejected"] += 1
                    rows.write(
                        __import__("json").dumps(
                            {
                                "camera": camera,
                                "frame": frame,
                                "timestamp": frame / fps,
                                "track_id": tid,
                                "segment": seg,
                                "tracklet_key": key,
                                "bbox": list(box),
                                "detection_score": float(detection.confidence),
                                "overlap_blocked": False,
                                "recovery_after_overlap": True,
                                "recovery_rejected": True,
                                "recovery_scores": scores,
                            }
                        )
                        + "\n"
                    )
                    recovery_left[key] = max(0, recovery - 1)
                    if recovery_left[key] == 0:
                        self._recovery_refs.pop(key, None)
                    continue
                self._remember_recovery(key, vectors)
                stats.setdefault("recovery_accepted", 0)
                stats["recovery_accepted"] += 1

            last[key] = frame
            multi = {
                "swin": {name: [value] for name, value in swin_map.items()},
                "solider": {name: [value] for name, value in solider_map.items()},
            }
            self.add_body(
                key,
                camera,
                tid,
                seg,
                box,
                frame / fps,
                float(detection.confidence),
                person,
                resnet_map,
                multi,
            )

            stats["samples"] += 1
            stats["feature_batches"] += 1
            if due:
                stats["recovery_samples"] += 1
                recovery_left[key] = max(0, recovery - 1)
                if recovery_left[key] == 0:
                    self._recovery_refs.pop(key, None)
