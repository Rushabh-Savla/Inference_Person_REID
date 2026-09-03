from __future__ import annotations

import json
from typing import Dict

import numpy as np

from rebuild.batch_state_invariant_joint import BatchPipelineStateInvariantJoint
from rebuild.batch_state_invariant_joint_guarded import HighConfidenceIdentityBodyV6
from rebuild.identity_v2 import crop, illumination_variant, quality
from rebuild.person_attributes import pack


class BatchPipelineStateInvariantJointAttributes(BatchPipelineStateInvariantJoint):
    """Joint V6 MTMC with conservative attribute and overlap recovery guards."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        guard = self.cfg.get("overlap_guard", {}) or {}
        self.overlap_iou = float(guard.get("iou_min", 0.80))
        self.overlap_intersection = float(guard.get("intersection_min", 0.85))
        self.recovery_samples = max(1, int(guard.get("recovery_samples", 4)))
        recovery = self.cfg.get("recovery_guard", {}) or {}
        self.recovery_models = max(2, int(recovery.get("required_models", 2)))
        self.recovery_fused = float(recovery.get("fused_min", 0.52))
        self.recovery_min = {
            "resnet": float(recovery.get("resnet_min", 0.48)),
            "swin": float(recovery.get("swin_min", 0.48)),
            "solider": float(recovery.get("solider_min", 0.46)),
        }
        self.recovery_fails = max(2, int(recovery.get("fail_limit", 4)))
        self._refs: Dict[str, Dict[str, list[np.ndarray]]] = {}
        self._fails: Dict[str, int] = {}
        self._frame_image = None

    def _overlaps(self, items):
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

    def _recovery(self, key: str, vectors: Dict[str, np.ndarray]):
        refs = self._refs.get(key, {})
        scores = {}
        for model in ("resnet", "swin", "solider"):
            query = self._unit(vectors[model])
            vals = []
            if query is not None:
                for value in refs.get(model, [])[-4:]:
                    ref = self._unit(value)
                    if ref is not None and ref.shape == query.shape:
                        vals.append(float(np.dot(query, ref)))
            scores[model] = max(vals) if vals else 0.0
        support = sum(scores[name] >= self.recovery_min[name] for name in scores)
        ordered = sorted(scores.values(), reverse=True)
        fused = float(np.mean(ordered[:2])) if len(ordered) >= 2 else 0.0
        return support >= self.recovery_models and fused >= self.recovery_fused, scores

    def _remember(self, key: str, vectors: Dict[str, np.ndarray]):
        refs = self._refs.setdefault(key, {"resnet": [], "swin": [], "solider": []})
        for model in refs:
            refs[model].append(np.asarray(vectors[model], np.float32))
            refs[model] = refs[model][-4:]

    def add_body(self, key, camera, track_id, segment, bbox, stamp, score, image, feats, multi):
        measured = float(quality(image))
        super().add_body(
            key,
            camera,
            track_id,
            segment,
            bbox,
            stamp,
            measured,
            image,
            feats,
            multi,
        )
        track = self.tracks[key]
        bank = getattr(track, "state_bank", None)
        if bank is None:
            return
        bank.setdefault("resnet", {}).setdefault("attributes", [])
        value = pack(image, self._frame_image, bbox)
        bank["resnet"]["attributes"].append(value)
        bank["resnet"]["attributes"] = bank["resnet"]["attributes"][-32:]

    def _extract(self, camera, frame, fps, image, prepared, blocked, partners, info, rows):
        for item in prepared:
            tid = item["tid"]
            box = item["bbox"]
            detection = item["item"]
            old_key = item["key"]
            active = old_key in blocked
            previous = bool(info["was_overlap"].get(tid, False))
            key = old_key
            seg = item["seg"]
            boundary = False
            reason = "normal"

            if active and not previous:
                info["overlap_events"] += 1
                info["overlap_tids"].add(tid)
                info["recovery_left"][old_key] = 0
                self._fails.pop(old_key, None)
                reason = "high_overlap_start"
            elif previous and not active:
                old = self.tracks.get(old_key)
                refs = {"resnet": [], "swin": [], "solider": []}
                if old is not None:
                    bank = getattr(old, "state_bank", {})
                    for model in refs:
                        refs[model] = list(bank.get(model, {}).get("full", [])[-4:])
                info["segments"][tid] = info["segments"].get(tid, seg) + 1
                self._segments[camera][tid] = info["segments"][tid]
                seg = info["segments"][tid]
                key = f"{camera}:{tid}:{seg}"
                self._refs[key] = refs
                self._fails[key] = 0
                info["recovery_left"][key] = self.recovery_samples
                info["last"][key] = -10**9
                boundary = True
                reason = "high_overlap_exit_new_segment"
                info["overlap_tids"].discard(tid)

            info["was_overlap"][tid] = active
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
            }) + "\n")

            if active:
                continue

            recovery = info["recovery_left"].get(key, 0)
            due = recovery > 0
            if not due and frame - info["last"].get(key, -10**9) < self.interval:
                continue

            person = crop(image, box)
            q = quality(person) if person is not None else 0.0
            if person is None or q < self.min_quality:
                continue

            variants = {"full": person}
            if self.light and (due or frame - info["last"].get(key + ":light", -10**9) >= self.part_interval):
                variants["light"] = illumination_variant(person)
                info["last"][key + ":light"] = frame
            if due or frame - info["last"].get(key + ":parts", -10**9) >= self.part_interval:
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

            if due and self._refs.get(key):
                vectors = {
                    "resnet": np.asarray(resnet_map["full"], np.float32),
                    "swin": np.asarray(swin_map["full"], np.float32),
                    "solider": np.asarray(solider_map["full"], np.float32),
                }
                accepted, scores = self._recovery(key, vectors)
                info["last"][key] = frame
                if not accepted:
                    self._fails[key] = self._fails.get(key, 0) + 1
                    info.setdefault("recovery_rejected", 0)
                    info["recovery_rejected"] += 1
                    rows.write(json.dumps({
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
                        "recovery_fail_count": self._fails[key],
                    }) + "\n")
                    if self._fails[key] >= self.recovery_fails:
                        info["segments"][tid] = info["segments"].get(tid, seg) + 1
                        self._segments[camera][tid] = info["segments"][tid]
                        fresh = f"{camera}:{tid}:{self._segments[camera][tid]}"
                        info["recovery_left"].pop(key, None)
                        self._refs.pop(key, None)
                        self._fails.pop(key, None)
                        info["last"][fresh] = frame
                    continue
                self._remember(key, vectors)
                info.setdefault("recovery_accepted", 0)
                info["recovery_accepted"] += 1

            self._frame_image = image
            info["last"][key] = frame
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
            info["samples"] += 1
            info["feature_batches"] += 1
            if due:
                info["recovery_samples"] += 1
                info["recovery_left"][key] = max(0, recovery - 1)
                if info["recovery_left"][key] == 0:
                    self._refs.pop(key, None)
                    self._fails.pop(key, None)

    def run(self, values):
        import rebuild.identity_body_v6 as body_module
        original = body_module.GlobalIdentityBodyV6
        body_module.GlobalIdentityBodyV6 = HighConfidenceIdentityBodyV6
        try:
            return super().run(values)
        finally:
            body_module.GlobalIdentityBodyV6 = original
