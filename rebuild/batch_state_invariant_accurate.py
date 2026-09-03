from __future__ import annotations

import json
from typing import Dict

import numpy as np

from rebuild.batch_state_invariant_joint_attributes import BatchPipelineStateInvariantJointAttributes
from rebuild.identity_v2 import crop, illumination_variant, quality
from rebuild.person_attributes import pack


class BatchPipelineStateInvariantAccurate(BatchPipelineStateInvariantJointAttributes):
    """Feature-first Safe055/V6 pipeline with stable overlap identity anchors.

    Overlap never creates a segment, never changes a GID, and never decides an
    identity by geometry alone. At every configured observation interval the
    three Re-ID models are evaluated again. During severe overlap those features
    are compared with the clean pre-overlap identity history and only verified
    observations may update the trusted state bank.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.anchor = max(4, self.recovery_samples)
        self._refs: Dict[str, Dict[str, list[np.ndarray]]] = {}
        self._fails: Dict[str, int] = {}
        self._frame_image = None

    @staticmethod
    def _unit(value):
        arr = np.asarray(value, np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if arr.size == 0 or not np.isfinite(norm) or norm <= 0:
            return None
        return arr / norm

    def _save_refs(self, key: str) -> None:
        track = self.tracks.get(key)
        if track is None:
            return
        bank = getattr(track, "state_bank", {})
        refs = {"resnet": [], "swin": [], "solider": []}
        for model in refs:
            refs[model] = list(bank.get(model, {}).get("full", [])[-self.anchor:])
        if any(refs.values()):
            self._refs[key] = refs
            self._fails[key] = 0

    def _score(self, key: str, vectors: Dict[str, np.ndarray], relaxed: bool = False):
        refs = self._refs.get(key, {})
        strong = {
            "resnet": float(self.recovery_min["resnet"]),
            "swin": float(self.recovery_min["swin"]),
            "solider": float(self.recovery_min["solider"]),
        }
        fused_min = float(self.recovery_fused)
        if relaxed:
            strong = {
                "resnet": float(self.recovery_relaxed_min["resnet"]),
                "swin": float(self.recovery_relaxed_min["swin"]),
                "solider": float(self.recovery_relaxed_min["solider"]),
            }
            fused_min = float(self.recovery_relaxed)

        scores = {}
        for model in ("resnet", "swin", "solider"):
            query = self._unit(vectors[model])
            values = []
            if query is not None:
                for value in refs.get(model, [])[-self.anchor:]:
                    ref = self._unit(value)
                    if ref is not None and ref.shape == query.shape:
                        values.append(float(np.dot(query, ref)))
            values.sort(reverse=True)
            scores[model] = float(np.mean(values[: min(3, len(values))])) if values else 0.0

        ordered = sorted(scores.values(), reverse=True)
        fused = float(np.mean(ordered[:2])) if len(ordered) >= 2 else 0.0
        support = sum(scores[name] >= strong[name] for name in scores)
        ok = support >= self.recovery_models and fused >= fused_min
        return bool(ok), scores, fused

    def _remember(self, key: str, vectors: Dict[str, np.ndarray]) -> None:
        refs = self._refs.setdefault(key, {"resnet": [], "swin": [], "solider": []})
        for model in refs:
            refs[model].append(np.asarray(vectors[model], np.float32))
            refs[model] = refs[model][-self.anchor:]

    def _add_verified(
        self,
        key,
        camera,
        tid,
        seg,
        box,
        stamp,
        confidence,
        person,
        resnet_map,
        swin_map,
        solider_map,
    ):
        self._frame_image = self._frame_image if self._frame_image is not None else person
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
            stamp,
            confidence,
            person,
            resnet_map,
            multi,
        )

    def _extract(self, camera, frame, fps, image, prepared, blocked, partners, info, rows):
        for item in prepared:
            tid = item["tid"]
            box = item["bbox"]
            detection = item["item"]
            key = item["key"]
            seg = item["seg"]
            active = key in blocked
            previous = bool(info["was_overlap"].get(tid, False))
            boundary = False
            reason = "normal"

            if active and not previous:
                info["overlap_events"] += 1
                info["overlap_tids"].add(tid)
                self._save_refs(key)
                info["overlap_identity_anchor"] = info.get("overlap_identity_anchor", 0) + 1
                reason = "high_overlap_feature_lock"
            elif previous and not active:
                info["recovery_left"][key] = self.recovery_samples
                info["last"][key] = -10**9
                self._fails[key] = 0
                boundary = True
                reason = "high_overlap_exit_feature_recovery"
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
                "overlap_partners": partners.get(key, []) if active else [],
                "overlap_boundary": bool(boundary),
                "segment_reason": reason,
                "recovery_after_overlap": bool((not active) and info["recovery_left"].get(key, 0) > 0),
            }) + "\n")

            normal_due = frame - info["last"].get(key, -10**9) >= self.interval
            overlap_due = active and frame - info["last"].get(key + ":overlap", -10**9) >= self.interval
            recovery = info["recovery_left"].get(key, 0)
            due = recovery > 0
            if not normal_due and not overlap_due and not due:
                continue

            person = crop(image, box)
            q = quality(person) if person is not None else 0.0
            if person is None or q < self.min_quality:
                continue

            variants = {"full": person}
            if self.light and (active or due or frame - info["last"].get(key + ":light", -10**9) >= self.part_interval):
                variants["light"] = illumination_variant(person)
                info["last"][key + ":light"] = frame
            if active or due or frame - info["last"].get(key + ":parts", -10**9) >= self.part_interval:
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
            vectors = {
                "resnet": np.asarray(resnet_map["full"], np.float32),
                "swin": np.asarray(swin_map["full"], np.float32),
                "solider": np.asarray(solider_map["full"], np.float32),
            }

            self._frame_image = image
            stamp = frame / fps

            if active:
                matched, scores, fused = self._score(key, vectors, relaxed=True)
                info["last"][key + ":overlap"] = frame
                info["last"][key] = max(info["last"].get(key, -10**9), frame)
                info["overlap_feature_checks"] = info.get("overlap_feature_checks", 0) + 1
                rows.write(json.dumps({
                    "camera": camera,
                    "frame": frame,
                    "timestamp": stamp,
                    "track_id": tid,
                    "segment": seg,
                    "tracklet_key": key,
                    "bbox": list(box),
                    "detection_score": float(detection.confidence),
                    "overlap_feature_check": True,
                    "overlap_reid_match": bool(matched),
                    "overlap_reid_fused": float(fused),
                    "overlap_reid_scores": scores,
                }) + "\n")
                # Never change the identity because overlap occurred. A feature
                # verified against the clean anchor may enrich the existing bank;
                # an unverified feature is quarantined and cannot create a GID.
                if matched:
                    self._add_verified(key, camera, tid, seg, box, stamp, float(detection.confidence), person, resnet_map, swin_map, solider_map)
                    self._remember(key, vectors)
                    info["overlap_feature_accepts"] = info.get("overlap_feature_accepts", 0) + 1
                else:
                    info["overlap_feature_rejects"] = info.get("overlap_feature_rejects", 0) + 1
                continue

            if due and self._refs.get(key):
                relaxed = self._fails.get(key, 0) >= self.recovery_fails
                matched, scores, fused = self._score(key, vectors, relaxed=relaxed)
                if not matched:
                    self._fails[key] = self._fails.get(key, 0) + 1
                    info["recovery_rejected"] = info.get("recovery_rejected", 0) + 1
                    info["last"][key] = frame
                    rows.write(json.dumps({
                        "camera": camera,
                        "frame": frame,
                        "timestamp": stamp,
                        "track_id": tid,
                        "segment": seg,
                        "tracklet_key": key,
                        "bbox": list(box),
                        "detection_score": float(detection.confidence),
                        "recovery_after_overlap": True,
                        "recovery_rejected": True,
                        "recovery_relaxed": bool(relaxed),
                        "recovery_scores": scores,
                        "recovery_fused": float(fused),
                        "recovery_fail_count": self._fails[key],
                    }) + "\n")
                    # Keep checking this SAME track/identity. Never make a new
                    # segment merely because appearance is temporarily weak.
                    info["recovery_left"][key] = self.recovery_samples
                    continue
                info["recovery_accepted"] = info.get("recovery_accepted", 0) + 1
                self._remember(key, vectors)
                info["recovery_left"][key] = 0
                self._fails.pop(key, None)

            info["last"][key] = frame
            self._add_verified(key, camera, tid, seg, box, stamp, float(detection.confidence), person, resnet_map, swin_map, solider_map)
            self._remember(key, vectors)
            info["samples"] += 1
            info["feature_batches"] += 1
            if due:
                info["recovery_samples"] += 1

    def add_body(self, key, camera, track_id, segment, bbox, stamp, score, image, feats, multi):
        measured = float(quality(image))
        super().add_body(key, camera, track_id, segment, bbox, stamp, measured, image, feats, multi)
        bank = self.tracks[key].state_bank.setdefault("resnet", {})
        bank.setdefault("attributes", [])
        bank["attributes"].append(pack(image, self._frame_image, bbox))
        bank["attributes"] = bank["attributes"][-64:]
