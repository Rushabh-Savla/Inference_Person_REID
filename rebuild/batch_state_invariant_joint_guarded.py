from __future__ import annotations

from typing import Dict

import numpy as np

from rebuild.batch_state_invariant_joint import BatchPipelineStateInvariantJoint
from rebuild.identity_body_v6 import DecisionBodyV6, GlobalIdentityBodyV6
from rebuild.identity_v2 import crop, illumination_variant, quality


class HighConfidenceIdentityBodyV6(GlobalIdentityBodyV6):
    """V6 local resolver that never promotes weak evidence into a new GID."""

    def accept(self, best: dict, second: float):
        margin = best["score"] - second
        if (
            best["temporal"] >= self.continuity_min
            and best["spatial"] >= 0.35
            and best["body"] >= max(0.56, self.accumulated)
            and best["support"] >= self.support
            and margin >= self.margin
        ):
            return True, "recent_lost_track"
        if best["body"] >= self.strong and (
            margin >= self.margin or best["support"] >= self.support
        ):
            return True, "body_strong"
        if (
            best["body"] >= self.threshold
            and best["support"] >= self.support
            and margin >= self.margin
        ):
            return True, "body_gallery"
        if (
            best["body"] >= self.accumulated
            and best["support"] >= self.accumulated_support
            and margin >= self.margin
        ):
            return True, "body_accumulated"
        if (
            best["partial"]
            and best["body"] >= max(0.60, self.partial)
            and best["support"] >= self.partial_support
            and margin >= self.margin
        ):
            return True, "partial_body"
        return False, "pending"

    def new_identity(self, track, provisional):
        minimum = max(self.promote, 0.68)
        if track.count() < max(self.seed_count, 3) or track.evidence() < minimum:
            return None
        return super().new_identity(track, provisional)

    def run(self, tracks):
        self.identities.clear()
        self.mapping.clear()
        self.provisional.clear()
        self.decisions.clear()
        self.next_id = 1
        self.merge_count = 0
        self.same_camera_reassociated = 0
        self.cross_camera_reidentified = 0
        self.temporal_assisted = 0
        self.body_assisted = 0

        ordered = sorted(
            [x for x in tracks.values() if x.count() > 0],
            key=lambda x: (x.start, x.camera, x.key),
        )
        pending = []
        for track in ordered:
            ranked = self.rank(track, tracks)
            best = ranked[0] if ranked else None
            second = ranked[1]["score"] if len(ranked) > 1 else 0.0
            if best is None:
                pending.append(track)
                self.record(track, "PENDING", "unknown", "pending_evidence", None, f"P{len(pending):04d}")
                continue
            accepted, reason = self.accept(best, second)
            if not accepted:
                pending.append(track)
                self.record(track, "PENDING", "unknown", "pending_evidence", {**best, "second": second}, f"P{len(pending):04d}")
                continue
            gid = best["gid"]
            prior_cameras = set(self.identities[gid].cameras)
            self.mapping[track.key] = gid
            trusted = reason in {"body_strong", "recent_lost_track"} and track.evidence() >= 0.60
            self.add_track(gid, track, trusted)
            self.record(track, gid, "confirmed_existing", reason, {**best, "second": second}, gid)
            self.body_assisted += 1
            if reason == "recent_lost_track":
                self.same_camera_reassociated += 1
                self.temporal_assisted += 1
            if prior_cameras and track.camera not in prior_cameras:
                self.cross_camera_reidentified += 1

        changed = True
        while pending and changed:
            changed = False
            remain = []
            for track in pending:
                ranked = self.rank(track, tracks)
                best = ranked[0] if ranked else None
                second = ranked[1]["score"] if len(ranked) > 1 else 0.0
                if best is None:
                    remain.append(track)
                    continue
                accepted, reason = self.accept(best, second)
                if not accepted:
                    remain.append(track)
                    continue
                gid = best["gid"]
                prior_cameras = set(self.identities[gid].cameras)
                self.mapping[track.key] = gid
                trusted = reason in {"body_strong", "recent_lost_track"} and track.evidence() >= 0.60
                self.add_track(gid, track, trusted)
                self.record(track, gid, "confirmed_existing", reason, {**best, "second": second}, gid)
                self.body_assisted += 1
                changed = True
                if reason == "recent_lost_track":
                    self.same_camera_reassociated += 1
                    self.temporal_assisted += 1
                if prior_cameras and track.camera not in prior_cameras:
                    self.cross_camera_reidentified += 1
            pending = remain

        pending = sorted(pending, key=lambda x: (-x.evidence(), x.start, x.camera, x.key))
        while pending:
            ranked_any = False
            remain = []
            for track in pending:
                ranked = self.rank(track, tracks)
                best = ranked[0] if ranked else None
                second = ranked[1]["score"] if len(ranked) > 1 else 0.0
                if best is not None:
                    accepted, reason = self.accept(best, second)
                    if accepted:
                        gid = best["gid"]
                        prior_cameras = set(self.identities[gid].cameras)
                        self.mapping[track.key] = gid
                        self.add_track(gid, track, reason == "body_strong")
                        self.record(track, gid, "confirmed_existing", reason, {**best, "second": second}, gid)
                        self.body_assisted += 1
                        ranked_any = True
                        if reason == "recent_lost_track":
                            self.same_camera_reassociated += 1
                            self.temporal_assisted += 1
                        if prior_cameras and track.camera not in prior_cameras:
                            self.cross_camera_reidentified += 1
                        continue
                remain.append(track)
            if ranked_any:
                pending = remain
                continue

            seed = remain.pop(0)
            provisional = f"PNEW{self.next_id:04d}"
            gid = self.new_identity(seed, provisional)
            if gid is None:
                self.record(seed, "PENDING", "unknown", "insufficient_new_identity_evidence", None, provisional)
                for track in remain:
                    ranked = self.rank(track, tracks)
                    best = ranked[0] if ranked else None
                    second = ranked[1]["score"] if len(ranked) > 1 else 0.0
                    self.record(
                        track,
                        "PENDING",
                        "unknown",
                        "insufficient_new_identity_evidence",
                        ({**best, "second": second} if best is not None else None),
                        f"PNEW{self.next_id:04d}",
                    )
                break
            self.record(seed, gid, "promoted", "new_identity", None, provisional)
            pending = remain

        self.merge_pass(tracks)
        final = {key: gid for key, gid in self.mapping.items()}
        for row in self.decisions:
            row_gid = final.get(row.key, row.gid)
            if row_gid != row.gid:
                idx = self.decisions.index(row)
                self.decisions[idx] = DecisionBodyV6(
                    row.key,
                    row_gid,
                    row.state,
                    "identity_merge",
                    row.score,
                    row.margin,
                    row.body,
                    row.temporal,
                    row.spatial,
                    row.support,
                    row.camera,
                    row.provisional,
                    True,
                )
        return final, list(self.decisions)


class BatchPipelineStateInvariantJointGuarded(BatchPipelineStateInvariantJoint):
    """Joint V6 MTMC with high-overlap isolation and high-confidence promotion."""

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
        support = sum(scores[model] >= self.recovery_model_min[model] for model in scores)
        ordered = sorted(scores.values(), reverse=True)
        fused = float(np.mean(ordered[:2])) if len(ordered) >= 2 else 0.0
        return support >= self.recovery_models and fused >= self.recovery_fused, scores

    def _remember_recovery(self, key: str, vectors: Dict[str, np.ndarray]) -> None:
        refs = self._recovery_refs.setdefault(key, {"resnet": [], "swin": [], "solider": []})
        for model in refs:
            refs[model].append(np.asarray(vectors[model], np.float32))
            refs[model] = refs[model][-4:]

    def _extract(self, camera, frame, fps, image, prepared, blocked, partners, last, was_overlap, recovery_left, stats, rows):
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
            rows.write(__import__("json").dumps({
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
                "recovery_after_overlap": bool((not active) and recovery_left.get(key, 0) > 0),
            }) + "\n")

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
            if self.light and (due or frame - last.get(key + ":light", -10**9) >= self.part_interval):
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
                    rows.write(__import__("json").dumps({
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
                    }) + "\n")
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

    def run(self, values):
        import rebuild.identity_body_v6 as body_module
        original = body_module.GlobalIdentityBodyV6
        body_module.GlobalIdentityBodyV6 = HighConfidenceIdentityBodyV6
        try:
            return super().run(values)
        finally:
            body_module.GlobalIdentityBodyV6 = original
