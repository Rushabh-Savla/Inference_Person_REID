from __future__ import annotations

import json
from typing import Dict

import cv2
import numpy as np

from rebuild.batch_state_invariant_joint import BatchPipelineStateInvariantJoint
from rebuild.batch_state_invariant_joint_guarded import HighConfidenceIdentityBodyV6
from rebuild.identity_v2 import crop, illumination_variant, quality
from rebuild.person_attributes import pack


class BatchPipelineStateInvariantJointAttributes(BatchPipelineStateInvariantJoint):
    """Joint V6 MTMC with continuous ReID checks and stable overlap handling."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        guard = self.cfg.get("overlap_guard", {}) or {}
        self.overlap_iou = float(guard.get("iou_min", 0.80))
        self.overlap_intersection = float(guard.get("intersection_min", 0.85))
        self.recovery_samples = max(1, int(guard.get("recovery_samples", 4)))
        recovery = self.cfg.get("recovery_guard", {}) or {}
        self.recovery_models = max(2, int(recovery.get("required_models", 2)))
        self.recovery_fused = float(recovery.get("fused_min", 0.56))
        self.recovery_min = {
            "resnet": float(recovery.get("resnet_min", 0.52)),
            "swin": float(recovery.get("swin_min", 0.52)),
            "solider": float(recovery.get("solider_min", 0.50)),
        }
        self.recovery_relaxed = float(recovery.get("relaxed_fused_min", 0.50))
        self.recovery_relaxed_min = {
            "resnet": float(recovery.get("relaxed_resnet_min", 0.48)),
            "swin": float(recovery.get("relaxed_swin_min", 0.48)),
            "solider": float(recovery.get("relaxed_solider_min", 0.46)),
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

    def _recovery(self, key: str, vectors: Dict[str, np.ndarray], relaxed: bool = False):
        refs = self._refs.get(key, {})
        mins = self.recovery_relaxed_min if relaxed else self.recovery_min
        fused_min = self.recovery_relaxed if relaxed else self.recovery_fused
        scores = {}
        for model in ("resnet", "swin", "solider"):
            query = self._unit(vectors[model])
            vals = []
            if query is not None:
                for value in refs.get(model, [])[-4:]:
                    ref = self._unit(value)
                    if ref is not None and ref.shape == query.shape:
                        vals.append(float(np.dot(query, ref)))
            vals.sort(reverse=True)
            scores[model] = float(np.mean(vals[: min(3, len(vals))])) if vals else 0.0
        support = sum(scores[name] >= mins[name] for name in scores)
        ordered = sorted(scores.values(), reverse=True)
        fused = float(np.mean(ordered[:2])) if len(ordered) >= 2 else 0.0
        return support >= self.recovery_models and fused >= fused_min, scores, fused

    def _remember(self, key: str, vectors: Dict[str, np.ndarray]):
        refs = self._refs.setdefault(key, {"resnet": [], "swin": [], "solider": []})
        for model in refs:
            refs[model].append(np.asarray(vectors[model], np.float32))
            refs[model] = refs[model][-4:]

    def _store_attrs(self, key, image, bbox):
        track = self.tracks[key]
        bank = getattr(track, "state_bank", None)
        if bank is None:
            return
        bank.setdefault("resnet", {}).setdefault("attributes", [])
        value = pack(image, self._frame_image, bbox)
        bank["resnet"]["attributes"].append(value)
        bank["resnet"]["attributes"] = bank["resnet"]["attributes"][-64:]

    def add_body(self, key, camera, track_id, segment, bbox, stamp, score, image, feats, multi):
        measured = float(quality(image))
        super().add_body(key, camera, track_id, segment, bbox, stamp, measured, image, feats, multi)
        self._store_attrs(key, image, bbox)

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
                # Preserve the last four clean observations BEFORE the overlap.
                # We do not create a replacement tracklet at the boundary.
                track = self.tracks.get(old_key)
                refs = {"resnet": [], "swin": [], "solider": []}
                if track is not None:
                    bank = getattr(track, "state_bank", {})
                    for model in refs:
                        refs[model] = list(bank.get(model, {}).get("full", [])[-4:])
                self._refs[old_key] = refs
                self._fails[old_key] = 0
                info["recovery_left"][old_key] = 0
                reason = "high_overlap_start_identity_locked"

            elif previous and not active:
                # IMPORTANT: keep the SAME tracklet key. The old global identity
                # remains the assignment anchor while post-overlap observations
                # pass the recovery gate.
                info["recovery_left"][old_key] = self.recovery_samples
                info["last"][old_key] = -10**9
                self._fails[old_key] = 0
                boundary = True
                reason = "high_overlap_exit_recovery_same_identity"
                info["overlap_tids"].discard(tid)

            info["was_overlap"][tid] = active
            partner_keys = partners.get(old_key, []) if active else []
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
                "overlap_partners": partner_keys,
                "overlap_boundary": bool(boundary),
                "segment_reason": reason,
                "recovery_after_overlap": bool((not active) and info["recovery_left"].get(key, 0) > 0),
            }) + "\n")

            recovery = info["recovery_left"].get(key, 0)
            due = recovery > 0
            normal_due = frame - info["last"].get(key, -10**9) >= self.interval
            overlap_due = active and frame - info["last"].get(key + ":overlap", -10**9) >= self.interval
            if not due and not normal_due and not overlap_due:
                continue

            person = crop(image, box)
            q = quality(person) if person is not None else 0.0
            if person is None or q < self.min_quality:
                continue

            # During severe overlap we CONTINUE extracting features, but keep
            # them quarantined: they can be checked against pre-overlap identity
            # history, yet never contaminate the trusted gallery by themselves.
            if active:
                variants = {"full": person}
                ordered = ["full"]
                crops = [person]
                resnet = self.extractor.extract_batch(crops)
                swin = self.swin.extract_batch(crops)
                solider = self.solider.extract_batch(crops)
                self._check(resnet, "NVIDIA ResNet", 1)
                self._check(swin, "NVIDIA Swin", 1)
                self._check(solider, "SOLIDER", 1)
                vectors = {
                    "resnet": np.asarray(resnet[0], np.float32),
                    "swin": np.asarray(swin[0], np.float32),
                    "solider": np.asarray(solider[0], np.float32),
                }
                checked, scores, fused = self._recovery(key, vectors, relaxed=True)
                info["last"][key + ":overlap"] = frame
                info.setdefault("overlap_feature_checks", 0)
                info["overlap_feature_checks"] += 1
                rows.write(json.dumps({
                    "camera": camera,
                    "frame": frame,
                    "timestamp": frame / fps,
                    "track_id": tid,
                    "tracklet_key": key,
                    "overlap_feature_check": True,
                    "overlap_reid_match": bool(checked),
                    "overlap_reid_fused": float(fused),
                    "overlap_reid_scores": scores,
                }) + "\n")
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
                relaxed = self._fails.get(key, 0) >= self.recovery_fails
                accepted, scores, fused = self._recovery(key, vectors, relaxed=relaxed)
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
                        "recovery_relaxed": bool(relaxed),
                        "recovery_scores": scores,
                        "recovery_fused": float(fused),
                        "recovery_fail_count": self._fails[key],
                    }) + "\n")
                    # Never create a fresh identity merely because a few first
                    # recovery samples are weak. Keep the same key alive and keep
                    # checking; the final resolver also has the full history.
                    info["recovery_left"][key] = max(1, recovery - 1)
                    if info["recovery_left"][key] == 0:
                        info["recovery_left"][key] = self.recovery_samples
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
            self.add_body(key, camera, tid, seg, box, frame / fps, float(detection.confidence), person, resnet_map, multi)
            info["samples"] += 1
            info["feature_batches"] += 1
            if due:
                info["recovery_samples"] += 1
                info["recovery_left"][key] = max(0, recovery - 1)
                if info["recovery_left"][key] == 0:
                    self._refs.pop(key, None)
                    self._fails.pop(key, None)

    def render(self, mapping):
        """Render the assigned GID even during overlap; add a small OVL marker."""
        for camera, meta in self.meta.items():
            cap = cv2.VideoCapture(meta["source"])
            out = self.out / f"{camera}_v6.mp4"
            writer = cv2.VideoWriter(
                str(out), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"])
            )
            rows = {}
            with (self.cache / f"{camera}.detections.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    rows.setdefault(int(item["frame"]), []).append(item)
            frame = 0
            try:
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    frame += 1
                    for item in rows.get(frame, []):
                        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
                        gid = mapping.get(item["tracklet_key"], "UNKNOWN")
                        if gid == "UNKNOWN" or not str(gid).startswith("G"):
                            colour = (145, 145, 145)
                            label = str(gid)
                        else:
                            colour = self.gid_colour(gid)
                            label = self.short_gid(gid)
                        if item.get("overlap_blocked"):
                            label = f"{label} OVL"
                            colour = tuple(max(0, int(v * 0.82)) for v in colour)

                        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
                        scale, thickness = 0.68, 2
                        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                        pad_x, pad_y = 9, 7
                        bx1 = max(0, x1)
                        by2 = max(th + base + 2, y1)
                        by1 = max(0, by2 - th - base - 2 * pad_y)
                        bx2 = min(image.shape[1] - 1, bx1 + tw + 2 * pad_x)
                        by2 = min(image.shape[0] - 1, by2)
                        cv2.rectangle(image, (bx1, by1), (bx2, by2), colour, -1)
                        text_colour = self.label_text_colour(colour)
                        cv2.putText(image, label, (bx1 + pad_x, by2 - pad_y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, text_colour, thickness, cv2.LINE_AA)
                    cv2.putText(image, camera, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    writer.write(image)
            finally:
                cap.release()
                writer.release()
            print(f"[state-joint-attributes] wrote {out}")

    def run(self, values):
        import rebuild.identity_body_v6 as body_module
        original = body_module.GlobalIdentityBodyV6
        body_module.GlobalIdentityBodyV6 = HighConfidenceIdentityBodyV6
        try:
            return super().run(values)
        finally:
            body_module.GlobalIdentityBodyV6 = original
