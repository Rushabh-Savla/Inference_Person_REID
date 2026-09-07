from __future__ import annotations

import json
from typing import Dict

import cv2
import numpy as np

from rebuild import batch_state_invariant as state_pipeline
from rebuild.batch_state_invariant_joint_attributes import BatchPipelineStateInvariantJointAttributes
from rebuild.qdrant_authoritative import QdrantAuthoritativeResolver
from rebuild.identity_v2 import crop, illumination_variant, quality


class BatchPipelineStateAuthoritative(BatchPipelineStateInvariantJointAttributes):
    """Final recorded-video path for accuracy-first MTMC person ReID."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        guard = self.cfg.get("overlap_guard", {}) or {}
        self.overlap_enter = float(guard.get("iou_min", 0.70))
        self.overlap_intersection_enter = float(guard.get("intersection_min", 0.70))
        self.overlap_exit = float(self.cfg.get("authoritative_overlap_exit", 0.58))
        self.overlap_grace = max(1, int(guard.get("clear_grace_frames", 2)))
        self._overlap_pairs: Dict[tuple[str, str], Dict[str, float]] = {}
        state_pipeline.StateInvariantFinalResolver = QdrantAuthoritativeResolver

    @staticmethod
    def _metrics(left, right):
        ax1, ay1, ax2, ay2 = [float(v) for v in left]
        bx1, by1, bx2, by2 = [float(v) for v in right]
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter <= 0.0:
            return 0.0, 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if area_a <= 0.0 or area_b <= 0.0:
            return 0.0, 0.0
        union = area_a + area_b - inter
        iou = inter / union if union > 0.0 else 0.0
        iom = inter / min(area_a, area_b)
        return float(iou), float(iom)

    def _overlaps(self, items):
        blocked = set()
        partners = {}
        seen_pairs = set()

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                left = items[i]
                right = items[j]
                a = str(left["key"])
                b = str(right["key"])
                pair = tuple(sorted((a, b)))
                seen_pairs.add(pair)
                iou, iom = self._metrics(left["bbox"], right["bbox"])
                signal = max(iou, iom)
                state = self._overlap_pairs.get(pair)

                if state is None:
                    active = bool(
                        iou >= self.overlap_enter
                        or iom >= self.overlap_intersection_enter
                    )
                    self._overlap_pairs[pair] = {
                        "active": float(active),
                        "clear": 0.0,
                    }
                elif bool(state["active"]):
                    if signal <= self.overlap_exit:
                        state["clear"] += 1.0
                        if state["clear"] >= self.overlap_grace:
                            state["active"] = 0.0
                            state["clear"] = 0.0
                    else:
                        state["clear"] = 0.0
                    active = bool(state["active"])
                else:
                    active = bool(
                        iou >= self.overlap_enter
                        or iom >= self.overlap_intersection_enter
                    )
                    if active:
                        state["active"] = 1.0
                        state["clear"] = 0.0

                if active:
                    blocked.add(a)
                    blocked.add(b)
                    partners.setdefault(a, []).append(b)
                    partners.setdefault(b, []).append(a)

        for pair in list(self._overlap_pairs):
            if pair not in seen_pairs:
                self._overlap_pairs.pop(pair, None)

        return blocked, partners

    def _extract(self, camera, frame, fps, image, prepared, blocked, partners, info, rows):
        """Authoritative extraction: hard-freeze identity during severe overlap."""
        for item in prepared:
            tid = item["tid"]
            box = item["bbox"]
            detection = item["item"]
            key = item["key"]
            active = key in blocked
            previous = bool(info["was_overlap"].get(tid, False))
            seg = item["seg"]
            boundary = False
            reason = "normal"

            if active and not previous:
                info["overlap_events"] += 1
                info["overlap_tids"].add(tid)
                track = self.tracks.get(key)
                refs = {"resnet": [], "swin": [], "solider": []}
                if track is not None:
                    bank = getattr(track, "state_bank", {})
                    for model in refs:
                        refs[model] = list(bank.get(model, {}).get("full", [])[-6:])
                    setattr(track, "overlap_recovery", True)
                    setattr(track, "recovery_sources", [key])
                self._refs[key] = refs
                self._fails[key] = 0
                info["recovery_left"][key] = 0
                reason = "authoritative_overlap_identity_frozen"

            elif previous and not active:
                info["recovery_left"][key] = self.recovery_samples
                info["last"][key] = -10**9
                self._fails[key] = 0
                boundary = True
                reason = "authoritative_overlap_exit_recovery"
                info["overlap_tids"].discard(tid)
                track = self.tracks.get(key)
                if track is not None:
                    setattr(track, "overlap_recovery", True)
                    setattr(track, "recovery_sources", [key])

            info["was_overlap"][tid] = active
            partner_keys = partners.get(key, []) if active else []
            recovery_now = (not active) and info["recovery_left"].get(key, 0) > 0
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
                "recovery_after_overlap": bool(recovery_now),
            }) + "\n")

            # NEVER extract identity features while the overlap guard is active.
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
                accepted, scores, fused = self._recovery(key, vectors, relaxed=False)
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
                        "recovery_relaxed": False,
                        "recovery_scores": scores,
                        "recovery_fused": float(fused),
                        "recovery_fail_count": self._fails[key],
                    }) + "\n")
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

    def render(self, mapping):
        """Render PENDING during overlap/recovery; never expose an identity there."""
        for camera, meta in self.meta.items():
            cap = cv2.VideoCapture(meta["source"])
            out = self.out / f"{camera}_v6.mp4"
            writer = cv2.VideoWriter(
                str(out),
                cv2.VideoWriter_fourcc(*"mp4v"),
                meta["fps"],
                (meta["width"], meta["height"]),
            )
            rows = {}
            priority = {}
            with (self.cache / f"{camera}.detections.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    if "bbox" not in item or "tracklet_key" not in item:
                        continue
                    frame = int(item["frame"])
                    key = str(item["tracklet_key"])
                    rank = 3 if item.get("overlap_blocked") else (2 if item.get("recovery_after_overlap") else 1)
                    old = priority.get((frame, key), 0)
                    if rank >= old:
                        rows.setdefault(frame, {})[key] = item
                        priority[(frame, key)] = rank

            frame = 0
            try:
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    frame += 1
                    for item in rows.get(frame, {}).values():
                        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
                        overlap = bool(item.get("overlap_blocked"))
                        recovery = bool(item.get("recovery_after_overlap"))

                        if overlap or recovery:
                            label = "PENDING"
                            colour = (70, 70, 210) if overlap else (80, 150, 210)
                        else:
                            gid = self.label(mapping, item["tracklet_key"])
                            if gid == "UNKNOWN" or not str(gid).startswith("G"):
                                colour = (145, 145, 145)
                                label = str(gid)
                            else:
                                colour = self.gid_colour(gid)
                                label = self.short_gid(gid)

                        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
                        scale, thickness = 0.68, 2
                        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                        pad_x, pad_y = 9, 7
                        bx1 = max(0, x1)
                        by2 = max(th + base + 2, y1)
                        by1 = max(0, by2 - th - base - pad_y * 2)
                        bx2 = min(image.shape[1] - 1, bx1 + tw + pad_x * 2)
                        cv2.rectangle(image, (bx1, by1), (bx2, by2), colour, -1)
                        cv2.putText(
                            image,
                            label,
                            (bx1 + pad_x, by2 - base - pad_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scale,
                            (255, 255, 255),
                            thickness,
                            cv2.LINE_AA,
                        )
                    writer.write(image)
            finally:
                cap.release()
                writer.release()

    @staticmethod
    def label(mapping, key, overlap=False, recovery=False):
        value = mapping.get(key)
        if overlap or recovery:
            return "PENDING"
        if value is None:
            return "UNKNOWN"
        text = str(value)
        if text == "PENDING":
            return "PENDING"
        if text == "UNKNOWN":
            return "UNKNOWN"
        if text.startswith("G") and text[1:].isdigit():
            return text
        return "UNKNOWN"


__all__ = ["BatchPipelineStateAuthoritative"]
