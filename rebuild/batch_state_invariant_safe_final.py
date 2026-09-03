from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from detector import PersonDetector
from rebuild.batch_state_invariant_safe import BatchPipelineStateInvariantSafe
from rebuild.identity_v2 import crop, illumination_variant, quality


class BatchPipelineStateInvariantSafeFinal(BatchPipelineStateInvariantSafe):
    """Final safe state-invariant path with overlap-protected ReID sampling."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        guard = self.cfg.get("overlap_guard", {}) or {}
        self.overlap_iou = float(guard.get("iou_min", 0.05))
        self.overlap_intersection = float(guard.get("intersection_min", 0.20))
        self.recovery_samples = max(1, int(guard.get("recovery_samples", 2)))

    @staticmethod
    def _area(box) -> float:
        x1, y1, x2, y2 = [float(v) for v in box]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @classmethod
    def _overlap(cls, left, right) -> bool:
        lx1, ly1, lx2, ly2 = [float(v) for v in left]
        rx1, ry1, rx2, ry2 = [float(v) for v in right]
        ix1 = max(lx1, rx1)
        iy1 = max(ly1, ry1)
        ix2 = min(lx2, rx2)
        iy2 = min(ly2, ry2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return False
        la = cls._area(left)
        ra = cls._area(right)
        if la <= 0.0 or ra <= 0.0:
            return False
        union = la + ra - inter
        iou = inter / union if union > 0.0 else 0.0
        iom = inter / min(la, ra)
        return iou >= cls._active_iou or iom >= cls._active_intersection

    @classmethod
    def _pairs(cls, boxes):
        cls._active_iou = 0.05
        cls._active_intersection = 0.20
        blocked = set()
        partners = {}
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if cls._overlap(boxes[i]["bbox"], boxes[j]["bbox"]):
                    a = boxes[i]["key"]
                    b = boxes[j]["key"]
                    blocked.add(a)
                    blocked.add(b)
                    partners.setdefault(a, []).append(b)
                    partners.setdefault(b, []).append(a)
        return blocked, partners

    def _is_overlap(self, left, right) -> bool:
        lx1, ly1, lx2, ly2 = [float(v) for v in left]
        rx1, ry1, rx2, ry2 = [float(v) for v in right]
        ix1 = max(lx1, rx1)
        iy1 = max(ly1, ry1)
        ix2 = min(lx2, rx2)
        iy2 = min(ly2, ry2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return False
        la = self._area(left)
        ra = self._area(right)
        if la <= 0.0 or ra <= 0.0:
            return False
        union = la + ra - inter
        iou = inter / union if union > 0.0 else 0.0
        iom = inter / min(la, ra)
        return iou >= self.overlap_iou or iom >= self.overlap_intersection

    def _overlaps(self, items) -> tuple[set[str], dict[str, list[str]]]:
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

    def collect(self, camera: str, path: str):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video/RTSP source: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.meta[camera] = {
            "source": path,
            "fps": fps,
            "width": width,
            "height": height,
            "frames": total,
        }

        detector = PersonDetector(
            model_path=self.detector["model"],
            confidence_threshold=float(self.detector["conf"]),
            person_class_id=0,
            tracker_config=self.detector["tracker"],
            pose_ensemble=None,
            iou=float(self.detector["iou"]),
        )

        self.cache.mkdir(parents=True, exist_ok=True)
        rows = (self.cache / f"{camera}.detections.jsonl").open("w", encoding="utf-8")
        last: Dict[str, int] = {}
        segments: Dict[int, int] = {}
        seen: Dict[int, int] = {}
        was_overlap: Dict[str, bool] = {}
        recovery_left: Dict[str, int] = {}
        frame = 0
        samples = 0
        feature_batches = 0
        overlap_frames = 0
        overlap_events = 0
        recovery_samples = 0

        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                frame += 1

                raw_items = [
                    item
                    for item in detector.track(image)
                    if item.track_id is not None
                ]

                prepared = []
                for item in raw_items:
                    tid = int(item.track_id)
                    previous = seen.get(tid)
                    if previous is None or frame - previous > int(self.gap * fps):
                        segments[tid] = segments.get(tid, 0) + 1
                    seg = segments[tid]
                    key = f"{camera}:{tid}:{seg}"
                    seen[tid] = frame
                    box = (item.x1, item.y1, item.x2, item.y2)
                    prepared.append(
                        {
                            "item": item,
                            "tid": tid,
                            "seg": seg,
                            "key": key,
                            "bbox": box,
                        }
                    )

                blocked, partners = self._overlaps(prepared)
                if blocked:
                    overlap_frames += 1

                for item in prepared:
                    tid = item["tid"]
                    seg = item["seg"]
                    key = item["key"]
                    box = item["bbox"]
                    detection = item["item"]
                    active = key in blocked
                    previous_active = was_overlap.get(key, False)

                    if active and not previous_active:
                        overlap_events += 1
                        recovery_left[key] = 0
                    elif previous_active and not active:
                        recovery_left[key] = self.recovery_samples
                        last[key] = -10**9

                    was_overlap[key] = active

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
                                "overlap_partners": partners.get(key, []),
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
                        due
                        or frame - last.get(key + ":light", -10**9)
                        >= self.part_interval
                    ):
                        variants["light"] = illumination_variant(person)
                        last[key + ":light"] = frame

                    if due or frame - last.get(key + ":parts", -10**9) >= self.part_interval:
                        variants.update(self.parts(person))
                        last[key + ":parts"] = frame

                    last[key] = frame
                    ordered = list(variants.keys())
                    crops = [variants[name] for name in ordered]

                    resnet = self.extractor.extract_batch(crops)
                    swin = self.swin.extract_batch(crops)
                    solider = self.solider.extract_batch(crops)

                    self._check(resnet, "NVIDIA ResNet", len(crops))
                    self._check(swin, "NVIDIA Swin", len(crops))
                    self._check(solider, "SOLIDER", len(crops))

                    resnet_feats = {
                        name: value
                        for name, value in zip(ordered, resnet)
                    }
                    multi = {
                        "swin": {
                            name: [value]
                            for name, value in zip(ordered, swin)
                        },
                        "solider": {
                            name: [value]
                            for name, value in zip(ordered, solider)
                        },
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
                        resnet_feats,
                        multi,
                    )

                    samples += 1
                    feature_batches += 1
                    if due:
                        recovery_samples += 1
                        recovery_left[key] = max(0, recovery - 1)

        finally:
            cap.release()
            rows.close()

        tracks = [
            track
            for key, track in self.tracks.items()
            if key.startswith(camera + ":")
        ]
        complete = sum(
            1
            for track in tracks
            if all(
                getattr(track, "state_bank", {})
                .get(model, {})
                .get(view)
                for model in ("resnet", "swin", "solider")
                for view in ("full", "upper", "torso")
            )
        )

        if samples == 0 or complete == 0:
            raise RuntimeError(
                f"{camera}: multimodel extraction produced no usable complete "
                f"tracklets (samples={samples}, complete={complete})"
            )

        print(
            f"[safe-final] {camera}: frames={frame} "
            f"tracklets={len(tracks)} multiview_samples={samples} "
            f"feature_batches={feature_batches} complete_tracklets={complete} "
            f"overlap_frames={overlap_frames} overlap_events={overlap_events} "
            f"recovery_samples={recovery_samples} total={total}"
        )
