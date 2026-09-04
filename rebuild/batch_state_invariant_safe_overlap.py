from __future__ import annotations

import json
from typing import Dict

import cv2
import numpy as np

from rebuild.batch_state_invariant_safe_final import BatchPipelineStateInvariantSafeFinal
from detector import PersonDetector
from rebuild.identity_v2 import crop, illumination_variant, quality


class BatchPipelineStateInvariantSafeOverlap(BatchPipelineStateInvariantSafeFinal):
    """V6 state-invariant pipeline with overlap-boundary tracklet isolation.

    The V6 architecture is unchanged. This layer only prevents a ByteTrack
    identity swap during a physical overlap from contaminating one tracklet's
    multimodel state bank with pre- and post-overlap observations.
    """

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
        was_overlap: Dict[int, bool] = {}
        recovery_left: Dict[str, int] = {}
        frame = 0
        samples = 0
        feature_batches = 0
        overlap_frames = 0
        overlap_events = 0
        recovery_sample_count = 0
        overlap_tids: set[int] = set()

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

                base = []
                for item in raw_items:
                    tid = int(item.track_id)
                    previous = seen.get(tid)
                    gap_reset = previous is None or frame - previous > int(self.gap * fps)
                    if gap_reset:
                        segments[tid] = segments.get(tid, 0) + 1
                        was_overlap.pop(tid, None)
                    seg = segments[tid]
                    key = f"{camera}:{tid}:{seg}"
                    seen[tid] = frame
                    base.append(
                        {
                            "item": item,
                            "tid": tid,
                            "seg": seg,
                            "key": key,
                            "bbox": (item.x1, item.y1, item.x2, item.y2),
                        }
                    )

                blocked, partners = self._overlaps(base)
                if blocked:
                    overlap_frames += 1

                for item in base:
                    tid = item["tid"]
                    box = item["bbox"]
                    detection = item["item"]
                    active = item["key"] in blocked
                    previous_active = bool(was_overlap.get(tid, False))
                    boundary = False
                    segment_reason = "normal"
                    key = item["key"]
                    seg = item["seg"]

                    if active and not previous_active:
                        overlap_events += 1
                        overlap_tids.add(tid)
                        recovery_left[key] = 0
                        segment_reason = "overlap_start"

                    elif previous_active and not active:
                        segments[tid] = segments.get(tid, seg) + 1
                        seg = segments[tid]
                        key = f"{camera}:{tid}:{seg}"
                        recovery_left[key] = self.recovery_samples
                        last[key] = -10**9
                        boundary = True
                        segment_reason = "overlap_exit_new_segment"
                        overlap_tids.discard(tid)

                    was_overlap[tid] = active

                    partner_keys = partners.get(item["key"], []) if active else []
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
                                "overlap_partners": partner_keys,
                                "overlap_boundary": bool(boundary),
                                "segment_reason": segment_reason,
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
                        or frame - last.get(key + ":light", -10**9) >= self.part_interval
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
                        name: value for name, value in zip(ordered, resnet)
                    }
                    multi = {
                        "swin": {name: [value] for name, value in zip(ordered, swin)},
                        "solider": {name: [value] for name, value in zip(ordered, solider)},
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
                        recovery_sample_count += 1
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
            f"[overlap-safe-v6] {camera}: frames={frame} "
            f"tracklets={len(tracks)} multiview_samples={samples} "
            f"feature_batches={feature_batches} complete_tracklets={complete} "
            f"overlap_frames={overlap_frames} overlap_events={overlap_events} "
            f"recovery_samples={recovery_sample_count} total={total}"
        )

    def render(self, mapping):
        """Keep ambiguous overlap frames visibly unassigned in the presentation."""
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
                        if item.get("overlap_blocked"):
                            colour = (120, 120, 120)
                            label = "OVERLAP"
                        else:
                            gid = mapping.get(item["tracklet_key"], "UNKNOWN")
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
                        by1 = max(0, by2 - th - base - 2 * pad_y)
                        bx2 = min(image.shape[1] - 1, bx1 + tw + 2 * pad_x)
                        by2 = min(image.shape[0] - 1, by2)
                        cv2.rectangle(image, (bx1, by1), (bx2, by2), colour, -1)
                        text_colour = self.label_text_colour(colour)
                        cv2.putText(
                            image,
                            label,
                            (bx1 + pad_x, by2 - pad_y - base),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scale,
                            text_colour,
                            thickness,
                            cv2.LINE_AA,
                        )
                    cv2.putText(image, camera, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    writer.write(image)
            finally:
                cap.release()
                writer.release()
            print(f"[overlap-safe-v6] wrote {out}")
