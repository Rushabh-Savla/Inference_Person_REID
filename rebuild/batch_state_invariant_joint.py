from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from detector import PersonDetector
from rebuild.batch_state_invariant_safe_overlap import BatchPipelineStateInvariantSafeOverlap
from rebuild import batch_state_invariant as state_pipeline


class BatchPipelineStateInvariantJoint(BatchPipelineStateInvariantSafeOverlap):
    """Safe V6 state-invariant MTMC session that advances all cameras together.

    The established V6 extraction, local-body resolver, state-invariant resolver,
    persistence, and overlap isolation are retained. The only orchestration
    change is that all source videos are opened at once and advanced in one shared
    processing loop, so camera evidence is accumulated in a common session rather
    than exhausting one camera before starting the next.
    """

    def _open(self, camera: str, path: str):
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
        return cap, detector, fps, total

    def _variants(self, person: np.ndarray, frame: int, key: str, last: Dict[str, int], recovery: bool):
        variants: Dict[str, np.ndarray] = {"full": person}
        if self.light and (
            recovery
            or frame - last.get(key + ":light", -10**9) >= self.part_interval
        ):
            variants["light"] = __import__("rebuild.identity_v2", fromlist=["illumination_variant"]).illumination_variant(person)
            last[key + ":light"] = frame
        if recovery or frame - last.get(key + ":parts", -10**9) >= self.part_interval:
            variants.update(self.parts(person))
            last[key + ":parts"] = frame
        return variants

    def _extract(self, camera: str, frame: int, fps: float, image: np.ndarray, prepared, blocked, partners, last, was_overlap, recovery_left, stats, rows):
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
                reason = "overlap_start"
            elif previous_active and not active:
                self._overlap_exit_segment(tid, camera, key, seg, recovery_left, last)
                seg = self._segments[camera][tid]
                key = f"{camera}:{tid}:{seg}"
                recovery_left[key] = self.recovery_samples
                last[key] = -10**9
                boundary = True
                reason = "overlap_exit_new_segment"
                stats["overlap_tids"].discard(tid)

            was_overlap[tid] = active
            partner_keys = partners.get(old_key, []) if active else []
            rows.write(
                json.dumps({
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
                }) + "\n"
            )

            if active:
                continue

            recovery = recovery_left.get(key, 0)
            due = recovery > 0
            if not due and frame - last.get(key, -10**9) < self.interval:
                continue

            person = __import__("rebuild.identity_v2", fromlist=["crop"]).crop(image, box)
            q = __import__("rebuild.identity_v2", fromlist=["quality"]).quality(person) if person is not None else 0.0
            if person is None or q < self.min_quality:
                continue

            variants = self._variants(person, frame, key, last, due)
            last[key] = frame
            ordered = list(variants.keys())
            crops = [variants[name] for name in ordered]

            resnet = self.extractor.extract_batch(crops)
            swin = self.swin.extract_batch(crops)
            solider = self.solider.extract_batch(crops)
            self._check(resnet, "NVIDIA ResNet", len(crops))
            self._check(swin, "NVIDIA Swin", len(crops))
            self._check(solider, "SOLIDER", len(crops))

            resnet_feats = {name: value for name, value in zip(ordered, resnet)}
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
            stats["samples"] += 1
            stats["feature_batches"] += 1
            if due:
                stats["recovery_samples"] += 1
                recovery_left[key] = max(0, recovery - 1)

    def _overlap_exit_segment(self, tid: int, camera: str, key: str, seg: int, recovery_left: Dict[str, int], last: Dict[str, int]):
        del camera, key, recovery_left, last
        self._segments[tid] = self._segments.get(tid, seg) + 1

    def run(self, values: List[str]):
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")
        cameras = [camera for camera, _ in sources]

        caps = {}
        detectors = {}
        state = {}
        self._segments = {camera: {} for camera in cameras}

        for camera, path in sources:
            cap, detector, fps, total = self._open(camera, path)
            caps[camera] = cap
            detectors[camera] = detector
            state[camera] = {
                "fps": fps,
                "total": total,
                "seen": {},
                "segments": {},
                "last": {},
                "was_overlap": {},
                "recovery_left": {},
                "frame": 0,
                "samples": 0,
                "feature_batches": 0,
                "overlap_frames": 0,
                "overlap_events": 0,
                "recovery_samples": 0,
                "overlap_tids": set(),
            }

        self.cache.mkdir(parents=True, exist_ok=True)
        rows = {
            camera: (self.cache / f"{camera}.detections.jsonl").open("w", encoding="utf-8")
            for camera in cameras
        }

        try:
            print(f"[state-joint] ResNet: {self.extractor.describe()}")
            print(f"[state-joint] Swin:   {self.swin.describe()}")
            print(f"[state-joint] SOLIDER:{self.solider.describe()}")
            print("[state-joint] FACE: OFF")
            print(f"[state-joint] cameras: {', '.join(cameras)}")
            print("[state-joint] pass 1: joint multi-camera detect + track + verified multimodel extraction")

            active = set(cameras)
            while active:
                for camera in cameras:
                    if camera not in active:
                        continue
                    info = state[camera]
                    ok, image = caps[camera].read()
                    if not ok:
                        active.discard(camera)
                        continue
                    info["frame"] += 1
                    frame = info["frame"]
                    fps = info["fps"]
                    raw_items = [item for item in detectors[camera].track(image) if item.track_id is not None]

                    prepared = []
                    for item in raw_items:
                        tid = int(item.track_id)
                        previous = info["seen"].get(tid)
                        if previous is None or frame - previous > int(self.gap * fps):
                            info["segments"][tid] = info["segments"].get(tid, 0) + 1
                            info["was_overlap"].pop(tid, None)
                        seg = info["segments"][tid]
                        self._segments[camera][tid] = seg
                        key = f"{camera}:{tid}:{seg}"
                        info["seen"][tid] = frame
                        prepared.append({
                            "item": item,
                            "tid": tid,
                            "seg": seg,
                            "key": key,
                            "bbox": (item.x1, item.y1, item.x2, item.y2),
                        })

                    blocked, partners = self._overlaps(prepared)
                    if blocked:
                        info["overlap_frames"] += 1
                    stats = {
                        "samples": info["samples"],
                        "feature_batches": info["feature_batches"],
                        "overlap_frames": info["overlap_frames"],
                        "overlap_events": info["overlap_events"],
                        "recovery_samples": info["recovery_samples"],
                        "overlap_tids": info["overlap_tids"],
                    }
                    self._extract(
                        camera,
                        frame,
                        fps,
                        image,
                        prepared,
                        blocked,
                        partners,
                        info["last"],
                        info["was_overlap"],
                        info["recovery_left"],
                        stats,
                        rows[camera],
                    )
                    info["samples"] = stats["samples"]
                    info["feature_batches"] = stats["feature_batches"]
                    info["overlap_frames"] = stats["overlap_frames"]
                    info["overlap_events"] = stats["overlap_events"]
                    info["recovery_samples"] = stats["recovery_samples"]
                    info["overlap_tids"] = stats["overlap_tids"]

            self.save_cache()
            for camera in cameras:
                info = state[camera]
                tracks = [track for key, track in self.tracks.items() if key.startswith(camera + ":")]
                complete = sum(
                    1
                    for track in tracks
                    if all(
                        getattr(track, "state_bank", {}).get(model, {}).get(view)
                        for model in ("resnet", "swin", "solider")
                        for view in ("full", "upper", "torso")
                    )
                )
                if info["samples"] == 0 or complete == 0:
                    raise RuntimeError(
                        f"{camera}: multimodel extraction produced no usable complete tracklets "
                        f"(samples={info['samples']}, complete={complete})"
                    )
                print(
                    f"[state-joint] {camera}: frames={info['frame']} tracklets={len(tracks)} "
                    f"multiview_samples={info['samples']} feature_batches={info['feature_batches']} "
                    f"complete_tracklets={complete} overlap_frames={info['overlap_frames']} "
                    f"overlap_events={info['overlap_events']} recovery_samples={info['recovery_samples']} "
                    f"total={info['total']}"
                )

            local_mapping: Dict[str, str] = {}
            print("[state-joint] pass 2: protected V6 local appearance proposals")
            for camera in cameras:
                subset = {key: track for key, track in self.tracks.items() if track.camera == camera}
                mapping, _ = __import__("rebuild.identity_body_v6", fromlist=["GlobalIdentityBodyV6"]).GlobalIdentityBodyV6(self.cfg["identity_v6"]).run(subset)
                local_mapping.update(mapping)
                print(f"[state-joint] local {camera}: tracklets={len(subset)} local_ids={len(set(mapping.values()))}")

            print("[state-joint] pass 3: tracker-reset repair + state-invariant MTMC")
            resolver_cls = state_pipeline.StateInvariantFinalResolver
            resolver = resolver_cls(dict(self.cfg["identity_v6"]), registry=self.registry)
            global_mapping, components, edges = resolver.resolve(local_mapping, self.tracks, cameras)
            debug = {
                "mode": "joint_multicamera",
                "local_mapping": local_mapping,
                "global_mapping": global_mapping,
                "components": components,
                "edges": edges,
            }
            (self.out / "state_invariant_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
            same_count = sum(
                1 for edge in edges
                if str(edge.get("left", "")).split(":", 1)[0] == str(edge.get("right", "")).split(":", 1)[0]
            )
            cross_count = len(edges) - same_count
            print(f"[state-joint] accepted same-camera repairs: {same_count}")
            print(f"[state-joint] accepted cross-camera links: {cross_count}")
            print(f"[state-joint] final global IDs: {len(set(global_mapping.values()))}")
            print(f"[state-joint] persistent global IDs: {self.registry.gids()}")
            self.render(global_mapping)

            multi = {
                gid: sorted({key.split(":", 1)[0] for key in members})
                for gid, members in components.items()
                if len({key.split(":", 1)[0] for key in members}) > 1
            }
            print("MULTI-CAMERA IDS:")
            for gid, cams in sorted(multi.items()):
                print(f"  {gid}: {', '.join(cams)}")
            print(f"outputs: {self.out}")
            return global_mapping
        finally:
            for handle in rows.values():
                handle.close()
            for cap in caps.values():
                cap.release()
            self.registry.close()
