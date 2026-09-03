from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from detector import PersonDetector
from rebuild.batch_state_invariant_safe_overlap import BatchPipelineStateInvariantSafeOverlap
from rebuild import batch_state_invariant as state_pipeline
from rebuild.identity_v2 import crop, illumination_variant, quality
from rebuild.identity_body_v6 import GlobalIdentityBodyV6


class BatchPipelineStateInvariantJoint(BatchPipelineStateInvariantSafeOverlap):
    """Safe V6 MTMC session with one shared multi-camera processing loop.

    The three cameras are opened before processing starts and advanced in a
    common session. Video inputs are resolved robustly and, when OpenCV cannot
    decode an otherwise valid source, FFmpeg creates a local H.264 working copy
    so detection/tracking/rendering use a decoder that is known to be available
    on the deployment host.
    """

    @staticmethod
    def _resolve_source(path: str) -> str:
        value = Path(path).expanduser()
        if value.exists():
            return str(value)
        candidates = []
        try:
            for base in (Path.cwd(), Path.cwd().parent):
                candidates.extend(x for x in base.rglob(value.name) if x.is_file())
        except Exception:
            pass
        unique = []
        seen = set()
        for item in candidates:
            key = str(item.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(item)
        if len(unique) == 1:
            print(f"[state-joint] resolved missing input {path} -> {unique[0]}")
            return str(unique[0])
        if not value.exists():
            hint = ", ".join(str(item) for item in unique[:5])
            if hint:
                raise RuntimeError(f"Input video not found: {path}. Candidate match(es): {hint}")
            raise RuntimeError(f"Input video not found: {path}")
        return str(value)

    @staticmethod
    def _transcode(path: str, target: Path) -> str:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
            "-i", path,
            "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not target.exists() or target.stat().st_size <= 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"FFmpeg could not create a working video copy for {path}: {detail[-1000:]}")
        return str(target)

    def _open(self, camera: str, path: str):
        source = self._resolve_source(path)
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            cache = self.cache / "decoded_inputs"
            stem = Path(source).stem.replace(".", "_")
            working = cache / f"{camera}_{stem}_h264.mp4"
            print(f"[state-joint] OpenCV decode failed for {source}; creating FFmpeg H.264 working copy")
            working_source = self._transcode(source, working)
            cap = cv2.VideoCapture(working_source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(working_source)
            source = working_source

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video/RTSP source: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.meta[camera] = {
            "source": source,
            "original_source": path,
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
        print(f"[state-joint] {camera}: source={source} frames={total} fps={fps:.2f} size={width}x{height}")
        return cap, detector, fps, total

    def _extract(
        self,
        camera: str,
        frame: int,
        fps: float,
        image: np.ndarray,
        prepared,
        blocked,
        partners,
        info,
        rows,
    ):
        for item in prepared:
            tid = item["tid"]
            box = item["bbox"]
            detection = item["item"]
            old_key = item["key"]
            active = old_key in blocked
            previous_active = bool(info["was_overlap"].get(tid, False))
            key = old_key
            seg = item["seg"]
            boundary = False
            reason = "normal"

            if active and not previous_active:
                info["overlap_events"] += 1
                info["overlap_tids"].add(tid)
                info["recovery_left"][key] = 0
                reason = "overlap_start"
            elif previous_active and not active:
                info["segments"][tid] = info["segments"].get(tid, seg) + 1
                self._segments[camera][tid] = info["segments"][tid]
                seg = info["segments"][tid]
                key = f"{camera}:{tid}:{seg}"
                info["recovery_left"][key] = self.recovery_samples
                info["last"][key] = -10**9
                boundary = True
                reason = "overlap_exit_new_segment"
                info["overlap_tids"].discard(tid)

            info["was_overlap"][tid] = active
            partner_keys = partners.get(old_key, []) if active else []
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
                        "segment_reason": reason,
                        "recovery_after_overlap": bool(
                            (not active) and info["recovery_left"].get(key, 0) > 0
                        ),
                    }
                )
                + "\n"
            )

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

            variants: Dict[str, np.ndarray] = {"full": person}
            if self.light and (
                due
                or frame - info["last"].get(key + ":light", -10**9)
                >= self.part_interval
            ):
                variants["light"] = illumination_variant(person)
                info["last"][key + ":light"] = frame

            if due or frame - info["last"].get(key + ":parts", -10**9) >= self.part_interval:
                variants.update(self.parts(person))
                info["last"][key + ":parts"] = frame

            info["last"][key] = frame
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
            info["samples"] += 1
            info["feature_batches"] += 1
            if due:
                info["recovery_samples"] += 1
                info["recovery_left"][key] = max(0, recovery - 1)

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
            print("[state-joint] PASS 1: all cameras active in one shared processing session")

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

                    raw_items = [
                        item
                        for item in detectors[camera].track(image)
                        if item.track_id is not None
                    ]
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
                        prepared.append(
                            {
                                "item": item,
                                "tid": tid,
                                "seg": seg,
                                "key": key,
                                "bbox": (item.x1, item.y1, item.x2, item.y2),
                            }
                        )

                    blocked, partners = self._overlaps(prepared)
                    if blocked:
                        info["overlap_frames"] += 1
                    self._extract(
                        camera,
                        frame,
                        fps,
                        image,
                        prepared,
                        blocked,
                        partners,
                        info,
                        rows[camera],
                    )

            self.save_cache()

            for camera in cameras:
                info = state[camera]
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
            print("[state-joint] PASS 2: protected V6 local appearance proposals")
            for camera in cameras:
                subset = {
                    key: track
                    for key, track in self.tracks.items()
                    if track.camera == camera
                }
                mapping, _ = GlobalIdentityBodyV6(self.cfg["identity_v6"]).run(subset)
                local_mapping.update(mapping)
                print(
                    f"[state-joint] local {camera}: tracklets={len(subset)} "
                    f"local_ids={len(set(mapping.values()))}"
                )

            print("[state-joint] PASS 3: tracker-reset repair + state-invariant MTMC")
            resolver = state_pipeline.StateInvariantFinalResolver(
                dict(self.cfg["identity_v6"]), registry=self.registry
            )
            global_mapping, components, edges = resolver.resolve(
                local_mapping, self.tracks, cameras
            )
            debug = {
                "mode": "joint_multicamera",
                "local_mapping": local_mapping,
                "global_mapping": global_mapping,
                "components": components,
                "edges": edges,
            }
            (self.out / "state_invariant_debug.json").write_text(
                json.dumps(debug, indent=2), encoding="utf-8"
            )

            same_count = sum(
                1
                for edge in edges
                if str(edge.get("left", "")).split(":", 1)[0]
                == str(edge.get("right", "")).split(":", 1)[0]
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
