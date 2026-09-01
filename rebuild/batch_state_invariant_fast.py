from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from rebuild.batch_state_invariant_safe import BatchPipelineStateInvariantSafe
from rebuild.identity_body_v6 import GlobalIdentityBodyV6
from rebuild.multimodel_state_invariant_final import StateInvariantFinalResolver


class BatchPipelineStateInvariantFast(BatchPipelineStateInvariantSafe):
    """State-invariant V6 with cross-sample model batching and live progress."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        perf = self.cfg.get("performance", {})
        self.batch = max(1, int(perf.get("sample_batch", 16)))
        self.step = max(0.5, float(perf.get("progress_sec", 2.0)))

    @staticmethod
    def clock() -> float:
        return time.monotonic()

    @staticmethod
    def span(seconds: float) -> str:
        value = max(0, int(seconds))
        hour, value = divmod(value, 3600)
        minute, second = divmod(value, 60)
        if hour:
            return f"{hour}h {minute:02d}m {second:02d}s"
        if minute:
            return f"{minute}m {second:02d}s"
        return f"{second}s"

    def show(self, camera: str, frame: int, total: int, fps: float, start: float, samples: int, queue: int, tracklets: int) -> None:
        now = self.clock()
        wall = max(now - start, 1e-6)
        rate = frame / wall
        left = max(0, total - frame)
        eta = left / rate if rate > 0 else 0.0
        done = 100.0 * frame / total if total > 0 else 100.0
        video = frame / max(fps, 1e-6)
        length = total / max(fps, 1e-6) if total > 0 else 0.0
        print(
            f"[progress] {camera} | {done:6.2f}% | frame {frame}/{total} "
            f"| video {video:.1f}/{length:.1f}s | {rate:.2f} FPS "
            f"| ETA {self.span(eta)} | samples={samples} queued={queue} tracklets={tracklets}",
            flush=True,
        )

    def extract(self, jobs):
        flat = []
        starts = []
        for job in jobs:
            starts.append(len(flat))
            flat.extend(job[9])

        res = self.extractor.extract_batch(flat)
        swin = self.swin.extract_batch(flat)
        solider = self.solider.extract_batch(flat)
        self._check(res, "NVIDIA ResNet", len(flat))
        self._check(swin, "NVIDIA Swin", len(flat))
        self._check(solider, "SOLIDER", len(flat))

        for index, job in enumerate(jobs):
            key, camera, tid, seg, box, stamp, score, person, names, crops = job[0:10]
            left = starts[index]
            right = left + len(crops)
            body = {name: value for name, value in zip(names, res[left:right])}
            multi = {
                "swin": {name: [value] for name, value in zip(names, swin[left:right])},
                "solider": {name: [value] for name, value in zip(names, solider[left:right])},
            }
            self.add_body(key, camera, tid, seg, box, stamp, score, person, body, multi)

    def collect(self, camera: str, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video/RTSP source: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.meta[camera] = {"source": path, "fps": fps, "width": width, "height": height, "frames": total}

        detector = __import__("detector", fromlist=["PersonDetector"]).PersonDetector(
            model_path=self.detector["model"],
            confidence_threshold=float(self.detector["conf"]),
            person_class_id=0,
            tracker_config=self.detector["tracker"],
            pose_ensemble=None,
            iou=float(self.detector["iou"]),
        )

        rows = (self.cache / f"{camera}.detections.jsonl").open("w", encoding="utf-8")
        last: Dict[str, int] = {}
        segments: Dict[int, int] = {}
        seen: Dict[int, int] = {}
        jobs = []
        frame = 0
        samples = 0
        batches = 0
        start = self.clock()
        mark = start

        def flush():
            nonlocal jobs, samples, batches
            if not jobs:
                return
            self.extract(jobs)
            samples += len(jobs)
            batches += 1
            jobs = []

        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                frame += 1
                for item in detector.track(image):
                    if item.track_id is None:
                        continue
                    tid = int(item.track_id)
                    previous = seen.get(tid)
                    if previous is None or frame - previous > int(self.gap * fps):
                        segments[tid] = segments.get(tid, 0) + 1
                    seg = segments[tid]
                    key = f"{camera}:{tid}:{seg}"
                    seen[tid] = frame
                    box = (item.x1, item.y1, item.x2, item.y2)
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
                                "detection_score": float(item.confidence),
                            }
                        )
                        + "\n"
                    )
                    if frame - last.get(key, -10**9) < self.interval:
                        continue
                    person = __import__("rebuild.identity_v2", fromlist=["crop"]).crop(image, box)
                    quality = __import__("rebuild.identity_v2", fromlist=["quality"]).quality
                    value = quality(person) if person is not None else 0.0
                    if person is None or value < self.min_quality:
                        continue

                    variants: Dict[str, np.ndarray] = {"full": person}
                    if self.light and frame - last.get(key + ":light", -10**9) >= self.part_interval:
                        variant = __import__("rebuild.identity_v2", fromlist=["illumination_variant"]).illumination_variant(person)
                        variants["light"] = variant
                        last[key + ":light"] = frame
                    if frame - last.get(key + ":part", -10**9) >= self.part_interval:
                        variants.update(self.parts(person))
                        last[key + ":part"] = frame

                    last[key] = frame
                    names = list(variants.keys())
                    crops = [variants[name] for name in names]
                    jobs.append((key, camera, tid, seg, box, frame / fps, float(item.confidence), person, names, crops))
                    if len(jobs) >= self.batch:
                        flush()

                now = self.clock()
                if now - mark >= self.step:
                    rows.flush()
                    self.show(camera, frame, total, fps, start, samples, len(jobs), len(segments))
                    mark = now
        finally:
            flush()
            cap.release()
            rows.close()

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
        wall = self.clock() - start
        rate = frame / max(wall, 1e-6)
        print(
            f"[progress] {camera} | 100.00% | frame {frame}/{total} | {rate:.2f} FPS "
            f"| ETA 0s | wall={self.span(wall)} | samples={samples} batches={batches} "
            f"tracklets={len(tracks)} complete={complete}",
            flush=True,
        )
        if samples == 0 or complete == 0:
            raise RuntimeError(
                f"{camera}: multimodel extraction produced no usable complete tracklets "
                f"(samples={samples}, complete={complete})"
            )

    def render(self, mapping):
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
            total = int(meta.get("frames", 0))
            start = self.clock()
            mark = start
            try:
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    frame += 1
                    for item in rows.get(frame, []):
                        gid = mapping.get(item["tracklet_key"], "UNKNOWN")
                        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
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
                        text = self.label_text_colour(colour)
                        cv2.putText(
                            image,
                            label,
                            (bx1 + pad_x, by2 - pad_y - base),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scale,
                            text,
                            thickness,
                            cv2.LINE_AA,
                        )
                    cv2.putText(image, camera, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    writer.write(image)
                    now = self.clock()
                    if now - mark >= self.step:
                        wall = max(now - start, 1e-6)
                        rate = frame / wall
                        left = max(0, total - frame)
                        eta = left / rate if rate > 0 else 0.0
                        done = 100.0 * frame / total if total > 0 else 100.0
                        print(
                            f"[progress] {camera} render | {done:6.2f}% | frame {frame}/{total} "
                            f"| {rate:.2f} FPS | ETA {self.span(eta)}",
                            flush=True,
                        )
                        mark = now
            finally:
                cap.release()
                writer.release()
            print(f"[progress] {camera} render | 100.00% | wrote {out}", flush=True)

    def run(self, values):
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")
        cameras = [item[0] for item in sources]
        whole = self.clock()
        print(f"[state-final] ResNet: {self.extractor.describe()}")
        print(f"[state-final] Swin:   {self.swin.describe()}")
        print(f"[state-final] SOLIDER:{self.solider.describe()}")
        print("[state-final] FACE: OFF")
        print(f"[state-final] cameras: {', '.join(cameras)}")
        print(f"[state-final] performance: cross-sample batch={self.batch}, progress={self.step:.1f}s")
        print("[state-final] pass 1: detect + track + verified multimodel view extraction")
        for index, pair in enumerate(sources, 1):
            camera, path = pair
            print(f"[progress] pass 1 camera {index}/{len(sources)}: {camera}", flush=True)
            self.collect(camera, path)
        self.save_cache()
        print(f"[progress] pass 1 | 100.00% | all {len(sources)} cameras complete", flush=True)

        local_mapping: Dict[str, str] = {}
        print("[state-final] pass 2: protected V6 local appearance proposals")
        for index, camera in enumerate(cameras, 1):
            start = self.clock()
            subset = {key: track for key, track in self.tracks.items() if track.camera == camera}
            mapping, _ = GlobalIdentityBodyV6(self.cfg["identity_v6"]).run(subset)
            local_mapping.update(mapping)
            print(
                f"[progress] pass 2 camera {index}/{len(cameras)} | 100.00% | {camera} "
                f"| tracklets={len(subset)} | local_ids={len(set(mapping.values()))} "
                f"| wall={self.span(self.clock() - start)}",
                flush=True,
            )

        print("[state-final] pass 3: tracker-reset repair + state-invariant MTMC")
        start = self.clock()
        resolver = StateInvariantFinalResolver(dict(self.cfg["identity_v6"]), registry=self.registry)
        global_mapping, components, edges = resolver.resolve(local_mapping, self.tracks, cameras)
        debug = {"local_mapping": local_mapping, "global_mapping": global_mapping, "components": components, "edges": edges}
        (self.out / "state_invariant_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
        same = sum(1 for edge in edges if str(edge.get("left", "")).split(":", 1)[0] == str(edge.get("right", "")).split(":", 1)[0])
        cross = len(edges) - same
        print(
            f"[progress] pass 3 | 100.00% | edges={len(edges)} same={same} cross={cross} "
            f"global_ids={len(set(global_mapping.values()))} | wall={self.span(self.clock() - start)}",
            flush=True,
        )
        print(f"[state-final] accepted same-camera repairs: {same}")
        print(f"[state-final] accepted cross-camera links: {cross}")
        print(f"[state-final] final global IDs: {len(set(global_mapping.values()))}")
        print(f"[state-final] persistent global IDs: {self.registry.gids()}")

        print("[state-final] pass 4: render")
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
        print(f"[progress] job | 100.00% | total wall={self.span(self.clock() - whole)}", flush=True)
        return global_mapping
