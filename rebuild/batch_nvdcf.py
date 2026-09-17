
from __future__ import annotations

import json
from pathlib import Path

import cv2

from rebuild.multimodal_identity import MultiModal
from rebuild.nvdcf_tracker import NvDCF


class BatchNvDCF:
    """NvDCF tracking with feature-first, one-to-one multimodal GID assignment."""

    def __init__(self, config_path):
        import yaml

        with open(config_path, "r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle) or {}

        self.out = Path(
            self.cfg.get("input", {}).get(
                "output_dir", "rebuild_outputs_nvdcf"
            )
        )
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out / "cache_nvdcf"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.det = NvDCF(self.cfg["detector"])
        self.identity = MultiModal(self.cfg)

    @staticmethod
    def _source(value):
        path = Path(str(value)).expanduser()
        if path.is_file():
            return str(path)
        matches = list(Path.cwd().rglob(path.name))
        if len(matches) == 1:
            return str(matches[0])
        raise RuntimeError(f"Input video not found: {value}")

    @staticmethod
    def _name(path):
        return Path(path).stem

    @staticmethod
    def _load(path):
        data = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def sources(self, values):
        if values:
            return [
                (self._name(value), self._source(value))
                for value in values
            ]
        configured = self.cfg.get("input", {}).get("videos", [])
        out = []
        for item in configured:
            if isinstance(item, dict):
                path = item["path"]
                out.append(
                    (
                        str(item.get("name") or self._name(path)),
                        self._source(path),
                    )
                )
            else:
                out.append((self._name(item), self._source(item)))
        return out

    @staticmethod
    def _metrics(left, right):
        ax1, ay1, ax2, ay2 = [float(x) for x in left]
        bx1, by1, bx2, by2 = [float(x) for x in right]
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        ab = max(1.0, (bx2 - bx1) * (by2 - by1))
        union = max(1.0, aa + ab - inter)
        iou = inter / union
        iom = inter / min(aa, ab)
        return float(iou), float(iom)

    def _overlap_ids(self, current):
        overlap_iou = float(
            (self.cfg.get("overlap", {}) or {}).get("iou", 0.45)
        )
        overlap_iom = float(
            (self.cfg.get("overlap", {}) or {}).get(
                "intersection", 0.60
            )
        )
        active = set()
        for index in range(len(current)):
            for other in range(index + 1, len(current)):
                iou, iom = self._metrics(
                    current[index]["bbox"],
                    current[other]["bbox"],
                )
                if iou >= overlap_iou and iom >= overlap_iom:
                    active.add(int(current[index]["track_id"]))
                    active.add(int(current[other]["track_id"]))
        return active

    def _render(self, camera, path, labels):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for rendering: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = self.out / f"{camera}_nvdcf.mp4"
        writer = cv2.VideoWriter(
            str(out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        grouped = {}
        for item in labels:
            grouped.setdefault(int(item["frame"]), []).append(item)

        frame = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                frame += 1
                used = set()
                for item in grouped.get(frame, []):
                    gid = str(item["gid"])
                    if gid.startswith("G") and gid in used:
                        raise RuntimeError(
                            f"same-frame duplicate global ID {gid} "
                            f"in {camera} frame {frame}"
                        )
                    if gid.startswith("G"):
                        used.add(gid)

                    x1, y1, x2, y2 = [
                        int(round(float(value)))
                        for value in item["bbox"]
                    ]
                    if gid == "PENDING":
                        draw = (40, 80, 220)
                    else:
                        number = int(gid[1:]) if gid.startswith("G") else 0
                        draw = (
                            40 + (number * 67) % 180,
                            90 + (number * 43) % 150,
                            70 + (number * 29) % 170,
                        )

                    label = gid
                    if item.get("overlap"):
                        label = f"{label} OV"
                    elif item.get("recovery"):
                        label = f"{label} REC"

                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        draw,
                        3,
                    )
                    cv2.putText(
                        image,
                        label,
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        draw,
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    image,
                    f"{camera} | NvDCF | multimodal feature-first ReID",
                    (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(image)
        finally:
            cap.release()
            writer.release()
        print(f"[nvdcf] wrote {out}")

    def _solve_camera(self, camera, path, tracker_rows):
        byframe = {}
        for item in tracker_rows:
            byframe.setdefault(int(item["frame"]), []).append(item)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        interval = int(
            self.cfg.get("identity", {}).get("interval", 2)
        )
        recovery_frames = int(
            self.cfg.get("identity", {}).get(
                "recovery_frames", 8
            )
        )

        labels = []
        last = {}
        previous_overlap = set()
        recovery_left = {}
        frame = 0

        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                frame += 1
                current = byframe.get(frame, [])
                if not current:
                    continue

                self.identity.stats["frames"] += 1
                active_overlap = self._overlap_ids(current)
                entered = active_overlap - previous_overlap
                exited = previous_overlap - active_overlap

                for tid in exited:
                    recovery_left[(str(camera), int(tid))] = recovery_frames

                current_ids = {
                    int(item["track_id"])
                    for item in current
                }
                for key in list(recovery_left):
                    if key[0] == str(camera) and key[1] not in current_ids:
                        recovery_left.pop(key, None)

                recovering_now = {
                    int(tid)
                    for (cam, tid), remaining in recovery_left.items()
                    if cam == str(camera) and remaining > 0
                }

                force = bool(
                    active_overlap
                    or entered
                    or exited
                    or recovering_now
                )
                need = force or any(
                    frame - int(
                        last.get(
                            int(item["track_id"]),
                            -10**9,
                        )
                    ) >= interval
                    for item in current
                )

                feature_map = {}
                if need:
                    # Active overlap is resolved from features but is
                    # quarantined: these blended crops cannot alter memory.
                    feature_map = self.identity.observe(
                        image,
                        current,
                        commit=not bool(active_overlap),
                    )

                gids = {}
                for item in current:
                    tid = int(item["track_id"])
                    key = (str(camera), tid)
                    recovering = (
                        key in recovery_left
                        and recovery_left[key] > 0
                    )

                    if tid in feature_map:
                        gid = str(feature_map[tid])
                    elif active_overlap and key in self.identity.trackmap:
                        # Temporal continuity during overlap only. This does
                        # not assign or create a global identity.
                        gid = f"G{int(self.identity.trackmap[key]):06d}"
                    elif recovering:
                        # After overlap, the tracker ID is explicitly forbidden
                        # as a GID fallback. Wait for feature confirmation.
                        gid = "PENDING"
                    else:
                        prior = self.identity.trackmap.get(key)
                        gid = (
                            f"G{int(prior):06d}"
                            if prior is not None
                            else "PENDING"
                        )
                    gids[tid] = gid

                duplicates = {}
                for tid, gid in gids.items():
                    if gid.startswith("G"):
                        duplicates.setdefault(gid, []).append(tid)
                bad = {
                    gid: tids
                    for gid, tids in duplicates.items()
                    if len(tids) > 1
                }

                if bad:
                    # Re-run all people in this frame with multimodal
                    # one-to-one assignment before accepting any duplicate.
                    feature_map = self.identity.observe(
                        image,
                        current,
                        commit=not bool(active_overlap),
                    )
                    gids = {}
                    for item in current:
                        tid = int(item["track_id"])
                        key = (str(camera), tid)
                        recovering = (
                            key in recovery_left
                            and recovery_left[key] > 0
                        )
                        if tid in feature_map:
                            gids[tid] = str(feature_map[tid])
                        elif active_overlap and key in self.identity.trackmap:
                            gids[tid] = (
                                f"G{int(self.identity.trackmap[key]):06d}"
                            )
                        elif recovering:
                            gids[tid] = "PENDING"
                        else:
                            gids[tid] = self.identity.trackmap.get(
                                key,
                                "PENDING",
                            )
                    self.identity.stats["duplicate_frames"] += 1

                # Final hard collision gate. A duplicated GID is never rendered.
                used = set()
                for tid in list(gids):
                    gid = str(gids[tid])
                    if gid.startswith("G") and gid in used:
                        gids[tid] = "PENDING"
                    elif gid.startswith("G"):
                        used.add(gid)

                for item in current:
                    tid = int(item["track_id"])
                    key = (str(camera), tid)
                    gid = str(gids[tid])
                    recovering = (
                        key in recovery_left
                        and recovery_left[key] > 0
                    )
                    labels.append(
                        {
                            "camera": camera,
                            "frame": frame,
                            "track_id": tid,
                            "bbox": item["bbox"],
                            "gid": gid,
                            "overlap": tid in active_overlap,
                            "recovery": bool(recovering),
                        }
                    )

                    if gid.startswith("G") and not recovering:
                        self.identity.trackmap[key] = int(gid[1:])
                        last[tid] = frame

                for tid in list(recovering_now):
                    key = (str(camera), int(tid))
                    gid = str(gids.get(int(tid), "PENDING"))
                    if gid.startswith("G"):
                        recovery_left.pop(key, None)
                    else:
                        recovery_left[key] = max(
                            0,
                            recovery_left[key] - 1,
                        )
                        if recovery_left[key] <= 0:
                            recovery_left.pop(key, None)

                # New overlap starts replace any stale post-overlap recovery state.
                for tid in entered:
                    recovery_left.pop(
                        (str(camera), int(tid)),
                        None,
                    )

                previous_overlap = set(active_overlap)
        finally:
            cap.release()

        # Independent full-output validation.
        checks = {}
        for item in labels:
            gid = str(item["gid"])
            if gid.startswith("G"):
                checks.setdefault(
                    int(item["frame"]), set()
                ).add(gid)
        for frameid, gids in checks.items():
            total = sum(
                1
                for item in labels
                if int(item["frame"]) == frameid
                and str(item["gid"]).startswith("G")
            )
            if len(gids) != total:
                raise RuntimeError(
                    "same-frame duplicate GID survived validation: "
                    f"{camera}:{frameid}"
                )

        target = self.cache / f"{camera}.labels.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for item in labels:
                handle.write(json.dumps(item) + "\n")

        self._render(camera, path, labels)
        return labels

    def run(self, values):
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")

        metas = []
        print("[nvdcf] PRIMARY TRACKER: NVIDIA NvDCF")
        print(f"[nvdcf] ResNet: {self.identity.resnet.describe()}")
        print(f"[nvdcf] Swin: {self.identity.swin.describe()}")
        print(f"[nvdcf] SOLIDER: {self.identity.solider.describe()}")
        if self.identity.face is None:
            raise RuntimeError(
                "Face extractor is mandatory in the final multimodal pipeline"
            )
        print(f"[nvdcf] FACE: {self.identity.face.describe()}")
        print(
            f"[nvdcf] POSE: "
            f"{'ON' if self.identity.pose is not None else 'OFF'}"
        )
        print("[nvdcf] Qdrant: ENABLED")
        print(
            "[nvdcf] GID source: multimodal features; "
            "tracker ID is temporal bookkeeping only"
        )

        for camera, path in sources:
            target = self.cache / f"{camera}.tracker.jsonl"
            fps, width, height = self.det.track(
                camera,
                path,
                target,
            )
            metas.append(
                (
                    camera,
                    path,
                    fps,
                    width,
                    height,
                    target,
                )
            )

        all_labels = []
        for camera, path, _fps, _width, _height, target in metas:
            labels = self._solve_camera(
                camera,
                path,
                self._load(target),
            )
            all_labels.extend(labels)

        debug = {
            "tracker": "NVIDIA NvDCF",
            "global_assignment": "multimodal_feature_matching",
            "required_features": [
                "face_when_visibility_at_least_0.68",
                "top_clothing_color",
                "bottom_clothing_color",
                "top_clothing_pattern",
                "bottom_clothing_pattern",
                "nvidia_resnet",
                "nvidia_swin",
                "solider",
                "pose",
                "qdrant",
            ],
            "same_frame_duplicate_gids": int(
                self.identity.stats["duplicate_frames"]
            ),
            "stats": self.identity.stats,
            "gids": sorted(self.identity.profiles),
            "labels": len(all_labels),
        }
        (self.out / "identity_debug_nvdcf.json").write_text(
            json.dumps(debug, indent=2),
            encoding="utf-8",
        )

        count = len(self.identity.profiles)
        duplicates = int(
            self.identity.stats["duplicate_frames"]
        )
        self.identity.close()
        print(f"[nvdcf] FINAL GLOBAL IDS: {count}")
        print(
            f"[nvdcf] SAME-FRAME DUPLICATE GIDS: {duplicates}"
        )
        print(f"[nvdcf] OUTPUTS: {self.out}")
        return {
            "labels": all_labels,
            "gids": sorted(self.identity.profiles),
        }
