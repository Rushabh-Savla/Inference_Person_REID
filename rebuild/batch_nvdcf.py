from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from rebuild.overlap_guard import merge

from rebuild.multimodal_identity import MultiModal
from rebuild.nvdcf_tracker import NvDCF


class BatchNvDCF:
    """Real NvDCF tracking with strict multimodal, one-to-one global IDs."""

    def __init__(self, config_path):
        import yaml

        with open(config_path, "r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle) or {}
        self.out = Path(self.cfg["input"]["output_dir"])
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
        matches = [item for item in Path.cwd().rglob(path.name) if item.is_file()]
        if len(matches) == 1:
            return str(matches[0])
        raise RuntimeError(f"Input video not found: {value}")

    @staticmethod
    def _name(path):
        return Path(path).stem

    @staticmethod
    def _load(path):
        output = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    output.append(json.loads(line))
        return output

    def sources(self, values):
        if values:
            return [(self._name(value), self._source(value)) for value in values]
        configured = self.cfg.get("input", {}).get("videos", [])
        output = []
        for item in configured:
            if isinstance(item, dict):
                path = item["path"]
                output.append((str(item.get("name") or self._name(path)), self._source(path)))
            else:
                output.append((self._name(item), self._source(item)))
        return output

    @staticmethod
    def _metrics(left, right):
        ax1, ay1, ax2, ay2 = [float(x) for x in left]
        bx1, by1, bx2, by2 = [float(x) for x in right]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        ab = max(1.0, (bx2 - bx1) * (by2 - by1))
        union = max(1.0, aa + ab - inter)
        return float(inter / union), float(inter / min(aa, ab))

    def _overlap_ids(self, current):
        overlap = self.cfg.get("overlap", {}) or {}
        iou_min = float(overlap.get("iou", 0.35))
        iom_min = float(overlap.get("intersection", 0.50))
        active = set()
        for first in range(len(current)):
            for second in range(first + 1, len(current)):
                iou, iom = self._metrics(current[first]["bbox"], current[second]["bbox"])
                if iou >= iou_min or iom >= iom_min:
                    active.add(int(current[first]["track_id"]))
                    active.add(int(current[second]["track_id"]))
        return active

    def _render(self, camera, path, labels):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for rendering: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = self.out / f"{camera}_nvdcf.mp4"
        writer = cv2.VideoWriter(output.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

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
                    if gid.startswith("G"):
                        if gid in used:
                            raise RuntimeError(f"same-frame duplicate GID {gid} in {camera} frame {frame}")
                        used.add(gid)
                    x1, y1, x2, y2 = [int(round(float(value))) for value in item["bbox"]]
                    if gid == "PENDING":
                        draw = (40, 80, 220)
                    else:
                        number = int(gid[1:])
                        draw = (40 + (number * 67) % 180, 90 + (number * 43) % 150, 70 + (number * 29) % 170)
                    label = gid
                    if item.get("overlap"):
                        label += " OV"
                    elif item.get("recovery"):
                        label += " REC"
                    cv2.rectangle(image, (x1, y1), (x2, y2), draw, 3)
                    cv2.putText(image, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, draw, 2, cv2.LINE_AA)
                cv2.putText(image, f"{camera} | NVIDIA NvDCF | multimodal ReID", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(image)
        finally:
            cap.release()
            writer.release()
        return output

    def _frame_gids(self, current, feature_map):
        """Resolve a frame strictly from feature scores; tracker IDs are keys only."""
        return {
            int(item["track_id"]): str(
                feature_map.get(int(item["track_id"]), "PENDING")
            )
            for item in current
        }

    @staticmethod
    def _collision(gids):
        seen = {}
        for tid, gid in gids.items():
            if not str(gid).startswith("G"):
                continue
            seen.setdefault(str(gid), []).append(int(tid))
        return {gid: tids for gid, tids in seen.items() if len(tids) > 1}

    def _solve_camera(self, camera, path, tracker_rows):
        detections_path = self.cache / f"{camera}.tracker.detections.jsonl"
        detections = self._load(detections_path)
        detections_byframe = {}
        for item in detections:
            detections_byframe.setdefault(int(item["frame"]), []).append(item)

        byframe = {}
        for item in tracker_rows:
            byframe.setdefault(int(item["frame"]), []).append(item)
        for frame_id, values in detections_byframe.items():
            byframe[frame_id] = merge(byframe.get(frame_id, []), values, frame_id)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        recovery_frames = max(1, int(self.cfg["identity"].get("recovery_frames", 10)))
        labels = []
        previous_overlap = set()
        recovery_until = -1
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

                tids = [int(item["track_id"]) for item in current]
                if len(tids) != len(set(tids)):
                    raise RuntimeError(
                        f"NvDCF emitted duplicate tracker IDs in one frame: {camera}:{frame}:{tids}"
                    )

                geometric_overlap = self._overlap_ids(current)
                collapse_overlap = {
                    int(item["track_id"])
                    for item in current
                    if str(item.get("shadow_reason", "")) == "collapse"
                }
                active_overlap = geometric_overlap | collapse_overlap
                ended = previous_overlap - active_overlap
                if ended:
                    recovery_until = max(recovery_until, frame + recovery_frames)
                recovery_mode = bool(active_overlap) or frame <= recovery_until

                # The resolver sees every person independently and returns IDs from
                # Qdrant-backed multimodal feature matching. Overlap changes only
                # the write/admission policy; it never changes the identity source.
                feature_map = self.identity.observe(
                    image,
                    current,
                    commit=not recovery_mode,
                    recovery=recovery_mode,
                )
                gids = self._frame_gids(current, feature_map)

                collisions = self._collision(gids)
                if collisions:
                    self.identity.stats["duplicate"] += 1
                    # Re-extract and re-solve the complete frame in recovery mode.
                    # This is still feature-only and cannot mint an identity.
                    feature_map = self.identity.observe(
                        image,
                        current,
                        commit=False,
                        recovery=True,
                    )
                    gids = self._frame_gids(current, feature_map)
                    collisions = self._collision(gids)

                # Never emit two established global identities for two detections
                # in one frame. Keep unresolved evidence pending instead of forcing
                # a false merge. The tracker ID is deliberately not consulted here.
                if collisions:
                    for gid, tids_for_gid in collisions.items():
                        for tid in tids_for_gid[1:]:
                            gids[tid] = "PENDING"

                used = set()
                for tid in list(gids):
                    gid = str(gids[tid])
                    if gid.startswith("G"):
                        if gid in used:
                            gids[tid] = "PENDING"
                        else:
                            used.add(gid)

                for item in current:
                    tid = int(item["track_id"])
                    labels.append({
                        "camera": camera,
                        "frame": frame,
                        "track_id": tid,
                        "bbox": item["bbox"],
                        "gid": gids[tid],
                        "overlap": tid in active_overlap,
                        "recovery": bool(recovery_mode),
                        "identity_source": "multimodal_qdrant",
                    })

                previous_overlap = set(active_overlap)
        finally:
            cap.release()

        by_frame = {}
        for item in labels:
            by_frame.setdefault(int(item["frame"]), []).append(item)
        for frame_id, items in by_frame.items():
            gids = [str(item["gid"]) for item in items if str(item["gid"]).startswith("G")]
            if len(gids) != len(set(gids)):
                raise RuntimeError(
                    f"same-frame duplicate GID survived validation: {camera}:{frame_id}"
                )

        target = self.cache / f"{camera}.labels.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for item in labels:
                handle.write(json.dumps(item) + "\\n")
        output = self._render(camera, path, labels)
        print(f"[nvdcf] wrote {output}")
        return labels

    def run(self, values):
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")

        print("[nvdcf] PRIMARY TRACKER: NVIDIA NvDCF")
        print(f"[nvdcf] ResNet: {self.identity.resnet.describe()}")
        print(f"[nvdcf] Swin: {self.identity.swin.describe()}")
        print(f"[nvdcf] SOLIDER: {self.identity.solider.describe()}")
        print(f"[nvdcf] POSE: {self.cfg['pose']['model']}")
        print("[nvdcf] QDRANT: ENABLED — candidate retrieval participates in identity comparison")
        print("[nvdcf] FACE: InsightFace SCRFD + ArcFace, reliable face has highest fusion priority")
        print("[nvdcf] CLOTHING: top + bottom + upper/lower pattern every comparison")
        print("[nvdcf] POSE: YOLO pose participates in matching")
        print("[nvdcf] GID assignment: feature-only after overlap; Hungarian one-to-one every frame")

        all_labels = []
        try:
            for camera, path in sources:
                target = self.cache / f"{camera}.tracker.jsonl"
                fps, width, height = self.det.track(camera, path, target)
                del fps, width, height
                all_labels.extend(self._solve_camera(camera, path, self._load(target)))

            unique_gids = sorted(
                {
                    str(item["gid"])
                    for item in all_labels
                    if str(item["gid"]).startswith("G")
                }
            )
            debug = {
                "tracker": "NVIDIA NvDCF",
                "identity": "Qdrant + top clothing + bottom clothing + NVIDIA ResNet + NVIDIA Swin + SOLIDER + pose",
                "post_overlap_identity": "feature_only",
                "tracker_gid_fallback": False,
                "same_frame_gid_invariant": True,
                "unique_global_ids": int(len(unique_gids)),
                "max_global_id": int(max(
                    (int(value[1:]) for value in unique_gids),
                    default=0,
                )),
                "new_gids": int(self.identity.stats["new"]),
                "duplicate_frames": int(self.identity.stats["duplicate"]),
                "pending_frames": int(self.identity.stats["pending"]),
                "recovery_frames": int(self.identity.stats["recovery"]),
                "recovery_matches": int(self.identity.stats["recovery_match"]),
                "cross_camera_matches": int(self.identity.stats["cross"]),
                "pending_new_observations": int(self.identity.stats["pending_new_observations"]),
                "pending_new_confirmed": int(self.identity.stats["pending_new_confirmed"]),
                "overlap_frames": int(sum(1 for x in all_labels if x.get("overlap"))),
                "collapse_frames": int(sum(1 for x in all_labels if x.get("overlap") and x.get("track_id", 0) < 0)),
                "face_observations": int(self.identity.stats["face_observations"]),
                "face_reliable": int(self.identity.stats["face_reliable"]),
                "qdrant_retrievals": int(self.identity.stats["qdrant_retrievals"]),
                "memory_reject": int(self.identity.stats.get("memory_reject", 0)),
            }
            (self.out / "nvdcf_identity_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
            print(
                f"[nvdcf] result: new_gids={debug['new_gids']} "
                f"recovery_matches={debug['recovery_matches']} "
                f"cross_camera_matches={debug['cross_camera_matches']} "
                f"duplicate_frames={debug['duplicate_frames']} "
                f"pending_frames={debug['pending_frames']} "
                f"face_reliable={debug['face_reliable']} "
                f"qdrant_retrievals={debug['qdrant_retrievals']}"
            )
            return all_labels
        finally:
            self.identity.close()


__all__ = ["BatchNvDCF"]