from __future__ import annotations

import json
from pathlib import Path

import cv2

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

    def _solve_camera(self, camera, path, tracker_rows):
        byframe = {}
        for item in tracker_rows:
            byframe.setdefault(int(item["frame"]), []).append(item)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        recovery_frames = max(1, int(self.cfg["identity"].get("recovery_frames", 10)))
        labels = []
        previous_overlap = set()
        last_gids = {}
        overlap_anchors = {}
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

                active_overlap = self._overlap_ids(current)
                if previous_overlap - active_overlap:
                    recovery_until = max(recovery_until, frame + recovery_frames)
                recovery_mode = frame <= recovery_until

                # Every overlap frame still runs the complete feature stack, but
                # its assignments are quarantined. Nothing learned from a
                # mixed/occluded crop is committed to the identity gallery.
                #
                # After overlap, recovery=True forces feature-only identity
                # assignment. NvDCF track_id is never used to choose the GID.
                feature_map = self.identity.observe(
                    image,
                    current,
                    commit=not bool(active_overlap),
                    recovery=bool(active_overlap) or recovery_mode,
                )
                if active_overlap:
                    # Keep the last clean, feature-resolved identities visible
                    # during the overlap. These anchors are temporary display/
                    # bookkeeping only: they are never written to identity memory
                    # and are never used after the overlap ends.
                    if not previous_overlap:
                        overlap_anchors = {
                            int(item["track_id"]): str(
                                last_gids.get(int(item["track_id"]), "PENDING")
                            )
                            for item in current
                            if int(item["track_id"]) in active_overlap
                        }
                    anchor_used = {
                        value
                        for value in overlap_anchors.values()
                        if str(value).startswith("G")
                    }
                    for item in current:
                        tid = int(item["track_id"])
                        if tid in overlap_anchors and str(overlap_anchors[tid]).startswith("G"):
                            continue
                        candidate = str(
                            feature_map.get(tid, "PENDING")
                        )
                        if candidate.startswith("G") and candidate not in anchor_used:
                            overlap_anchors[tid] = candidate
                            anchor_used.add(candidate)
                        else:
                            overlap_anchors[tid] = "PENDING"
                    gids = {
                        int(item["track_id"]): str(
                            overlap_anchors.get(int(item["track_id"]), "PENDING")
                        )
                        for item in current
                    }
                else:
                    gids = {
                        int(item["track_id"]): str(
                            feature_map.get(int(item["track_id"]), "PENDING")
                        )
                        for item in current
                    }
                    overlap_anchors = {}

                # Hard same-frame collision invariant. Re-solve the whole frame
                # with feature-only recovery, then keep any unresolved collision
                # as PENDING instead of emitting a false merge.
                grouped = {}
                for tid, gid in gids.items():
                    if gid.startswith("G"):
                        grouped.setdefault(gid, []).append(tid)
                if any(len(items) > 1 for items in grouped.values()):
                    self.identity.stats["duplicate_frames"] += 1
                    feature_map = self.identity.observe(image, current, commit=False, recovery=True)
                    gids = {int(item["track_id"]): str(feature_map.get(int(item["track_id"]), "PENDING")) for item in current}

                used = set()
                for tid in list(gids):
                    gid = gids[tid]
                    if gid.startswith("G") and gid in used:
                        gids[tid] = "PENDING"
                    elif gid.startswith("G"):
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
                    })

                if not active_overlap:
                    last_gids = dict(gids)
                previous_overlap = set(active_overlap)
        finally:
            cap.release()

        by_frame = {}
        for item in labels:
            by_frame.setdefault(int(item["frame"]), []).append(item)
        for frame_id, items in by_frame.items():
            gids = [str(item["gid"]) for item in items if str(item["gid"]).startswith("G")]
            if len(gids) != len(set(gids)):
                raise RuntimeError(f"same-frame duplicate GID survived validation: {camera}:{frame_id}")

        target = self.cache / f"{camera}.labels.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for item in labels:
                handle.write(json.dumps(item) + "\n")
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
        print(f"[nvdcf] FACE: {self.identity.face.describe()}")
        print(f"[nvdcf] POSE: {self.cfg['pose']['model']}")
        print("[nvdcf] QDRANT: ENABLED")
        print("[nvdcf] GID assignment: feature-only after overlap and one-to-one every frame")

        all_labels = []
        try:
            for camera, path in sources:
                target = self.cache / f"{camera}.tracker.jsonl"
                fps, width, height = self.det.track(camera, path, target)
                del fps, width, height
                all_labels.extend(self._solve_camera(camera, path, self._load(target)))

            debug = {
                "tracker": "NVIDIA NvDCF",
                "identity": "Qdrant + reliable face + top clothing + bottom clothing + NVIDIA ResNet + NVIDIA Swin + SOLIDER + pose",
                "face_visibility_threshold": float(self.identity.face_threshold),
                "post_overlap_identity": "feature_only",
                "tracker_id_global_fallback": False,
                "same_frame_gid_invariant": True,
                "new_gids": int(self.identity.stats["new"]),
                "duplicate_frames": int(self.identity.stats["duplicate"]),
                "pending_frames": int(self.identity.stats["pending"]),
                "recovery_frames": int(self.identity.stats["recovery"]),
                "recovery_matches": int(self.identity.stats["recovery_match"]),
                "cross_camera_matches": int(self.identity.stats["cross"]),
                "pending_new_observations": int(self.identity.stats["pending_new_observations"]),
                "pending_new_confirmed": int(self.identity.stats["pending_new_confirmed"]),
                "pending_new_observations": int(self.identity.stats["pending_new_observations"]),
                "pending_new_confirmed": int(self.identity.stats["pending_new_confirmed"]),
            }
            (self.out / "nvdcf_identity_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
            print(f"[nvdcf] result: new_gids={debug['new_gids']} recovery_matches={debug['recovery_matches']} cross_camera_matches={debug['cross_camera_matches']} duplicate_frames={debug['duplicate_frames']} pending_frames={debug['pending_frames']}")
            return all_labels
        finally:
            self.identity.close()


__all__ = ["BatchNvDCF"]