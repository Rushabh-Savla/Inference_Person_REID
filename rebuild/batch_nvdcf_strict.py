from __future__ import annotations

import json
from pathlib import Path

import cv2

from rebuild.multimodal_identity_strict import MultiModalStrict
from rebuild.nvdcf_tracker import NvDCF


class BatchNvDCFStrict:
    """NvDCF tracking with feature-only global identity recovery."""

    def __init__(self, path):
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle) or {}
        self.out = Path(self.cfg["input"]["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out / "cache_nvdcf"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.det = NvDCF(self.cfg["detector"])
        self.identity = MultiModalStrict(self.cfg)
        ov = self.cfg.get("overlap", {}) or {}
        self.enter_iou = float(ov.get("enter_iou", ov.get("iou", 0.30)))
        self.enter_iom = float(ov.get("enter_iom", ov.get("intersection", 0.50)))
        self.exit_iou = float(ov.get("exit_iou", 0.18))
        self.exit_iom = float(ov.get("exit_iom", 0.32))
        self.grace = max(1, int(ov.get("grace", 2)))
        self.recovery = max(1, int(self.cfg["identity"].get("recovery_frames", 10)))

    @staticmethod
    def source(value):
        item = Path(str(value)).expanduser()
        if item.is_file():
            return str(item)
        found = [x for x in Path.cwd().rglob(item.name) if x.is_file()]
        if len(found) == 1:
            return str(found[0])
        raise RuntimeError(f"Input video not found: {value}")

    @staticmethod
    def name(path):
        return Path(path).stem

    @staticmethod
    def load(path):
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def sources(self, values):
        if values:
            return [(self.name(self.source(x)), self.source(x)) for x in values]
        result = []
        for item in self.cfg.get("input", {}).get("videos", []):
            if isinstance(item, dict):
                path = self.source(item["path"])
                result.append((str(item.get("name") or self.name(path)), path))
            else:
                path = self.source(item)
                result.append((self.name(path), path))
        return result

    @staticmethod
    def metrics(left, right):
        ax1, ay1, ax2, ay2 = [float(x) for x in left]
        bx1, by1, bx2, by2 = [float(x) for x in right]
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        ab = max(1.0, (bx2 - bx1) * (by2 - by1))
        union = max(1.0, aa + ab - inter)
        return float(inter / union), float(inter / min(aa, ab))

    def overlaps(self, rows, states):
        active = set()
        pairs = set()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left = rows[i]
                right = rows[j]
                a = int(left["track_id"])
                b = int(right["track_id"])
                key = tuple(sorted((a, b)))
                pairs.add(key)
                iou, iom = self.metrics(left["bbox"], right["bbox"])
                hit = bool(iou >= self.enter_iou or iom >= self.enter_iom)
                state = states.get(key)
                if state is None:
                    states[key] = {"active": hit, "clear": 0}
                elif state["active"]:
                    if iou <= self.exit_iou and iom <= self.exit_iom:
                        state["clear"] += 1
                        if state["clear"] >= self.grace:
                            state["active"] = False
                            state["clear"] = 0
                    else:
                        state["clear"] = 0
                elif hit:
                    state["active"] = True
                    state["clear"] = 0
                if states[key]["active"]:
                    active.add(a)
                    active.add(b)
        for key in list(states):
            if key not in pairs:
                states.pop(key, None)
        return active

    def render(self, camera, path, labels):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for rendering: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp = self.out / f"{camera}_strict_tmp.mp4"
        output = self.out / f"{camera}_nvdcf.mp4"
        writer = cv2.VideoWriter(
            temp.as_posix(),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        groups = {}
        for item in labels:
            groups.setdefault(int(item["frame"]), []).append(item)
        frame = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                frame += 1
                used = set()
                for item in groups.get(frame, []):
                    gid = str(item["gid"])
                    if gid.startswith("G"):
                        if gid in used:
                            raise RuntimeError(
                                f"same-frame duplicate GID {gid}: {camera}:{frame}"
                            )
                        used.add(gid)
                    box = [int(round(float(x))) for x in item["bbox"]]
                    x1, y1, x2, y2 = box
                    if gid == "PENDING":
                        draw = (20, 80, 235)
                    else:
                        number = int(gid[1:])
                        draw = (
                            45 + (number * 61) % 170,
                            70 + (number * 37) % 160,
                            55 + (number * 29) % 180,
                        )
                    label = gid
                    if item.get("overlap"):
                        label += " OV"
                    elif item.get("recovery"):
                        label += " REC"
                    cv2.rectangle(image, (x1, y1), (x2, y2), draw, 3)
                    cv2.putText(
                        image,
                        label,
                        (x1, max(28, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        draw,
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    image,
                    f"{camera} | NVIDIA NvDCF | face+ReID+top+bottom+pose+Qdrant",
                    (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.78,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(image)
        finally:
            cap.release()
            writer.release()
        temp.replace(output)
        return output

    def solve(self, camera, path, tracker):
        byframe = {}
        for item in tracker:
            byframe.setdefault(int(item["frame"]), []).append(item)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        labels = []
        states = {}
        previous = set()
        until = -1
        frame = 0
        overlaps = 0
        collisions = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                frame += 1
                current = byframe.get(frame, [])
                if not current:
                    continue
                active = self.overlaps(current, states)
                if active:
                    overlaps += 1
                if previous - active:
                    until = max(until, frame + self.recovery)
                recover = frame <= until
                gids = self.identity.observe(
                    image,
                    current,
                    commit=not bool(active),
                    recovery=bool(recover),
                )
                if len(gids) != len(current):
                    raise RuntimeError(
                        f"identity output length mismatch: {camera}:{frame}"
                    )
                used = {}
                for index, gid in enumerate(gids):
                    text = str(gid)
                    if text.startswith("G"):
                        used.setdefault(text, []).append(index)
                dup = [key for key, value in used.items() if len(value) > 1]
                if dup:
                    collisions += 1
                    gids = self.identity.observe(
                        image,
                        current,
                        commit=False,
                        recovery=True,
                    )
                    used = {}
                    for index, gid in enumerate(gids):
                        text = str(gid)
                        if text.startswith("G"):
                            used.setdefault(text, []).append(index)
                    if any(len(value) > 1 for value in used.values()):
                        raise RuntimeError(
                            f"same-frame duplicate GID survived feature re-solve: {camera}:{frame}"
                        )
                for index, item in enumerate(current):
                    labels.append(
                        {
                            "camera": camera,
                            "frame": frame,
                            "track_id": int(item["track_id"]),
                            "bbox": item["bbox"],
                            "gid": str(gids[index]),
                            "overlap": int(item["track_id"]) in active,
                            "recovery": bool(recover),
                        }
                    )
                previous = set(active)
        finally:
            cap.release()
        target = self.cache / f"{camera}.labels.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for item in labels:
                handle.write(json.dumps(item) + "\n")
        output = self.render(camera, path, labels)
        print(
            f"[nvdcf] {camera}: frames={frame} labels={len(labels)} "
            f"overlap_frames={overlaps} collision_frames={collisions} output={output}"
        )
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
        print("[nvdcf] CLOTHING: top + bottom mandatory")
        print("[nvdcf] QDRANT: ENABLED")
        print("[nvdcf] POST-OVERLAP GID: feature-only, tracker ID forbidden")
        all_labels = []
        try:
            for camera, path in sources:
                target = self.cache / f"{camera}.tracker.jsonl"
                fps, width, height = self.det.track(camera, path, target)
                del fps, width, height
                all_labels.extend(self.solve(camera, path, self.load(target)))
            debug = {
                "tracker": "NVIDIA NvDCF",
                "identity": "Qdrant + face + NVIDIA ResNet + NVIDIA Swin + SOLIDER + top clothing + bottom clothing + pose",
                "post_overlap_identity": "feature_only",
                "same_frame_gid_invariant": True,
                "new_gids": int(self.identity.stats["new"]),
                "duplicate_frames": int(self.identity.stats["duplicate"]),
                "pending_frames": int(self.identity.stats["pending"]),
                "recovery_frames": int(self.identity.stats["recovery"]),
                "recovery_matches": int(self.identity.stats["recovery_match"]),
                "cross_camera_matches": int(self.identity.stats["cross"]),
                "face_frames": int(self.identity.stats["face"]),
                "face_matches": int(self.identity.stats["face_match"]),
            }
            (self.out / "nvdcf_identity_debug.json").write_text(
                json.dumps(debug, indent=2), encoding="utf-8"
            )
            print(f"[nvdcf] RESULT: {json.dumps(debug, sort_keys=True)}")
            return all_labels
        finally:
            self.identity.close()


__all__ = ["BatchNvDCFStrict"]
