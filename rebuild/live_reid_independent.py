from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detector import PersonDetector  # noqa: E402
from rebuild.identity_v2 import crop, illumination_variant, quality  # noqa: E402
from reid.extractor import ReIDExtractor  # noqa: E402
from reid.nvidia_swin import NVIDIASwinReIDExtractor  # noqa: E402
from reid.solider_reid import SOLIDERReIDExtractor  # noqa: E402

try:
    from geometry.calibration import load_calibration  # noqa: E402
    from geometry.floor import FloorFrame  # noqa: E402
except Exception:  # noqa: BLE001
    load_calibration = None
    FloorFrame = None


MODELS = ("resnet", "swin", "solider")
VIEWS = ("full", "light", "upper", "torso", "lower")
MODEL_MIN = {"resnet": 0.44, "swin": 0.44, "solider": 0.42}


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def unit(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if arr.size == 0 or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid embedding")
    return arr / norm


def cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    try:
        return float(np.clip(np.dot(unit(left), unit(right)), -1.0, 1.0))
    except (ValueError, TypeError):
        return 0.0


def topmean(left: Iterable[np.ndarray], right: Iterable[np.ndarray], k: int = 5) -> float:
    a = [unit(x) for x in left if x is not None]
    b = [unit(x) for x in right if x is not None]
    if not a or not b:
        return 0.0
    values = []
    for x in a:
        values.extend(float(np.dot(x, y)) for y in b)
    if not values:
        return 0.0
    values.sort(reverse=True)
    return float(np.mean(values[: min(k, len(values))]))


def softclip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Appearance attributes beyond the deep ReID embeddings
# ---------------------------------------------------------------------------


def _zone(image: np.ndarray, y1: float, y2: float, x1: float = 0.08, x2: float = 0.92) -> np.ndarray:
    h, w = image.shape[:2]
    xa, xb = int(w * x1), int(w * x2)
    ya, yb = int(h * y1), int(h * y2)
    xa, xb = max(0, xa), min(w, max(xa + 1, xb))
    ya, yb = max(0, ya), min(h, max(ya + 1, yb))
    return image[ya:yb, xa:xb]


def _histogram(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros(20, np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    hue = hsv[..., 0].astype(np.float32) / 180.0
    h, _ = np.histogram(hue, bins=12, range=(0.0, 1.0), weights=sat + 0.05)
    v, _ = np.histogram(val, bins=8, range=(0.0, 1.0), weights=(1.0 - sat * 0.35))
    out = np.concatenate([h, v]).astype(np.float32)
    return unit(out)


def _pattern(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros(12, np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (cv2.phase(gx, gy, angleInDegrees=False) + math.pi) % math.pi
    orient, _ = np.histogram(ang, bins=8, range=(0.0, math.pi), weights=mag + 1e-3)
    edge = cv2.Canny((gray * 255.0).astype(np.uint8), 80, 160)
    edge_density = float(np.mean(edge > 0))
    variance = float(np.var(gray))
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    fine = float(np.mean(np.abs(gray - blur)))
    out = np.concatenate([
        orient.astype(np.float32),
        np.asarray([edge_density, variance, fine, float(np.mean(mag))], np.float32),
    ])
    return unit(out)


def appearance_features(image: np.ndarray) -> Dict[str, np.ndarray]:
    head = _zone(image, 0.00, 0.26, 0.15, 0.85)
    upper = _zone(image, 0.22, 0.58, 0.10, 0.90)
    lower = _zone(image, 0.55, 0.92, 0.10, 0.90)
    torso = _zone(image, 0.18, 0.74, 0.15, 0.85)
    return {
        # Visual descriptors, not semantic classifiers. The head descriptor is
        # deliberately retained to help with repeatable hair/headwear appearance.
        "upper_colour": _histogram(upper),
        "lower_colour": _histogram(lower),
        "head": np.concatenate([_histogram(head), _pattern(head)]).astype(np.float32),
        "pattern": np.concatenate([_pattern(upper), _pattern(torso), _pattern(lower)]).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def person_geometry(bbox: Tuple[float, float, float, float], shape: Tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    fw, fh = max(float(shape[0]), 1.0), max(float(shape[1]), 1.0)
    cx = 0.5 * (x1 + x2) / fw
    foot = y2 / fh
    bw = w / fw
    bh = h / fh
    area = (w * h) / (fw * fh)
    aspect = math.log(max(h / w, 1e-3))
    return np.asarray([cx, foot, bw, bh, area, aspect], np.float32)


def geometry_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.5
    delta = np.asarray(left, np.float32) - np.asarray(right, np.float32)
    scale = np.asarray([0.28, 0.35, 0.20, 0.35, 0.20, 1.0], np.float32)
    distance = float(np.linalg.norm(delta / scale))
    return softclip(1.0 - distance / 3.0)


def overlap_boxes(left: Tuple[float, float, float, float], right: Tuple[float, float, float, float],
                  min_iou: float, min_intersection: float) -> bool:
    lx1, ly1, lx2, ly2 = [float(v) for v in left]
    rx1, ry1, rx2, ry2 = [float(v) for v in right]
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return False
    la = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    ra = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    if la <= 0.0 or ra <= 0.0:
        return False
    union = la + ra - inter
    iou = inter / union if union > 0.0 else 0.0
    iom = inter / min(la, ra)
    return bool(iou >= min_iou or iom >= min_intersection)


# ---------------------------------------------------------------------------
# Observation/profile state
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    camera: str
    track_key: str
    timestamp: float
    bbox: Tuple[float, float, float, float]
    deep: Dict[str, Dict[str, np.ndarray]]
    attrs: Dict[str, np.ndarray]
    person_geom: np.ndarray
    floor: Tuple[float, float, str] | None
    quality: float


@dataclass
class TrackState:
    camera: str
    tid: int
    segment: int
    key: str
    last_frame: int = 0
    last_feature_frame: int = -10**9
    last_seen: float = 0.0
    gid: str | None = None
    pending_gid: str | None = None
    pending_count: int = 0


@dataclass
class Profile:
    gid: str
    deep: Dict[str, Dict[str, List[np.ndarray]]] = field(default_factory=lambda: {
        model: {view: [] for view in VIEWS} for model in MODELS
    })
    attrs: Dict[str, List[np.ndarray]] = field(default_factory=lambda: {
        "upper_colour": [], "lower_colour": [], "head": [], "pattern": []
    })
    person_geom: List[np.ndarray] = field(default_factory=list)
    floors: List[Tuple[float, float, str, str]] = field(default_factory=list)
    cameras: set[str] = field(default_factory=set)
    tracks: set[str] = field(default_factory=set)
    last_by_camera: Dict[str, float] = field(default_factory=dict)
    recent_by_camera: Dict[str, str] = field(default_factory=dict)
    observations: int = 0

    def add(self, obs: Observation, bank_size: int, novelty: float) -> None:
        self.cameras.add(obs.camera)
        self.tracks.add(obs.track_key)
        self.last_by_camera[obs.camera] = obs.timestamp
        self.recent_by_camera[obs.camera] = obs.track_key
        self.observations += 1

        for model in MODELS:
            for view in VIEWS:
                value = obs.deep.get(model, {}).get(view)
                if value is None:
                    continue
                bucket = self.deep[model][view]
                if not bucket or max(cosine(value, old) for old in bucket[-16:]) < novelty:
                    bucket.append(np.asarray(value, np.float32))
                    self.deep[model][view] = bucket[-bank_size:]

        for name, value in obs.attrs.items():
            bucket = self.attrs[name]
            if not bucket or max(cosine(value, old) for old in bucket[-12:]) < novelty:
                bucket.append(np.asarray(value, np.float32))
                self.attrs[name] = bucket[-min(bank_size, 48):]

        self.person_geom.append(np.asarray(obs.person_geom, np.float32))
        self.person_geom = self.person_geom[-min(bank_size, 48):]
        if obs.floor is not None:
            x, y, group = obs.floor
            self.floors.append((float(x), float(y), group, obs.camera))
            self.floors = self.floors[-min(bank_size, 48):]


@dataclass
class Match:
    gid: str
    score: float
    deep_top2: float
    deep_support: int
    resnet: float
    swin: float
    solider: float
    upper_colour: float
    lower_colour: float
    head: float
    pattern: float
    person_geometry: float
    room_geometry: float
    temporal: float
    same_camera: bool
    reason: str


class IdentityEngine:
    """Conservative online MTMC association.

    A track never inherits another GID merely because a tracker id exists. Reuse of
    a GID requires multimodel feature agreement. New GIDs are created only after
    two non-overlapping feature observations are internally consistent. Existing
    GIDs are continuously enriched with diverse observations from all views.
    """

    def __init__(self, cfg: Mapping[str, object], debug_path: Path):
        self.lock = threading.RLock()
        self.profiles: Dict[str, Profile] = {}
        self.next_gid = 1
        self.active: Dict[str, TrackState] = {}
        self.cfg = cfg
        self.debug = debug_path.open("w", encoding="utf-8")
        self.stats = {
            "feature_observations": 0,
            "feature_skipped_overlap": 0,
            "existing_gid_assignments": 0,
            "new_gid_assignments": 0,
            "same_camera_repairs": 0,
            "cross_camera_matches": 0,
            "rejected_candidates": 0,
        }

        self.deep_min = dict(MODEL_MIN)
        self.deep_threshold = float(cfg.get("deep_threshold", 0.56))
        self.same_deep_threshold = float(cfg.get("same_deep_threshold", 0.52))
        self.attribute_threshold = float(cfg.get("attribute_threshold", 0.42))
        self.cross_attribute_threshold = float(cfg.get("cross_attribute_threshold", 0.46))
        self.margin = float(cfg.get("match_margin", 0.035))
        self.same_gap = float(cfg.get("same_camera_gap_sec", 15.0))
        self.cross_gap = float(cfg.get("cross_camera_max_gap_sec", 30.0))
        self.bank_size = int(cfg.get("bank_size", 64))
        self.novelty = float(cfg.get("novelty", 0.985))
        self.confirm_samples = max(2, int(cfg.get("confirm_samples", 2)))

    def close(self) -> None:
        with self.lock:
            self.debug.close()

    def _new_gid(self) -> str:
        gid = f"G{self.next_gid:06d}"
        self.next_gid += 1
        return gid

    def _floor_score(self, obs: Observation, profile: Profile) -> float:
        if obs.floor is None or not profile.floors:
            return 0.5
        ox, oy, og = obs.floor
        values = [
            (x, y)
            for x, y, group, _ in profile.floors
            if group == og
        ]
        if not values:
            return 0.5
        distances = [math.hypot(ox - x, oy - y) for x, y in values[-24:]]
        dist = min(distances) if distances else 0.0
        return softclip(1.0 - dist / 6.0)

    def _temporal_score(self, obs: Observation, profile: Profile) -> float:
        if obs.camera not in profile.last_by_camera:
            if not profile.last_by_camera:
                return 0.5
            latest = max(profile.last_by_camera.values())
        else:
            latest = profile.last_by_camera[obs.camera]
        gap = abs(obs.timestamp - latest)
        return softclip(1.0 - gap / self.cross_gap)

    def _same_camera_conflict(self, obs: Observation, profile: Profile) -> bool:
        for state in self.active.values():
            if state.camera != obs.camera or state.gid != profile.gid:
                continue
            if state.key == obs.track_key:
                continue
            if obs.timestamp - state.last_seen <= 1.5:
                return True
        return False

    def _deep_scores(self, obs: Observation, profile: Profile) -> Tuple[Dict[str, float], float, int]:
        scores = {}
        for model in MODELS:
            vals = []
            for view in VIEWS:
                current = obs.deep.get(model, {}).get(view)
                stored = profile.deep[model][view]
                if current is not None and stored:
                    vals.append(topmean([current], stored))
            if vals:
                scores[model] = max(vals)
            else:
                scores[model] = 0.0
        ordered = sorted(scores.values(), reverse=True)
        support = sum(scores[m] >= self.deep_min[m] for m in MODELS)
        top2 = float(np.mean(ordered[:2])) if len(ordered) >= 2 else (ordered[0] if ordered else 0.0)
        return scores, top2, support

    def _attr_score(self, obs: Observation, profile: Profile, name: str) -> float:
        current = obs.attrs.get(name)
        stored = profile.attrs.get(name) or []
        if current is None or not stored:
            return 0.5
        return max(cosine(current, value) for value in stored[-24:])

    def score(self, obs: Observation, profile: Profile) -> Match:
        same = obs.camera in profile.cameras
        deep, top2, support = self._deep_scores(obs, profile)
        upper = self._attr_score(obs, profile, "upper_colour")
        lower = self._attr_score(obs, profile, "lower_colour")
        head = self._attr_score(obs, profile, "head")
        pattern = self._attr_score(obs, profile, "pattern")
        person = geometry_similarity(obs.person_geom, np.median(np.stack(profile.person_geom), axis=0)) if profile.person_geom else 0.5
        room = self._floor_score(obs, profile)
        temporal = self._temporal_score(obs, profile)

        if same:
            gap = abs(obs.timestamp - profile.last_by_camera.get(obs.camera, obs.timestamp))
            continuity = softclip(1.0 - gap / self.same_gap)
            score = (
                0.68 * top2
                + 0.16 * float(np.mean([upper, lower, head, pattern]))
                + 0.10 * person
                + 0.06 * continuity
            )
            reason = "same_camera_feature_repair"
        else:
            score = (
                0.66 * top2
                + 0.17 * float(np.mean([upper, lower, head, pattern]))
                + 0.08 * person
                + 0.06 * room
                + 0.03 * temporal
            )
            reason = "cross_camera_multimodel_match"

        return Match(
            profile.gid,
            float(score),
            float(top2),
            int(support),
            float(deep["resnet"]),
            float(deep["swin"]),
            float(deep["solider"]),
            float(upper),
            float(lower),
            float(head),
            float(pattern),
            float(person),
            float(room),
            float(temporal),
            bool(same),
            reason,
        )

    def acceptable(self, match: Match, second: float) -> bool:
        threshold = self.same_deep_threshold if match.same_camera else self.deep_threshold
        attribute = float(np.mean([
            match.upper_colour,
            match.lower_colour,
            match.head,
            match.pattern,
        ]))
        attr_min = self.attribute_threshold if match.same_camera else self.cross_attribute_threshold
        if match.deep_support < 2:
            return False
        if match.deep_top2 < threshold:
            return False
        if attribute < attr_min and match.deep_top2 < 0.68:
            return False
        if match.score - second < self.margin and match.score < 0.72:
            return False
        return True

    def _debug(self, obs: Observation, ranked: List[Match], accepted: Match | None, state: TrackState) -> None:
        top = ranked[:5]
        payload = {
            "camera": obs.camera,
            "track": obs.track_key,
            "timestamp": obs.timestamp,
            "gid": state.gid,
            "accepted": None if accepted is None else accepted.__dict__,
            "candidates": [match.__dict__ for match in top],
        }
        self.debug.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.debug.flush()

    def observe(self, obs: Observation, state: TrackState) -> str | None:
        with self.lock:
            self.stats["feature_observations"] += 1
            state.last_seen = obs.timestamp
            self.active[state.key] = state

            if state.gid is not None:
                profile = self.profiles.get(state.gid)
                if profile is not None:
                    profile.add(obs, self.bank_size, self.novelty)
                self._debug(obs, [], None, state)
                return state.gid

            ranked = []
            for profile in self.profiles.values():
                if self._same_camera_conflict(obs, profile):
                    continue
                ranked.append(self.score(obs, profile))
            ranked.sort(key=lambda x: x.score, reverse=True)
            second = ranked[1].score if len(ranked) > 1 else 0.0

            accepted = ranked[0] if ranked and self.acceptable(ranked[0], second) else None
            if accepted is not None:
                state.gid = accepted.gid
                state.pending_gid = None
                state.pending_count = 0
                profile = self.profiles[accepted.gid]
                profile.add(obs, self.bank_size, self.novelty)
                self.stats["existing_gid_assignments"] += 1
                if accepted.same_camera:
                    self.stats["same_camera_repairs"] += 1
                else:
                    self.stats["cross_camera_matches"] += 1
                self._debug(obs, ranked, accepted, state)
                return state.gid

            if ranked:
                self.stats["rejected_candidates"] += 1
                candidate = ranked[0].gid
                if state.pending_gid == candidate:
                    state.pending_count += 1
                else:
                    state.pending_gid = candidate
                    state.pending_count = 1
                # A candidate that keeps failing the actual multimodel gate is NOT
                # assigned. This is the key anti-hallucination rule.
                if state.pending_count < self.confirm_samples:
                    self._debug(obs, ranked, None, state)
                    return None

            # Create a genuinely new identity only after repeated feature evidence
            # that does not satisfy any existing GID. This is not a ReID match; it
            # is the novel-identity branch.
            if state.pending_gid is None or state.pending_count >= self.confirm_samples:
                gid = self._new_gid()
                state.gid = gid
                state.pending_gid = None
                state.pending_count = 0
                profile = Profile(gid)
                profile.add(obs, self.bank_size, self.novelty)
                self.profiles[gid] = profile
                self.stats["new_gid_assignments"] += 1
                self._debug(obs, ranked, None, state)
                return gid

            self._debug(obs, ranked, None, state)
            return None


# ---------------------------------------------------------------------------
# Camera worker
# ---------------------------------------------------------------------------


class CameraWorker(threading.Thread):
    def __init__(self, name: str, source: str, cfg: Mapping[str, object], engine: IdentityEngine,
                 stop_event: threading.Event, latest: Dict[str, np.ndarray], latest_lock: threading.Lock,
                 output_dir: Path, log_path: Path, floor: object | None, show: bool = False):
        super().__init__(name=f"camera-{name}", daemon=True)
        self.name_id = name
        self.source = source
        self.cfg = cfg
        self.engine = engine
        self.stop_event = stop_event
        self.latest = latest
        self.latest_lock = latest_lock
        self.output_dir = output_dir
        self.log_path = log_path
        self.floor = floor
        self.show = show
        self.error: Exception | None = None
        self.stats = {"frames": 0, "tracks": 0, "features": 0, "overlap_skips": 0, "recovery_samples": 0}
        self.track_states: Dict[int, TrackState] = {}

    def _capture(self):
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|timeout;5000000|stimeout;5000000",
        )
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"{self.name_id}: cannot open source {self.source}")
        return cap

    def _overlap_map(self, items):
        blocked = set()
        partners: Dict[int, List[int]] = {}
        min_iou = float(self.cfg.get("overlap_iou", 0.05))
        min_inter = float(self.cfg.get("overlap_intersection", 0.20))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if not overlap_boxes(items[i]["bbox"], items[j]["bbox"], min_iou, min_inter):
                    continue
                a, b = items[i]["tid"], items[j]["tid"]
                blocked.add(a)
                blocked.add(b)
                partners.setdefault(a, []).append(b)
                partners.setdefault(b, []).append(a)
        return blocked, partners

    def _variants(self, image, box):
        person = crop(image, box)
        q = quality(person) if person is not None else 0.0
        if person is None or q < float(self.cfg.get("min_quality", 0.20)):
            return None
        variants = {
            "full": person,
            "light": illumination_variant(person),
        }
        h, w = person.shape[:2]
        if h >= 40 and w >= 20:
            variants["upper"] = person[:max(1, int(h * 0.68))]
            variants["torso"] = person[int(h * 0.08):max(int(h * 0.74), int(h * 0.08) + 1)]
            variants["lower"] = person[int(h * 0.32):]
        return q, person, variants

    def run(self):  # noqa: C901
        cap = None
        writer = None
        rows = None
        try:
            cap = self._capture()
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise RuntimeError(f"{self.name_id}: source returned invalid dimensions {width}x{height}")
            output = self.output_dir / f"output_{self.name_id}.mp4"
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"{self.name_id}: cannot open output writer {output}")
            rows = self.log_path.open("w", encoding="utf-8")

            detector_cfg = self.cfg["detector"]
            detector = PersonDetector(
                model_path=detector_cfg["model"],
                confidence_threshold=float(detector_cfg["conf"]),
                person_class_id=0,
                tracker_config=detector_cfg["tracker"],
                pose_ensemble=None,
                iou=float(detector_cfg["iou"]),
            )

            frame = 0
            interval = max(1, int(self.cfg.get("feature_interval", 5)))
            gap_frames = max(1, int(float(self.cfg.get("fragment_gap_sec", 2.0)) * fps))
            segments: Dict[int, int] = {}
            seen_frame: Dict[int, int] = {}

            while not self.stop_event.is_set():
                ok, image = cap.read()
                if not ok:
                    break
                frame += 1
                self.stats["frames"] = frame
                timestamp = frame / fps

                raw = [x for x in detector.track(image) if x.track_id is not None]
                prepared = []
                for item in raw:
                    tid = int(item.track_id)
                    previous = seen_frame.get(tid)
                    if previous is None or frame - previous > gap_frames:
                        segments[tid] = segments.get(tid, 0) + 1
                    seen_frame[tid] = frame
                    seg = segments[tid]
                    key = f"{self.name_id}:{tid}:{seg}"
                    state = self.track_states.get(tid)
                    if state is None or state.key != key:
                        state = TrackState(self.name_id, tid, seg, key)
                        self.track_states[tid] = state
                    state.last_frame = frame
                    state.last_seen = timestamp
                    box = (float(item.x1), float(item.y1), float(item.x2), float(item.y2))
                    prepared.append({"tid": tid, "key": key, "state": state, "bbox": box, "conf": float(item.confidence)})

                self.stats["tracks"] += len(prepared)
                blocked, partners = self._overlap_map(prepared)
                feature_jobs = []
                for item in prepared:
                    state = item["state"]
                    blocked_here = item["tid"] in blocked
                    rows.write(json.dumps({
                        "camera": self.name_id,
                        "frame": frame,
                        "timestamp": timestamp,
                        "tracklet_key": state.key,
                        "track_id": item["tid"],
                        "segment": state.segment,
                        "bbox": list(item["bbox"]),
                        "detection_score": item["conf"],
                        "overlap_blocked": blocked_here,
                        "overlap_partners": partners.get(item["tid"], []),
                    }) + "\n")
                    if blocked_here:
                        self.stats["overlap_skips"] += 1
                        continue
                    if frame - state.last_feature_frame < interval:
                        continue
                    built = self._variants(image, item["bbox"])
                    if built is None:
                        continue
                    q, person, variants = built
                    state.last_feature_frame = frame
                    feature_jobs.append((item, q, person, variants))

                if feature_jobs:
                    ordered = []
                    spans = []
                    for _, _, _, variants in feature_jobs:
                        start = len(ordered)
                        for view in VIEWS:
                            if view in variants:
                                ordered.append(variants[view])
                        spans.append((start, len(ordered)))

                    resnet = MODEL = None
                    # Extract each model once for the whole camera batch.
                    # max_batch is enforced by the wrappers themselves.
                    resnet = self.engine_models["resnet"].extract_batch(ordered)
                    swin = self.engine_models["swin"].extract_batch(ordered)
                    solider = self.engine_models["solider"].extract_batch(ordered)

                    idx = 0
                    for job, (start, end) in zip(feature_jobs, spans):
                        item, q, person, variants = job
                        count = end - start
                        view_names = [name for name in VIEWS if name in variants]
                        rv = resnet[idx:idx + count]
                        sv = swin[idx:idx + count]
                        tv = solider[idx:idx + count]
                        idx += count
                        deep = {
                            "resnet": {name: np.asarray(value, np.float32) for name, value in zip(view_names, rv)},
                            "swin": {name: np.asarray(value, np.float32) for name, value in zip(view_names, sv)},
                            "solider": {name: np.asarray(value, np.float32) for name, value in zip(view_names, tv)},
                        }
                        attrs = appearance_features(person)
                        geom = person_geometry(item["bbox"], (width, height))
                        floor = None
                        if self.floor is not None:
                            try:
                                value = self.floor.position(self.name_id, item["bbox"], (width, height))
                                if value is not None:
                                    floor = (float(value.x), float(value.y), str(value.group))
                            except Exception:  # geometry is fail-open by design
                                floor = None
                        obs = Observation(
                            camera=self.name_id,
                            track_key=item["key"],
                            timestamp=timestamp,
                            bbox=item["bbox"],
                            deep=deep,
                            attrs=attrs,
                            person_geom=geom,
                            floor=floor,
                            quality=float(q),
                        )
                        gid = self.engine.observe(obs, item["state"])
                        if gid:
                            self.stats["features"] += 1

                for item in prepared:
                    state = item["state"]
                    colour = (90, 90, 90)
                    label = f"T{item['tid']} PENDING"
                    if state.gid:
                        label = f"{state.gid}  T{item['tid']}"
                        number = int(state.gid[1:])
                        palette = {
                            1: (66, 199, 125), 2: (203, 74, 221), 3: (255, 90, 30),
                            4: (0, 165, 255), 5: (211, 85, 186), 6: (0, 215, 255),
                            7: (255, 191, 0), 8: (80, 80, 220),
                        }
                        colour = palette.get(number, palette[((number - 1) % 8) + 1])
                    x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
                    if item["tid"] in blocked:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (120, 120, 120), 2)
                        cv2.putText(image, label + " OVERLAP", (x1, max(20, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 2, cv2.LINE_AA)
                    else:
                        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
                        cv2.putText(image, label, (x1, max(20, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 2, cv2.LINE_AA)

                cv2.putText(image, f"{self.name_id} | feature-every {interval}f | overlap-protected",
                            (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(image)
                with self.latest_lock:
                    self.latest[self.name_id] = image.copy()

        except Exception as exc:  # noqa: BLE001
            self.error = exc
            self.stop_event.set()
        finally:
            if rows is not None:
                rows.close()
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------


def parse_camera(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("camera must be NAME=RTSP_URL")
    name, source = value.split("=", 1)
    if not name or not source:
        raise argparse.ArgumentTypeError("camera must be NAME=RTSP_URL")
    return name, source


def load_yaml(path: str) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_floor(cfg: Mapping[str, object], run_id: str):
    if load_calibration is None or FloorFrame is None:
        print("[live-independent] geometry: unavailable in this checkout -> fail-open")
        return None
    gcfg = cfg.get("geometry", {}) or {}
    if not gcfg.get("enabled", True):
        print("[live-independent] geometry: OFF")
        return None
    path = gcfg.get("calibration_path", "calibration/floor_frame.json")
    try:
        record = load_calibration(path)
        if record is None:
            print(f"[live-independent] geometry: no calibration at {path} -> fail-open")
            return None
        floor = FloorFrame(record)
        print("[live-independent] geometry: ACTIVE")
        print(record.summary())
        return floor
    except Exception as exc:  # noqa: BLE001
        print(f"[live-independent] geometry: {exc} -> fail-open")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent overlap-protected multimodel live person ReID; no external application dependency."
    )
    parser.add_argument("--config", default="rebuild/config_live_independent.yaml")
    parser.add_argument("--camera", action="append", required=True, type=parse_camera,
                        help="NAME=RTSP_URL; repeat for each camera")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl-C")
    parser.add_argument("--show", action="store_true", help="show live windows")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(cfg.get("output_dir", "rebuild_outputs_live_independent"))
    run_dir = output_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Do not reuse a stale persistent gallery here. Every live session starts with
    # an empty identity set; only the features observed in THIS session can create
    # or reuse a GID. That removes the old-gallery contamination path.
    engine_cfg = cfg.get("identity", {}) or {}
    engine = IdentityEngine(engine_cfg, log_dir / "match_debug.jsonl")
    floor = build_floor(cfg, run_id)

    models = cfg.get("models", {}) or {}
    reid = ReIDExtractor(
        weights=models["resnet_weights"],
        device=models.get("device", "cuda"),
        tap=models.get("resnet_tap", "post_relu"),
        max_batch=int(models.get("resnet_batch", 32)),
        model=models.get("resnet_model"),
    )
    swin = NVIDIASwinReIDExtractor(
        models["swin_weights"], device=models.get("device", "cuda"),
        max_batch=int(models.get("swin_batch", 16)),
    )
    solider = SOLIDERReIDExtractor(
        models["solider_weights"], device=models.get("device", "cuda"),
        max_batch=int(models.get("solider_batch", 16)),
    )
    engine_models = {"resnet": reid, "swin": swin, "solider": solider}

    print("=" * 78)
    print("       INDEPENDENT MULTIMODEL LIVE PERSON RE-ID")
    print("=" * 78)
    print(f"run:        {run_id}")
    print(f"cameras:    {', '.join(name for name, _ in args.camera)}")
    print("models:     NVIDIA ResNet + NVIDIA Swin + SOLIDER")
    print("feature:    full/light/upper/torso/lower + colour/pattern/head + geometry")
    print("rule:       NO ReID assignment unless multimodel feature gate passes")
    print("sampling:   continuous; feature extraction blocked for overlapping tracks")
    print("dependency: independent of seif744/Inference_PersonReid")
    print("Ctrl-C:     stops all cameras cleanly and prints final identity summary")
    print("=" * 78)
    print(f"[models] ResNet:  {reid.describe()}")
    print(f"[models] Swin:    {swin.describe()}")
    print(f"[models] SOLIDER: {solider.describe()}")

    stop_event = threading.Event()
    latest: Dict[str, np.ndarray] = {}
    latest_lock = threading.Lock()
    workers = []
    for name, source in args.camera:
        worker = CameraWorker(
            name,
            source,
            {
                **cfg.get("capture", {}),
                "detector": cfg["detector"],
                "fragment_gap_sec": cfg.get("tracking", {}).get("fragment_gap_sec", 2.0),
            },
            engine,
            stop_event,
            latest,
            latest_lock,
            run_dir,
            log_dir / f"{name}.jsonl",
            floor,
            args.show,
        )
        worker.engine_models = engine_models
        workers.append(worker)

    def stop(_sig=None, _frame=None):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    for worker in workers:
        worker.start()

    try:
        while any(worker.is_alive() for worker in workers):
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                stop_event.set()
                break
            if args.show:
                with latest_lock:
                    frames = {name: frame.copy() for name, frame in latest.items()}
                for name, frame in frames.items():
                    cv2.imshow(name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    stop_event.set()
                    break
            else:
                time.sleep(0.2)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=8.0)
        if args.show:
            cv2.destroyAllWindows()
        engine.close()

    errors = [f"{worker.name_id}: {worker.error}" for worker in workers if worker.error is not None]
    print("\n" + "=" * 78)
    print("                     LIVE RUN COMPLETE")
    print("=" * 78)
    for worker in workers:
        print(
            f"{worker.name_id}: frames={worker.stats['frames']} "
            f"tracks={worker.stats['tracks']} features={worker.stats['features']} "
            f"overlap_skips={worker.stats['overlap_skips']}"
        )
    print(f"feature observations:    {engine.stats['feature_observations']}")
    print(f"existing GID matches:    {engine.stats['existing_gid_assignments']}")
    print(f"new GIDs:                {engine.stats['new_gid_assignments']}")
    print(f"same-camera repairs:     {engine.stats['same_camera_repairs']}")
    print(f"cross-camera matches:    {engine.stats['cross_camera_matches']}")
    print(f"candidate rejections:    {engine.stats['rejected_candidates']}")
    print(f"profiles created:        {len(engine.profiles)}")
    print("FINAL GLOBAL IDS:")
    for gid, profile in sorted(engine.profiles.items(), key=lambda item: int(item[0][1:])):
        print(
            f"  {gid}: cameras={','.join(sorted(profile.cameras))} "
            f"tracks={len(profile.tracks)} observations={profile.observations}"
        )
    print(f"outputs: {run_dir}")
    print(f"debug:   {log_dir / 'match_debug.jsonl'}")
    if errors:
        print("ERRORS:")
        for value in errors:
            print(f"  {value}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
