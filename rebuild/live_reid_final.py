from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import signal
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from detector import PersonDetector
from rebuild.identity_v2 import crop, illumination_variant, quality
from reid.extractor import ReIDExtractor
from reid.nvidia_swin import NVIDIASwinReIDExtractor
from reid.solider_reid import SOLIDERReIDExtractor

try:
    from geometry.calibration import load_calibration
    from geometry.floor import FloorFrame
except Exception:  # geometry is optional and fail-open
    load_calibration = None
    FloorFrame = None


MODELS = ("resnet", "swin", "solider")
VIEWS = ("full", "light", "upper", "torso", "lower")
MODEL_MIN = {"resnet": 0.44, "swin": 0.44, "solider": 0.42}
TRANSITIONS = {
    ("full", "upper"), ("upper", "full"),
    ("full", "torso"), ("torso", "full"),
    ("upper", "torso"), ("torso", "upper"),
    ("full", "lower"), ("lower", "full"),
    ("torso", "lower"), ("lower", "torso"),
}


# ---------------------------------------------------------------------------
# Numeric and appearance features
# ---------------------------------------------------------------------------


def unit(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if arr.size == 0 or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid feature vector")
    return arr / norm


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(unit(left), unit(right)), -1.0, 1.0))


def topmean(values: Iterable[float], count: int = 3) -> float:
    data = sorted((float(x) for x in values), reverse=True)
    return float(np.mean(data[: min(count, len(data))])) if data else 0.0


def zone(image: np.ndarray, y1: float, y2: float,
         x1: float = 0.08, x2: float = 0.92) -> np.ndarray:
    h, w = image.shape[:2]
    xa = max(0, min(int(w * x1), max(w - 1, 0)))
    xb = max(xa + 1, min(int(w * x2), w))
    ya = max(0, min(int(h * y1), max(h - 1, 0)))
    yb = max(ya + 1, min(int(h * y2), h))
    return image[ya:yb, xa:xb]


def colour(image: np.ndarray, bins: int = 18) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros(bins + 10, np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.float32) / 180.0
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    hh, _ = np.histogram(hue, bins=bins, range=(0.0, 1.0), weights=sat + 0.05)
    vv, _ = np.histogram(val, bins=10, range=(0.0, 1.0), weights=1.0 - 0.35 * sat)
    return unit(np.concatenate([hh, vv]).astype(np.float32))


def texture(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros(20, np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=False) % math.pi
    orient, _ = np.histogram(ang, bins=12, range=(0.0, math.pi), weights=mag + 1e-4)
    edge = cv2.Canny((gray * 255.0).astype(np.uint8), 60, 140)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    residual = np.abs(gray - blur)
    extra = np.asarray([
        float(np.mean(edge > 0)),
        float(np.var(gray)),
        float(np.mean(residual)),
        float(np.mean(mag)),
        float(np.std(gray)),
        float(np.mean(gray)),
        float(np.median(gray)),
        float(np.percentile(gray, 90)),
    ], np.float32)
    return unit(np.concatenate([orient.astype(np.float32), extra]))


def appearance(image: np.ndarray) -> Dict[str, np.ndarray]:
    head = zone(image, 0.00, 0.28, 0.16, 0.84)
    shirt = zone(image, 0.20, 0.60, 0.10, 0.90)
    torso = zone(image, 0.18, 0.76, 0.14, 0.86)
    bottom = zone(image, 0.55, 0.94, 0.10, 0.90)
    return {
        "shirt_colour": colour(shirt),
        "bottom_colour": colour(bottom),
        # This is an appearance descriptor, not a false semantic claim that a
        # dedicated cap/hair classifier exists.
        "head_appearance": unit(np.concatenate([colour(head), texture(head)])),
        "cloth_pattern": unit(np.concatenate([texture(shirt), texture(torso), texture(bottom)])),
    }


def person_geometry(box: Tuple[float, float, float, float], frame_size: Tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = [float(x) for x in box]
    fw, fh = max(float(frame_size[0]), 1.0), max(float(frame_size[1]), 1.0)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return np.asarray([
        0.5 * (x1 + x2) / fw,
        y2 / fh,
        bw / fw,
        bh / fh,
        bw * bh / (fw * fh),
        math.log(bh / bw),
    ], np.float32)


def geometry_score(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.asarray([0.25, 0.30, 0.18, 0.30, 0.18, 1.0], np.float32)
    dist = float(np.linalg.norm((left - right) / scale))
    return float(np.clip(1.0 - dist / 3.0, 0.0, 1.0))


def overlap(left, right, min_iou: float, min_intersection: float) -> bool:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
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
# State
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    camera: str
    track_key: str
    timestamp: float
    bbox: Tuple[float, float, float, float]
    deep: Dict[str, Dict[str, np.ndarray]]
    attributes: Dict[str, np.ndarray]
    geometry: np.ndarray
    floor: Optional[Tuple[float, float, str]]
    quality: float


@dataclass
class TrackState:
    camera: str
    tid: int
    segment: int
    key: str
    gid: Optional[str] = None
    last_frame: int = 0
    last_feature_frame: int = -10**9
    last_seen: float = 0.0
    overlap: bool = False
    recovery_left: int = 0
    verify_failures: int = 0
    switch_gid: Optional[str] = None
    switch_count: int = 0
    pending_new: List[Observation] = field(default_factory=list)


@dataclass
class Match:
    gid: str
    score: float
    deep_top2: float
    deep_support: int
    models: Dict[str, float]
    attributes: Dict[str, float]
    person_geometry: float
    room_geometry: float
    temporal: float
    same_camera: bool
    view_support: int
    reason: str


@dataclass
class Profile:
    gid: str
    deep: Dict[str, Dict[str, List[np.ndarray]]] = field(default_factory=lambda: {
        model: {view: [] for view in VIEWS} for model in MODELS
    })
    attributes: Dict[str, List[np.ndarray]] = field(default_factory=lambda: {
        "shirt_colour": [], "bottom_colour": [], "head_appearance": [], "cloth_pattern": []
    })
    geometry: List[np.ndarray] = field(default_factory=list)
    floor: List[Tuple[float, float, str, str]] = field(default_factory=list)
    cameras: set[str] = field(default_factory=set)
    tracks: set[str] = field(default_factory=set)
    last_by_camera: Dict[str, float] = field(default_factory=dict)
    observations: int = 0

    def add(self, obs: Observation, bank_size: int, novelty: float) -> None:
        self.cameras.add(obs.camera)
        self.tracks.add(obs.track_key)
        self.last_by_camera[obs.camera] = obs.timestamp
        self.observations += 1

        for model in MODELS:
            for view in VIEWS:
                value = obs.deep.get(model, {}).get(view)
                if value is None:
                    continue
                bank = self.deep[model][view]
                nearest = max((cosine(value, old) for old in bank[-16:]), default=-1.0)
                if nearest < novelty:
                    bank.append(np.asarray(value, np.float32))
                    self.deep[model][view] = bank[-bank_size:]

        for name, value in obs.attributes.items():
            bank = self.attributes[name]
            nearest = max((cosine(value, old) for old in bank[-12:]), default=-1.0)
            if nearest < novelty:
                bank.append(np.asarray(value, np.float32))
                self.attributes[name] = bank[-min(bank_size, 48):]

        self.geometry.append(np.asarray(obs.geometry, np.float32))
        self.geometry = self.geometry[-min(bank_size, 48):]
        if obs.floor is not None:
            self.floor.append((*obs.floor, obs.camera))
            self.floor = self.floor[-min(bank_size, 48):]


class Gallery:
    """Small local persistent profile store, independent of any external repo."""

    VERSION = 1

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS profiles (gid TEXT PRIMARY KEY, blob BLOB NOT NULL)")
        self.db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('version',?)", (str(self.VERSION),))
        self.db.commit()

    def load(self) -> Dict[str, Profile]:
        with self.lock:
            rows = self.db.execute("SELECT gid, blob FROM profiles").fetchall()
        result = {}
        for gid, blob in rows:
            profile = pickle.loads(blob)
            if isinstance(profile, Profile) and profile.gid == gid:
                result[gid] = profile
        return result

    def save(self, profile: Profile) -> None:
        blob = sqlite3.Binary(pickle.dumps(profile, protocol=pickle.HIGHEST_PROTOCOL))
        with self.lock:
            self.db.execute(
                "INSERT INTO profiles(gid,blob) VALUES(?,?) "
                "ON CONFLICT(gid) DO UPDATE SET blob=excluded.blob",
                (profile.gid, blob),
            )
            self.db.commit()

    def close(self) -> None:
        with self.lock:
            self.db.commit()
            self.db.close()


# ---------------------------------------------------------------------------
# Identity engine
# ---------------------------------------------------------------------------


class IdentityEngine:
    def __init__(self, cfg: Mapping[str, object], gallery: Optional[Gallery], debug_path: Path):
        self.cfg = cfg
        self.gallery = gallery
        self.profiles = {} if gallery is None else gallery.load()
        self.active: Dict[str, TrackState] = {}
        self.lock = threading.RLock()
        self.debug = debug_path.open("w", encoding="utf-8")
        self.next_gid = self._next_gid()
        self.deep_threshold = float(cfg.get("cross_deep_threshold", 0.56))
        self.same_threshold = float(cfg.get("same_deep_threshold", 0.52))
        self.model_min = {**MODEL_MIN, **(cfg.get("model_min", {}) or {})}
        self.required_models = int(cfg.get("required_models", 2))
        self.required_views = int(cfg.get("required_views", 2))
        self.strong_deep = float(cfg.get("strong_deep", 0.76))
        self.attr_threshold = float(cfg.get("attribute_mean", 0.46))
        self.strong_attr_relax = float(cfg.get("strong_attribute_relax", 0.68))
        self.margin = float(cfg.get("match_margin", 0.035))
        self.switch_margin = float(cfg.get("switch_margin", 0.05))
        self.confirm = max(2, int(cfg.get("confirm_samples", 2)))
        self.verify_grace = max(1, int(cfg.get("verify_grace_samples", 3)))
        self.cross_gap = float(cfg.get("cross_gap_sec", 30.0))
        self.same_gap = float(cfg.get("same_gap_sec", 15.0))
        self.bank_size = int(cfg.get("bank_size", 64))
        self.novelty = float(cfg.get("novelty", 0.985))
        self.new_model_min = {
            "resnet": float(cfg.get("new_identity_min_resnet", 0.74)),
            "swin": float(cfg.get("new_identity_min_swin", 0.74)),
            "solider": float(cfg.get("new_identity_min_solider", 0.72)),
        }
        self.new_attr = float(cfg.get("new_identity_attribute", 0.42))
        self.save_every = max(1, int(cfg.get("save_every", 50)))
        self.unsaved = 0
        self.stats = defaultdict(int)

    def _next_gid(self) -> int:
        nums = [int(g[1:]) for g in self.profiles if g.startswith("G") and g[1:].isdigit()]
        return max(nums, default=0) + 1

    def _new_gid(self) -> str:
        gid = f"G{self.next_gid:06d}"
        self.next_gid += 1
        return gid

    def _active_conflict(self, obs: Observation, gid: str, state: TrackState) -> bool:
        window = float(self.cfg.get("active_conflict_sec", 2.0))
        for other in self.active.values():
            if other.key == state.key or other.camera != obs.camera or other.gid != gid:
                continue
            if abs(obs.timestamp - other.last_seen) <= window:
                return True
        return False

    @staticmethod
    def _attribute(obs: Observation, profile: Profile, name: str) -> float:
        current = obs.attributes.get(name)
        stored = profile.attributes.get(name) or []
        if current is None or not stored:
            return 0.5
        return max(cosine(current, old) for old in stored[-24:])

    def _deep(self, obs: Observation, profile: Profile) -> Tuple[Dict[str, float], float, int, int]:
        scores: Dict[str, float] = {}
        view_support = 0
        for model in MODELS:
            per_view = []
            for qview, qvec in obs.deep.get(model, {}).items():
                candidates = []
                for pview, bank in profile.deep[model].items():
                    if not bank:
                        continue
                    if qview == pview:
                        weight = 1.0
                    elif (qview, pview) in TRANSITIONS:
                        weight = 0.94
                    else:
                        continue
                    best = max(cosine(qvec, old) for old in bank[-32:])
                    candidates.append((best, weight))
                if not candidates:
                    continue
                candidates.sort(key=lambda x: x[0], reverse=True)
                best = candidates[0][0]
                per_view.append(best)
                if best >= 0.50:
                    view_support += 1
            scores[model] = topmean(per_view, 3)
        ordered = sorted(scores.values(), reverse=True)
        top2 = float(np.mean(ordered[:2])) if len(ordered) >= 2 else (ordered[0] if ordered else 0.0)
        support = sum(scores[m] >= float(self.model_min[m]) for m in MODELS)
        return scores, top2, support, view_support

    def _floor(self, obs: Observation, profile: Profile) -> float:
        if obs.floor is None or not profile.floor:
            return 0.5
        x, y, group = obs.floor
        points = [(a, b) for a, b, g, _ in profile.floor if g == group]
        if not points:
            return 0.5
        d = min(math.hypot(x - a, y - b) for a, b in points[-32:])
        return float(np.clip(1.0 - d / 6.0, 0.0, 1.0))

    @staticmethod
    def _temporal(obs: Observation, profile: Profile, window: float) -> float:
        if not profile.last_by_camera:
            return 0.5
        latest = profile.last_by_camera.get(obs.camera)
        if latest is None:
            latest = max(profile.last_by_camera.values())
        gap = abs(obs.timestamp - latest)
        return float(np.clip(1.0 - gap / max(window, 1.0), 0.0, 1.0))

    def score(self, obs: Observation, profile: Profile) -> Match:
        same = obs.camera in profile.cameras
        models, deep2, support, views = self._deep(obs, profile)
        attributes = {name: self._attribute(obs, profile, name) for name in obs.attributes}
        attr_mean = float(np.mean(list(attributes.values()))) if attributes else 0.5
        pgeom = 0.5
        if profile.geometry:
            pgeom = geometry_score(obs.geometry, np.median(np.stack(profile.geometry[-24:]), axis=0))
        room = self._floor(obs, profile)
        temporal = self._temporal(obs, profile, self.same_gap if same else self.cross_gap)
        if same:
            score = 0.76 * deep2 + 0.10 * attr_mean + 0.08 * pgeom + 0.06 * room
            reason = "same_camera_reid_repair"
        else:
            score = 0.76 * deep2 + 0.14 * attr_mean + 0.05 * room + 0.05 * temporal
            reason = "cross_camera_multimodel_reid"
        return Match(
            profile.gid, float(score), float(deep2), int(support),
            models, attributes, float(pgeom), float(room), float(temporal),
            same, int(views), reason,
        )

    def acceptable(self, match: Match, second: float) -> bool:
        threshold = self.same_threshold if match.same_camera else self.deep_threshold
        attr_floor = self.attr_threshold
        attr_mean = float(np.mean(list(match.attributes.values())))
        if match.deep_support < self.required_models:
            return False
        if match.deep_top2 < threshold:
            return False
        # At least two independent body views unless the deep evidence is very strong.
        if match.view_support < self.required_views and match.deep_top2 < self.strong_deep:
            return False
        if attr_mean < attr_floor and match.deep_top2 < self.strong_deep:
            return False
        if match.score < 0.62:
            return False
        if match.score - second < self.margin and match.deep_top2 < 0.74:
            return False
        return True

    def _new_consistent(self, obs: Observation, history: List[Observation]) -> bool:
        if not history:
            return False
        prior = history[-1]
        for model in MODELS:
            matches = []
            for view, vector in obs.deep[model].items():
                for old_view, old_vector in prior.deep[model].items():
                    if view == old_view or (view, old_view) in TRANSITIONS:
                        matches.append(cosine(vector, old_vector))
            if not matches or max(matches) < self.new_model_min[model]:
                return False
        values = []
        for name, value in obs.attributes.items():
            old = prior.attributes.get(name)
            if old is not None:
                values.append(cosine(value, old))
        return not values or float(np.mean(values)) >= self.new_attr

    def _debug(self, obs: Observation, ranked: List[Match], accepted: Optional[Match], state: TrackState, status: str) -> None:
        item = {
            "camera": obs.camera,
            "tracklet": obs.track_key,
            "timestamp": round(obs.timestamp, 3),
            "quality": round(obs.quality, 4),
            "overlap": state.overlap,
            "status": status,
            "gid": state.gid,
            "accepted": None if accepted is None else asdict(accepted),
            "candidates": [asdict(x) for x in ranked[:8]],
        }
        self.debug.write(json.dumps(item, separators=(",", ":")) + "\n")
        self.debug.flush()

    def _persist(self) -> None:
        if self.gallery is None:
            return
        if self.unsaved >= self.save_every:
            for profile in self.profiles.values():
                self.gallery.save(profile)
            self.unsaved = 0

    def observe(self, obs: Observation, state: TrackState) -> Optional[str]:
        with self.lock:
            self.stats["feature_observations"] += 1
            state.last_seen = obs.timestamp
            self.active[state.key] = state

            current = self.profiles.get(state.gid) if state.gid else None
            own = self.score(obs, current) if current is not None else None
            if own is not None and self._verify_own(own):
                state.verify_failures = 0
                state.switch_gid = None
                state.switch_count = 0
                current.add(obs, self.bank_size, self.novelty)
                self.unsaved += 1
                self._persist()
                self._debug(obs, [own], own, state, "verified")
                return state.gid

            ranked: List[Match] = []
            for profile in self.profiles.values():
                if self._active_conflict(obs, profile.gid, state):
                    continue
                ranked.append(self.score(obs, profile))
            ranked.sort(key=lambda x: x.score, reverse=True)
            second = ranked[1].score if len(ranked) > 1 else 0.0
            candidate = ranked[0] if ranked and self.acceptable(ranked[0], second) else None

            if state.gid is not None and candidate is not None and candidate.gid != state.gid:
                own_score = own.score if own is not None else 0.0
                if candidate.score >= own_score + self.switch_margin:
                    if state.switch_gid == candidate.gid:
                        state.switch_count += 1
                    else:
                        state.switch_gid = candidate.gid
                        state.switch_count = 1
                    if state.switch_count >= self.confirm:
                        state.gid = candidate.gid
                        state.verify_failures = 0
                        state.switch_gid = None
                        state.switch_count = 0
                        self.profiles[candidate.gid].add(obs, self.bank_size, self.novelty)
                        self.unsaved += 1
                        self.stats["gid_switches"] += 1
                        self._debug(obs, ranked, candidate, state, "switched")
                        return state.gid
                state.verify_failures += 1
                self.stats["verification_failures"] += 1
                self._debug(obs, ranked, None, state, "mismatch_hold")
                if state.verify_failures >= self.verify_grace:
                    state.gid = None
                    state.switch_gid = None
                    state.switch_count = 0
                return None

            if state.gid is not None:
                state.verify_failures += 1
                self.stats["verification_failures"] += 1
                self._debug(obs, ranked, None, state, "verification_failed")
                if state.verify_failures >= self.verify_grace:
                    state.gid = None
                return None

            if candidate is not None:
                state.gid = candidate.gid
                self.profiles[state.gid].add(obs, self.bank_size, self.novelty)
                self.unsaved += 1
                self.stats["existing_gid_assignments"] += 1
                self._debug(obs, ranked, candidate, state, "matched")
                return state.gid

            self.stats["rejected_candidates"] += 1
            if state.pending_new and self._new_consistent(obs, state.pending_new):
                gid = self._new_gid()
                profile = Profile(gid)
                profile.add(state.pending_new[-1], self.bank_size, self.novelty)
                profile.add(obs, self.bank_size, self.novelty)
                self.profiles[gid] = profile
                state.gid = gid
                state.pending_new.clear()
                self.unsaved += 1
                self.stats["new_gid_assignments"] += 1
                self._debug(obs, ranked, None, state, "new_confirmed")
                return gid

            state.pending_new.append(obs)
            state.pending_new = state.pending_new[-2:]
            self.stats["pending_new"] += 1
            self._debug(obs, ranked, None, state, "pending_new")
            return None

    def _verify_own(self, match: Match) -> bool:
        attr = float(np.mean(list(match.attributes.values())))
        return (
            match.deep_support >= self.required_models
            and match.deep_top2 >= self.same_threshold
            and (match.view_support >= self.required_views or match.deep_top2 >= self.strong_deep)
            and (attr >= self.attr_threshold or match.deep_top2 >= self.strong_deep)
        )

    def close(self) -> None:
        with self.lock:
            if self.gallery is not None:
                for profile in self.profiles.values():
                    self.gallery.save(profile)
            self.debug.close()


# ---------------------------------------------------------------------------
# Camera worker
# ---------------------------------------------------------------------------


class LockedModel:
    def __init__(self, model, lock: threading.Lock):
        self.model = model
        self.lock = lock

    def extract_batch(self, crops):
        with self.lock:
            return self.model.extract_batch(crops)

    def describe(self):
        return self.model.describe()


class CameraWorker(threading.Thread):
    def __init__(self, name: str, source: str, cfg: Mapping[str, object], models: Mapping[str, LockedModel],
                 engine: IdentityEngine, floor, stop_event: threading.Event, run_dir: Path, show: bool):
        super().__init__(name=f"reid-{name}", daemon=True)
        self.name_id = name
        self.source = source
        self.cfg = cfg
        self.models = models
        self.engine = engine
        self.floor = floor
        self.stop_event = stop_event
        self.run_dir = run_dir
        self.show = show
        self.error = None
        self.stats = defaultdict(int)
        self.states: Dict[int, TrackState] = {}
        self.seen: Dict[int, int] = {}
        self.segments: Dict[int, int] = {}
        self.last_overlap: Dict[str, bool] = {}
        self.fps = 20.0

    def _capture(self):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000|stimeout;5000000")
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"{self.name_id}: cannot open source")
        return cap

    def _feature_views(self, person: np.ndarray, frame_no: int, state: TrackState) -> Dict[str, np.ndarray]:
        views = {"full": person, "light": illumination_variant(person)}
        part_every = max(1, int(self.cfg.get("part_interval", 10)))
        if frame_no % part_every == 0 or state.recovery_left > 0:
            h, w = person.shape[:2]
            if h >= 40 and w >= 20:
                views.update({
                    "upper": person[:max(1, int(h * 0.68))],
                    "torso": person[int(h * 0.08):max(int(h * 0.74), int(h * 0.08) + 1)],
                    "lower": person[int(h * 0.32):],
                })
        return views

    def _extract(self, image: np.ndarray, item: dict, frame_no: int, frame_size: Tuple[int, int]) -> bool:
        state: TrackState = item["state"]
        interval = max(1, int(self.cfg.get("feature_interval", 5)))
        due = state.recovery_left > 0 or frame_no - state.last_feature_frame >= interval
        if not due:
            return False
        person = crop(image, item["bbox"])
        score = quality(person) if person is not None else 0.0
        if person is None or score < float(self.cfg.get("min_quality", 0.20)):
            return False
        views = self._feature_views(person, frame_no, state)
        names = [v for v in VIEWS if v in views]
        inputs = [views[v] for v in names]
        values = {
            model: self.models[model].extract_batch(inputs)
            for model in MODELS
        }
        for model in MODELS:
            arr = np.asarray(values[model], np.float32)
            if arr.shape[0] != len(names):
                raise RuntimeError(f"{self.name_id}: {model} returned {arr.shape[0]} features for {len(names)} views")
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"{self.name_id}: {model} returned non-finite features")
            if np.any(np.linalg.norm(arr, axis=1) <= 0.0):
                raise RuntimeError(f"{self.name_id}: {model} returned zero feature")

        deep = {
            model: {name: unit(vector) for name, vector in zip(names, values[model])}
            for model in MODELS
        }
        floor = None
        if self.floor is not None:
            try:
                value = self.floor.position(self.name_id, item["bbox"], frame_size)
                if value is not None:
                    floor = (float(value.x), float(value.y), str(value.group))
            except Exception:
                floor = None
        obs = Observation(
            camera=self.name_id,
            track_key=state.key,
            timestamp=frame_no / max(self.fps, 1.0),
            bbox=item["bbox"],
            deep=deep,
            attributes=appearance(person),
            geometry=person_geometry(item["bbox"], frame_size),
            floor=floor,
            quality=float(score),
        )
        state.last_feature_frame = frame_no
        if state.recovery_left > 0:
            state.recovery_left -= 1
            self.stats["recovery_features"] += 1
        self.engine.observe(obs, state)
        self.stats["features"] += 1
        return True

    def _draw(self, image: np.ndarray, item: dict, blocked: bool) -> None:
        state: TrackState = item["state"]
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
        palette = [
            (66,199,125), (203,74,221), (255,90,30), (0,165,255),
            (211,85,186), (0,215,255), (255,191,0), (80,80,220),
        ]
        if state.gid:
            number = int(state.gid[1:])
            colour_value = palette[(number - 1) % len(palette)]
            label = state.gid
        else:
            colour_value = (130, 130, 130)
            label = "PENDING"
        if blocked:
            colour_value = (110, 110, 110)
            label += " NO-FEATURE"
        cv2.rectangle(image, (x1, y1), (x2, y2), colour_value, 2)
        cv2.putText(image, label, (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour_value, 2, cv2.LINE_AA)

    def run(self):
        cap = None
        writer = None
        rows = None
        try:
            cap = self._capture()
            self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise RuntimeError(f"{self.name_id}: invalid dimensions {width}x{height}")
            output = self.run_dir / f"output_{self.name_id}.mp4"
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"{self.name_id}: cannot open output {output}")
            rows = (self.run_dir / f"{self.name_id}.jsonl").open("w", encoding="utf-8")
            detector_cfg = self.cfg["detector"]
            detector = PersonDetector(
                model_path=detector_cfg["model"],
                confidence_threshold=float(detector_cfg["conf"]),
                person_class_id=0,
                tracker_config=detector_cfg["tracker"],
                pose_ensemble=None,
                iou=float(detector_cfg["iou"]),
            )
            gap_frames = max(1, int(float(self.cfg.get("fragment_gap_sec", 2.0)) * self.fps))
            frame = 0
            min_iou = float(self.cfg.get("overlap_iou", 0.05))
            min_inter = float(self.cfg.get("overlap_intersection", 0.20))
            while not self.stop_event.is_set():
                ok, image = cap.read()
                if not ok:
                    break
                frame += 1
                raw = [x for x in detector.track(image) if x.track_id is not None]
                prepared = []
                for det in raw:
                    tid = int(det.track_id)
                    previous = self.seen.get(tid)
                    if previous is None or frame - previous > gap_frames:
                        self.segments[tid] = self.segments.get(tid, 0) + 1
                    self.seen[tid] = frame
                    seg = self.segments[tid]
                    key = f"{self.name_id}:{tid}:{seg}"
                    state = self.states.get(tid)
                    if state is None or state.key != key:
                        state = TrackState(self.name_id, tid, seg, key)
                        self.states[tid] = state
                    state.last_frame = frame
                    state.last_seen = frame / self.fps
                    prepared.append({
                        "tid": tid,
                        "state": state,
                        "bbox": (float(det.x1), float(det.y1), float(det.x2), float(det.y2)),
                        "conf": float(det.confidence),
                    })
                self.stats["frames"] = frame
                self.stats["detections"] += len(prepared)

                blocked = set()
                partners = defaultdict(list)
                for i in range(len(prepared)):
                    for j in range(i + 1, len(prepared)):
                        if overlap(prepared[i]["bbox"], prepared[j]["bbox"], min_iou, min_inter):
                            a, b = prepared[i]["tid"], prepared[j]["tid"]
                            blocked.update((a, b))
                            partners[a].append(b)
                            partners[b].append(a)
                if blocked:
                    self.stats["overlap_frames"] += 1

                for item in prepared:
                    state = item["state"]
                    now_overlap = item["tid"] in blocked
                    was_overlap = self.last_overlap.get(state.key, False)
                    if now_overlap and not was_overlap:
                        state.recovery_left = 0
                        self.stats["overlap_events"] += 1
                    elif was_overlap and not now_overlap:
                        state.recovery_left = max(1, int(self.cfg.get("recovery_samples", 2)))
                    self.last_overlap[state.key] = now_overlap
                    state.overlap = now_overlap

                    extracted = False
                    if now_overlap:
                        self.stats["overlap_skips"] += 1
                        self.engine.stats["feature_skipped_overlap"] += 1
                    else:
                        extracted = self._extract(image, item, frame, (width, height))
                    rows.write(json.dumps({
                        "camera": self.name_id,
                        "frame": frame,
                        "timestamp": frame / self.fps,
                        "tracklet_key": state.key,
                        "track_id": item["tid"],
                        "segment": state.segment,
                        "bbox": list(item["bbox"]),
                        "detection_score": item["conf"],
                        "overlap_blocked": now_overlap,
                        "overlap_partners": partners.get(item["tid"], []),
                        "feature_extracted": extracted,
                        "gid": state.gid,
                    }, separators=(",", ":")) + "\n")
                    self._draw(image, item, now_overlap)

                cv2.putText(image, f"{self.name_id} | 3-model ReID | overlap-protected",
                            (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
                writer.write(image)
                if self.show:
                    cv2.imshow(self.name_id, image)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        self.stop_event.set()
                        break
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
# CLI
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
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def build_floor(cfg: Mapping[str, object]):
    if load_calibration is None or FloorFrame is None:
        print("[geometry] unavailable -> fail-open")
        return None
    gcfg = cfg.get("geometry", {}) or {}
    if not bool(gcfg.get("enabled", True)):
        print("[geometry] disabled")
        return None
    path = str(gcfg.get("calibration_path", "calibration/floor_frame.json"))
    try:
        record = load_calibration(path)
        if record is None:
            print(f"[geometry] no calibration at {path} -> fail-open")
            return None
        print("[geometry] ACTIVE")
        print(record.summary())
        return FloorFrame(record)
    except Exception as exc:  # noqa: BLE001
        print(f"[geometry] calibration unavailable: {exc} -> fail-open")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent final multi-camera Person Re-ID")
    parser.add_argument("--config", default="rebuild/config_live_final.yaml")
    parser.add_argument("--camera", action="append", required=True, type=parse_camera)
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl-C")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignore persistent identity gallery for this run")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_cfg = cfg.get("output", {}) or {}
    root = Path(output_cfg.get("root", "rebuild_outputs_live_final")) / f"run_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    icfg = cfg.get("identity", {}) or {}
    gallery = None
    state = Path(icfg.get("state_path", "identity_state/live_reid_final.sqlite3"))
    if bool(icfg.get("persistence", True)) and not args.fresh:
        gallery = Gallery(state)

    engine = IdentityEngine(icfg, gallery, root / "match_debug.jsonl")
    floor = build_floor(cfg)
    models_cfg = cfg.get("models", {}) or {}
    device = str(models_cfg.get("device", "cuda"))

    resnet = ReIDExtractor(
        weights=models_cfg["resnet_weights"], device=device,
        tap=models_cfg.get("resnet_tap", "post_relu"),
        max_batch=int(models_cfg.get("resnet_batch", 32)),
        model=models_cfg.get("resnet_model"),
    )
    swin = NVIDIASwinReIDExtractor(
        models_cfg["swin_weights"], device=device,
        max_batch=int(models_cfg.get("swin_batch", 16)),
    )
    solider = SOLIDERReIDExtractor(
        models_cfg["solider_weights"], device=device,
        max_batch=int(models_cfg.get("solider_batch", 16)),
    )

    print("=" * 82)
    print("             FINAL INDEPENDENT MULTIMODEL PERSON RE-ID")
    print("=" * 82)
    print(f"run:        {stamp}")
    print(f"cameras:    {', '.join(name for name, _ in args.camera)}")
    print("detector:   YOLO11m + ByteTrack")
    print("models:     NVIDIA ResNet + NVIDIA Swin + SOLIDER")
    print("views:      full + light + upper + torso + lower")
    print("attributes: shirt colour + bottom colour + cloth pattern + head appearance")
    print("sampling:   every 5 frames; parts every 10; overlap blocks BOTH tracks")
    print("recovery:   2 feature samples after an overlap ends")
    print("identity:   every extracted feature re-verifies its current GID")
    print("switch:     replacement must win feature evidence for consecutive samples")
    print("new GID:    requires two internally consistent multimodel observations")
    print(f"gallery:    {'PERSISTENT' if gallery else 'FRESH/RUN-LOCAL'}")
    print("dependency: NO seif744/Inference_PersonReid dependency")
    print("=" * 82)
    print(f"[model] {resnet.describe()}")
    print(f"[model] {swin.describe()}")
    print(f"[model] {solider.describe()}")

    lock = threading.Lock()
    shared = {
        "resnet": LockedModel(resnet, lock),
        "swin": LockedModel(swin, lock),
        "solider": LockedModel(solider, lock),
    }

    stop_event = threading.Event()
    workers = []
    worker_cfg = {
        **(cfg.get("capture", {}) or {}),
        "detector": cfg["detector"],
        "fragment_gap_sec": float((cfg.get("tracking", {}) or {}).get("fragment_gap_sec", 2.0)),
        "part_interval": int((cfg.get("capture", {}) or {}).get("part_interval", 10)),
        "overlap_iou": float((cfg.get("capture", {}) or {}).get("overlap_iou", 0.05)),
        "overlap_intersection": float((cfg.get("capture", {}) or {}).get("overlap_intersection", 0.20)),
        "recovery_samples": int((cfg.get("capture", {}) or {}).get("recovery_samples", 2)),
    }
    for name, source in args.camera:
        workers.append(CameraWorker(name, source, worker_cfg, shared, engine, floor, stop_event, root, args.show))

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
            time.sleep(0.05 if args.show else 0.2)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=20.0)
        if args.show:
            cv2.destroyAllWindows()
        engine.close()
        if gallery:
            gallery.close()

    print("\n" + "=" * 82)
    print("                         RUN COMPLETE")
    print("=" * 82)
    for worker in workers:
        print(
            f"{worker.name_id}: frames={worker.stats['frames']} "
            f"detections={worker.stats['detections']} features={worker.stats['features']} "
            f"overlap_skips={worker.stats['overlap_skips']} "
            f"recovery_features={worker.stats['recovery_features']}"
        )
        if worker.error:
            print(f"  ERROR: {worker.error}")
    for key in (
        "feature_observations", "feature_skipped_overlap",
        "existing_gid_assignments", "new_gid_assignments",
        "gid_switches", "verification_failures",
        "rejected_candidates", "pending_new",
    ):
        print(f"{key:28s}: {int(engine.stats[key])}")
    print("FINAL GLOBAL IDS:")
    for gid, profile in sorted(engine.profiles.items(), key=lambda x: int(x[0][1:])):
        print(
            f"  {gid}: cameras={','.join(sorted(profile.cameras))} "
            f"tracks={len(profile.tracks)} observations={profile.observations}"
        )
    print(f"output: {root}")
    print(f"debug:  {root / 'match_debug.jsonl'}")
    return 1 if any(worker.error for worker in workers) else 0


if __name__ == "__main__":
    raise SystemExit(main())
