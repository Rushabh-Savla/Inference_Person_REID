from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        return value / (np.linalg.norm(value) + 1e-12)
    return value / (np.linalg.norm(value, axis=1, keepdims=True) + 1e-12)


def crop(frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def quality(image: np.ndarray, frame_shape: Tuple[int, int]) -> float:
    if image is None or image.size == 0:
        return 0.0
    h, w = image.shape[:2]
    fh, fw = frame_shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 180.0)
    bright = max(0.0, 1.0 - abs(float(gray.mean()) - 128.0) / 128.0)
    area = min(1.0, (h * w) / max(1.0, fh * fw * 0.05))
    height = min(1.0, h / 240.0)
    aspect = min(1.0, (h / max(1.0, w)) / 2.0)
    return float(0.30 * blur + 0.20 * bright + 0.20 * area + 0.20 * height + 0.10 * aspect)


@dataclass
class Observation:
    camera: str
    frame: int
    timestamp: float
    track_id: int
    bbox: Tuple[int, int, int, int]
    detection_score: float
    quality: float
    embedding_index: Optional[int] = None


@dataclass
class Tracklet:
    camera: str
    track_id: int
    segment: int
    fps: float
    observations: List[Observation] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    embedding_quality: List[float] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.camera}:{self.track_id}:{self.segment}"

    @property
    def start(self) -> float:
        return self.observations[0].timestamp if self.observations else 0.0

    @property
    def end(self) -> float:
        return self.observations[-1].timestamp if self.observations else 0.0

    @property
    def count(self) -> int:
        return len(self.embeddings)

    def add_embedding(self, embedding: np.ndarray, score: float) -> int:
        idx = len(self.embeddings)
        self.embeddings.append(unit(embedding))
        self.embedding_quality.append(float(score))
        return idx

    def gallery(self, size: int) -> np.ndarray:
        if not self.embeddings:
            raise ValueError(f"No embeddings in {self.key}")
        order = np.argsort(self.embedding_quality)[::-1][: max(1, min(size, len(self.embeddings)))]
        return unit(np.stack([self.embeddings[i] for i in order]))

    def prototype(self, size: int) -> np.ndarray:
        order = np.argsort(self.embedding_quality)[::-1][: max(1, min(size, len(self.embeddings)))]
        weights = np.asarray([max(0.10, self.embedding_quality[i]) for i in order], dtype=np.float32)
        matrix = np.stack([self.embeddings[i] for i in order])
        return unit((matrix * weights[:, None]).sum(axis=0) / weights.sum())


@dataclass(frozen=True)
class Match:
    left: str
    right: str
    score: float
    margin_left: float
    margin_right: float
    reciprocal: bool


class UnionFind:
    def __init__(self, keys: Iterable[str]):
        self.parent = {k: k for k in keys}
        self.rank = {k: 0 for k in keys}

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            nxt = self.parent[key]
            self.parent[key] = root
            key = nxt
        return root

    def union(self, a: str, b: str) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True


class GlobalIdentityEngine:
    """One shared global identity space for every camera.

    The tracker never creates a global ID. Finished tracklets are compared against
    every other finished tracklet using the same appearance representation.
    Same-camera temporal overlap is a hard impossibility constraint; elapsed time
    is NOT a reason to reject a ReID match.
    """

    def __init__(self, threshold: float, margin: float, strong: float, bank_size: int):
        self.threshold = float(threshold)
        self.margin = float(margin)
        self.strong = float(strong)
        self.bank_size = int(bank_size)

    @staticmethod
    def overlap(a: Tracklet, b: Tracklet) -> bool:
        return not (a.end < b.start or b.end < a.start)

    def allowed(self, a: Tracklet, b: Tracklet) -> bool:
        if a.camera != b.camera:
            return True
        return not self.overlap(a, b)

    def score(self, a: Tracklet, b: Tracklet) -> float:
        ap = a.prototype(self.bank_size)
        bp = b.prototype(self.bank_size)
        prototype = float(ap @ bp)
        ag = a.gallery(self.bank_size)
        bg = b.gallery(self.bank_size)
        pair = np.sort((ag @ bg.T).reshape(-1))
        top = float(np.mean(pair[-min(5, len(pair)) :]))
        return float(0.55 * prototype + 0.45 * top)

    def reconcile(self, tracklets: Dict[str, Tracklet]) -> Tuple[Dict[str, str], List[Match]]:
        usable = {k: v for k, v in tracklets.items() if v.count > 0}
        keys = sorted(usable)
        if not keys:
            return {}, []

        ranked: Dict[str, List[Tuple[float, str]]] = {k: [] for k in keys}
        pair_scores: Dict[Tuple[str, str], float] = {}

        for i, left_key in enumerate(keys):
            left = usable[left_key]
            for right_key in keys[i + 1 :]:
                right = usable[right_key]
                if not self.allowed(left, right):
                    continue
                score = self.score(left, right)
                pair_scores[(left_key, right_key)] = score
                ranked[left_key].append((score, right_key))
                ranked[right_key].append((score, left_key))

        for values in ranked.values():
            values.sort(key=lambda x: x[0], reverse=True)

        uf = UnionFind(keys)
        accepted: List[Match] = []

        for (left_key, right_key), score in sorted(pair_scores.items(), key=lambda x: x[1], reverse=True):
            left_rank = next(i for i, (_, k) in enumerate(ranked[left_key]) if k == right_key)
            right_rank = next(i for i, (_, k) in enumerate(ranked[right_key]) if k == left_key)
            reciprocal = left_rank == 0 and right_rank == 0
            left_second = ranked[left_key][1][0] if left_rank == 0 and len(ranked[left_key]) > 1 else 0.0
            right_second = ranked[right_key][1][0] if right_rank == 0 and len(ranked[right_key]) > 1 else 0.0
            left_margin = score - left_second if left_rank == 0 else 0.0
            right_margin = score - right_second if right_rank == 0 else 0.0

            accept = score >= self.threshold and (
                (reciprocal and min(left_margin, right_margin) >= self.margin)
                or score >= self.strong
            )
            if not accept:
                continue
            if self._would_create_same_camera_overlap(uf, left_key, right_key, usable):
                continue
            if uf.union(left_key, right_key):
                accepted.append(Match(left_key, right_key, score, left_margin, right_margin, reciprocal))

        roots: Dict[str, int] = {}
        mapping: Dict[str, str] = {}
        next_id = 1
        for key in keys:
            root = uf.find(key)
            if root not in roots:
                roots[root] = next_id
                next_id += 1
            mapping[key] = f"G{roots[root]:06d}"

        return mapping, accepted

    def _would_create_same_camera_overlap(self, uf: UnionFind, left: str, right: str, tracks: Dict[str, Tracklet]) -> bool:
        lr, rr = uf.find(left), uf.find(right)
        if lr == rr:
            return False
        left_members = [k for k in tracks if uf.find(k) == lr]
        right_members = [k for k in tracks if uf.find(k) == rr]
        for a in left_members:
            for b in right_members:
                if tracks[a].camera == tracks[b].camera and self.overlap(tracks[a], tracks[b]):
                    return True
        return False

    def summarize_scores(self, tracklets: Dict[str, Tracklet]) -> Dict[str, float]:
        values = []
        keys = sorted(tracklets)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if not self.allowed(tracklets[a], tracklets[b]):
                    continue
                values.append(self.score(tracklets[a], tracklets[b]))
        if not values:
            return {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0}
        values = np.asarray(values, dtype=np.float32)
        return {
            "count": float(len(values)),
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }


class LiveGlobalGallery:
    """Online global gallery using the same score and no camera-specific logic."""

    def __init__(self, threshold: float, margin: float, bank_size: int, strong: float, min_embeddings: int):
        self.engine = GlobalIdentityEngine(threshold, margin, strong, bank_size)
        self.min_embeddings = int(min_embeddings)
        self.tracklets: Dict[str, Tracklet] = {}
        self.track_to_gid: Dict[str, str] = {}
        self.gid_to_tracks: Dict[str, List[str]] = {}
        self.next_id = 1

    def _new_gid(self) -> str:
        gid = f"G{self.next_id:06d}"
        self.next_id += 1
        self.gid_to_tracks[gid] = []
        return gid

    def add(self, tracklet: Tracklet) -> Tuple[str, str, float]:
        self.tracklets[tracklet.key] = tracklet
        if tracklet.count < self.min_embeddings:
            return "PENDING", "pending", 0.0
        candidates = []
        for key, other in self.tracklets.items():
            if key == tracklet.key or other.count < self.min_embeddings:
                continue
            if other.camera == tracklet.camera and self.engine.overlap(other, tracklet):
                continue
            score = self.engine.score(tracklet, other)
            candidates.append((score, key))
        candidates.sort(reverse=True)
        if candidates:
            best, other_key = candidates[0]
            second = candidates[1][0] if len(candidates) > 1 else 0.0
            other_gid = self.track_to_gid.get(other_key)
            if other_gid and best >= self.engine.threshold and (best - second >= self.engine.margin or best >= self.engine.strong):
                self.track_to_gid[tracklet.key] = other_gid
                self.gid_to_tracks.setdefault(other_gid, []).append(tracklet.key)
                return other_gid, "reidentified", best
        gid = self._new_gid()
        self.track_to_gid[tracklet.key] = gid
        self.gid_to_tracks[gid].append(tracklet.key)
        return gid, "new", 1.0
