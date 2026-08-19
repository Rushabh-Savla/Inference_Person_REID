from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        return value / (np.linalg.norm(value) + 1e-12)
    return value / (np.linalg.norm(value, axis=1, keepdims=True) + 1e-12)


def crop(frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, min(w, int(x1)))
    y1 = max(0, min(h, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def quality(image: np.ndarray, frame_shape: Tuple[int, int]) -> float:
    if image is None or image.size == 0:
        return 0.0
    h, w = image.shape[:2]
    fh, fw = frame_shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    area = float(h * w) / max(1.0, float(fh * fw))
    blur_score = min(1.0, blur / 180.0)
    bright_score = max(0.0, 1.0 - abs(bright - 128.0) / 128.0)
    area_score = min(1.0, area / 0.05)
    height_score = min(1.0, h / 240.0)
    return float(
        0.35 * blur_score
        + 0.25 * bright_score
        + 0.20 * area_score
        + 0.20 * height_score
    )


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
    fps: float
    observations: List[Observation] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    embedding_quality: List[float] = field(default_factory=list)

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
        index = len(self.embeddings)
        self.embeddings.append(unit(embedding))
        self.embedding_quality.append(float(score))
        return index

    def prototype(self, bank_size: int) -> np.ndarray:
        if not self.embeddings:
            raise ValueError("Tracklet has no embeddings")
        order = np.argsort(self.embedding_quality)[::-1]
        order = order[: max(1, min(bank_size, len(order)))]
        weights = np.asarray(
            [max(0.05, self.embedding_quality[i]) for i in order],
            dtype=np.float32,
        )
        matrix = np.stack([self.embeddings[i] for i in order])
        return unit((matrix * weights[:, None]).sum(axis=0) / weights.sum())

    def gallery(self, bank_size: int) -> np.ndarray:
        order = np.argsort(self.embedding_quality)[::-1]
        order = order[: max(1, min(bank_size, len(order)))]
        return unit(np.stack([self.embeddings[i] for i in order]))


@dataclass
class Identity:
    gid: str
    tracklets: List[str] = field(default_factory=list)


class UnionFind:
    def __init__(self, keys: Iterable[str]):
        self.parent = {key: key for key in keys}
        self.rank = {key: 0 for key in keys}

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            nxt = self.parent[key]
            self.parent[key] = root
            key = nxt
        return root

    def union(self, left: str, right: str) -> bool:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return False
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1
        return True


class OfflineReconciler:
    """Camera-agnostic tracklet graph reconciliation.

    The detector/tracker never assigns global IDs. This class sees finished
    tracklets only, scores tracklet-to-tracklet appearance similarity, and then
    forms global-identity components. Same-camera overlap is a hard physical
    constraint; cross-camera simultaneous visibility is allowed.
    """

    def __init__(
        self,
        same_threshold: float,
        cross_threshold: float,
        min_margin: float,
        bank_size: int,
        max_same_camera_gap_sec: float,
    ):
        self.same_threshold = float(same_threshold)
        self.cross_threshold = float(cross_threshold)
        self.min_margin = float(min_margin)
        self.bank_size = int(bank_size)
        self.max_same_camera_gap_sec = float(max_same_camera_gap_sec)

    @staticmethod
    def score(left: Tracklet, right: Tracklet, bank_size: int) -> float:
        lp = left.prototype(bank_size)
        rp = right.prototype(bank_size)
        prototype = float(lp @ rp)

        lg = left.gallery(bank_size)
        rg = right.gallery(bank_size)
        pair = lg @ rg.T
        flat = np.sort(pair.reshape(-1))
        top = float(np.mean(flat[-min(3, flat.size):]))
        return float(0.65 * prototype + 0.35 * top)

    @staticmethod
    def overlap(left: Tracklet, right: Tracklet) -> bool:
        return not (left.end < right.start or right.end < left.start)

    def allowed(self, left: Tracklet, right: Tracklet) -> bool:
        if left.camera == right.camera:
            if self.overlap(left, right):
                return False
            gap = max(0.0, max(right.start - left.end, left.start - right.end))
            return gap <= self.max_same_camera_gap_sec
        return True

    def reconcile(self, tracklets: Dict[str, Tracklet]) -> Dict[str, str]:
        keys = sorted(tracklets)
        if not keys:
            return {}

        scores: Dict[Tuple[str, str], float] = {}
        best: Dict[str, List[Tuple[float, str]]] = {key: [] for key in keys}

        for index, left_key in enumerate(keys):
            left = tracklets[left_key]
            if left.count == 0:
                continue
            for right_key in keys[index + 1 :]:
                right = tracklets[right_key]
                if right.count == 0:
                    continue
                if not self.allowed(left, right):
                    continue
                value = self.score(left, right, self.bank_size)
                scores[(left_key, right_key)] = value
                best[left_key].append((value, right_key))
                best[right_key].append((value, left_key))

        for key in best:
            best[key].sort(reverse=True)

        uf = UnionFind(keys)
        accepted = []

        for (left_key, right_key), value in sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        ):
            left = tracklets[left_key]
            right = tracklets[right_key]
            threshold = self.same_threshold if left.camera == right.camera else self.cross_threshold
            if value < threshold:
                continue

            left_best = best[left_key]
            right_best = best[right_key]
            left_rank = next((i for i, item in enumerate(left_best) if item[1] == right_key), None)
            right_rank = next((i for i, item in enumerate(right_best) if item[1] == left_key), None)
            if left_rank is None or right_rank is None:
                continue

            left_margin = value - (left_best[1][0] if left_rank == 0 and len(left_best) > 1 else left_best[0][0])
            right_margin = value - (right_best[1][0] if right_rank == 0 and len(right_best) > 1 else right_best[0][0])

            reciprocal = left_rank == 0 and right_rank == 0
            strong = value >= min(0.92, threshold + 0.10)
            if not reciprocal and not strong:
                continue
            if not strong and min(left_margin, right_margin) < self.min_margin:
                continue

            if self._component_would_overlap(uf, left_key, right_key, tracklets):
                continue

            if uf.union(left_key, right_key):
                accepted.append((left_key, right_key, value))

        roots: Dict[str, int] = {}
        mapping: Dict[str, str] = {}
        next_id = 1
        for key in keys:
            root = uf.find(key)
            if root not in roots:
                roots[root] = next_id
                next_id += 1
            mapping[key] = f"G{roots[root]:06d}"
        return mapping

    def _component_would_overlap(
        self,
        uf: UnionFind,
        left_key: str,
        right_key: str,
        tracklets: Dict[str, Tracklet],
    ) -> bool:
        left_root = uf.find(left_key)
        right_root = uf.find(right_key)
        if left_root == right_root:
            return False

        left_members = [key for key in tracklets if uf.find(key) == left_root]
        right_members = [key for key in tracklets if uf.find(key) == right_root]
        for a in left_members:
            for b in right_members:
                if tracklets[a].camera != tracklets[b].camera:
                    continue
                if self.overlap(tracklets[a], tracklets[b]):
                    return True
        return False


class LiveIdentityManager:
    """Online layer using the same Identity/tracklet concepts as batch mode.

    IDs remain provisional until enough embeddings exist. No camera receives a
    special rule; a new track is matched against the same global gallery from
    every source.
    """

    def __init__(self, reconciler: OfflineReconciler, min_embeddings: int):
        self.reconciler = reconciler
        self.min_embeddings = int(min_embeddings)
        self.tracklets: Dict[str, Tracklet] = {}
        self.identities: Dict[str, Identity] = {}
        self.next_id = 1

    def _key(self, camera: str, track_id: int) -> str:
        return f"{camera}:{track_id}"

    def update(
        self,
        camera: str,
        track_id: int,
        fps: float,
        observation: Observation,
        embedding: np.ndarray,
    ) -> Optional[str]:
        key = self._key(camera, track_id)
        tracklet = self.tracklets.get(key)
        if tracklet is None:
            tracklet = Tracklet(camera=camera, track_id=track_id, fps=fps)
            self.tracklets[key] = tracklet

        tracklet.observations.append(observation)
        tracklet.add_embedding(embedding, observation.quality)

        if tracklet.count < self.min_embeddings:
            return None

        if key in self.identities_for_track():
            return self.identities_for_track()[key]

        current = tracklet.prototype(self.reconciler.bank_size)
        candidates = []
        for other_key, other in self.tracklets.items():
            if other_key == key or other.count < self.min_embeddings:
                continue
            if other.camera == camera and self.reconciler.overlap(tracklet, other):
                continue
            score = self.reconciler.score(tracklet, other, self.reconciler.bank_size)
            candidates.append((score, other_key))
        candidates.sort(reverse=True)

        if candidates:
            score, other_key = candidates[0]
            second = candidates[1][0] if len(candidates) > 1 else 0.0
            threshold = (
                self.reconciler.same_threshold
                if self.tracklets[other_key].camera == camera
                else self.reconciler.cross_threshold
            )
            if score >= threshold and score - second >= self.reconciler.min_margin:
                gid = self.identities_for_track().get(other_key)
                if gid is not None:
                    self._bind(key, gid)
                    return gid

        gid = f"G{self.next_id:06d}"
        self.next_id += 1
        self._bind(key, gid)
        self.identities[gid] = Identity(gid=gid, tracklets=[key])
        return gid

    def _bind(self, key: str, gid: str) -> None:
        identity = self.identities.setdefault(gid, Identity(gid=gid))
        if key not in identity.tracklets:
            identity.tracklets.append(key)

    def identities_for_track(self) -> Dict[str, str]:
        out = {}
        for gid, identity in self.identities.items():
            for key in identity.tracklets:
                out[key] = gid
        return out
