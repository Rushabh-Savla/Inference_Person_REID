from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        return value / (np.linalg.norm(value) + 1e-12)
    return value / (np.linalg.norm(value, axis=1, keepdims=True) + 1e-12)


def crop(frame: np.ndarray, box: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def quality(image: np.ndarray) -> float:
    """Camera-agnostic crop quality.

    The old score used crop_area / full_frame_area. That is resolution dependent
    and badly penalizes a wide camera such as cam_213. This score uses only the
    crop itself: sharpness, usable pixel resolution, contrast and person-box
    geometry. Lighting is deliberately NOT treated as a rejection criterion;
    illumination-normalized features are collected separately.
    """
    if image is None or image.size == 0:
        return 0.0
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp = min(1.0, np.log1p(lap) / np.log1p(900.0))
    pixels = min(1.0, np.sqrt(float(h * w)) / 220.0)
    contrast = min(1.0, float(gray.std()) / 55.0)
    ratio = h / max(1.0, w)
    shape = max(0.0, 1.0 - abs(np.log(max(0.25, ratio) / 2.0)) / np.log(4.0))
    return float(0.45 * sharp + 0.30 * pixels + 0.15 * contrast + 0.10 * shape)


def illumination_variant(image: np.ndarray) -> np.ndarray:
    """Create a conservative illumination-robust view without replacing RGB.

    The original crop is always retained. This second view only alters luminance
    with CLAHE and is therefore useful for exposure/shadow differences without
    forcing the ReID network to operate only on a transformed image.
    """
    if image is None or image.size == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(image, 0.35, out, 0.65, 0.0)


def geometry(boxes: Iterable[Tuple[int, int, int, int]]) -> float:
    """Scale-free body-shape proxy: median bbox aspect ratio."""
    values = []
    for x1, y1, x2, y2 in boxes:
        h = max(1.0, float(y2 - y1))
        w = max(1.0, float(x2 - x1))
        values.append(h / w)
    return float(np.median(values)) if values else 0.0


def geometry_similarity(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        return 0.5
    distance = abs(np.log(left / right))
    return float(np.exp(-distance / 0.45))


def diverse_select(matrix: np.ndarray, scores: np.ndarray, size: int) -> np.ndarray:
    if len(matrix) <= size:
        return unit(matrix)
    matrix = unit(matrix)
    scores = np.asarray(scores, dtype=np.float32)
    first = int(np.argmax(scores))
    chosen = [first]
    remaining = set(range(len(matrix)))
    remaining.remove(first)
    while remaining and len(chosen) < size:
        best_idx = None
        best_value = -1e9
        chosen_mat = matrix[chosen]
        for idx in remaining:
            novelty = 1.0 - float(np.max(chosen_mat @ matrix[idx]))
            value = 0.65 * float(scores[idx]) + 0.35 * novelty
            if value > best_value:
                best_value = value
                best_idx = idx
        chosen.append(int(best_idx))
        remaining.remove(best_idx)
    return matrix[chosen]


@dataclass
class Tracklet:
    camera: str
    track_id: int
    segment: int
    fps: float
    observations: List[dict] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    embedding_quality: List[float] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.camera}:{self.track_id}:{self.segment}"

    @property
    def start(self) -> float:
        return self.observations[0]["timestamp"] if self.observations else 0.0

    @property
    def end(self) -> float:
        return self.observations[-1]["timestamp"] if self.observations else 0.0

    @property
    def count(self) -> int:
        return len(self.embeddings)

    @property
    def shape(self) -> float:
        return geometry(obs["bbox"] for obs in self.observations)

    def add(self, embedding: np.ndarray, score: float, meta: dict) -> None:
        self.embeddings.append(unit(embedding))
        self.embedding_quality.append(float(score))
        if meta:
            self.observations.append(meta)

    def gallery(self, size: int) -> np.ndarray:
        if not self.embeddings:
            raise ValueError(f"No embeddings for {self.key}")
        matrix = np.stack(self.embeddings)
        scores = np.asarray(self.embedding_quality, dtype=np.float32)
        return diverse_select(matrix, scores, max(1, int(size)))

    def prototype(self, size: int) -> np.ndarray:
        values = self.gallery(size)
        return unit(values.mean(axis=0))


@dataclass(frozen=True)
class Match:
    left: str
    right: str
    score: float
    margin_left: float
    margin_right: float
    reciprocal: bool
    relation: str


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
    """Shared multi-view global identity space.

    Every accepted identity contains a diverse gallery of observations. The
    gallery is not collapsed to one centroid, so front/rear/side/lighting
    changes can remain represented. Appearance is the primary signal. Geometry
    is only a tie-breaker when appearance candidates are already close.
    """

    def __init__(self, threshold: float, margin: float, strong: float, bank_size: int, passes: int = 3):
        self.threshold = float(threshold)
        self.margin = float(margin)
        self.strong = float(strong)
        self.bank_size = int(bank_size)
        self.passes = max(1, int(passes))

    @staticmethod
    def overlap(a: Tracklet, b: Tracklet) -> bool:
        return not (a.end < b.start or b.end < a.start)

    @staticmethod
    def allowed(a: Tracklet, b: Tracklet) -> bool:
        return a.camera != b.camera or not GlobalIdentityEngine.overlap(a, b)

    def score_tracks(self, left: Tracklet, right: Tracklet) -> float:
        a = left.gallery(self.bank_size)
        b = right.gallery(self.bank_size)
        sims = a @ b.T
        best = float(sims.max())
        flat = np.sort(sims.reshape(-1))
        top = float(np.mean(flat[-min(4, len(flat)) :]))
        proto = float(left.prototype(self.bank_size) @ right.prototype(self.bank_size))
        return float(0.50 * best + 0.30 * top + 0.20 * proto)

    def score_group(self, query: Tracklet, members: List[Tracklet]) -> float:
        if not members:
            return 0.0
        vectors = []
        scores = []
        for member in members:
            gallery = member.gallery(self.bank_size)
            vectors.extend(gallery)
            scores.extend([1.0] * len(gallery))
        matrix = np.stack(vectors)
        matrix = diverse_select(matrix, np.asarray(scores, dtype=np.float32), min(4 * self.bank_size, len(matrix)))
        q = query.gallery(self.bank_size)
        sims = q @ matrix.T
        best = float(sims.max())
        flat = np.sort(sims.reshape(-1))
        top = float(np.mean(flat[-min(6, len(flat)) :]))
        proto = float(query.prototype(self.bank_size) @ unit(matrix.mean(axis=0)))
        return float(0.50 * best + 0.32 * top + 0.18 * proto)

    @staticmethod
    def group_shape(members: List[Tracklet]) -> float:
        values = [m.shape for m in members if m.shape > 0]
        return float(np.median(values)) if values else 0.0

    def _conflict(self, uf: UnionFind, a: str, b: str, tracks: Dict[str, Tracklet]) -> bool:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            return False
        left = [k for k in tracks if uf.find(k) == ra]
        right = [k for k in tracks if uf.find(k) == rb]
        for l in left:
            for r in right:
                if tracks[l].camera == tracks[r].camera and self.overlap(tracks[l], tracks[r]):
                    return True
        return False

    def reconcile(self, tracklets: Dict[str, Tracklet]) -> Tuple[Dict[str, str], List[Match]]:
        usable = {k: v for k, v in tracklets.items() if v.count > 0}
        keys = sorted(usable)
        if not keys:
            return {}, []

        uf = UnionFind(keys)
        matches: List[Match] = []
        pair_scores: Dict[Tuple[str, str], float] = {}
        ranked: Dict[str, List[Tuple[float, str]]] = {k: [] for k in keys}

        for i, left_key in enumerate(keys):
            for right_key in keys[i + 1 :]:
                left, right = usable[left_key], usable[right_key]
                if not self.allowed(left, right):
                    continue
                score = self.score_tracks(left, right)
                pair_scores[(left_key, right_key)] = score
                ranked[left_key].append((score, right_key))
                ranked[right_key].append((score, left_key))

        for values in ranked.values():
            values.sort(reverse=True)

        for (left_key, right_key), score in sorted(pair_scores.items(), key=lambda x: x[1], reverse=True):
            left_rank = next(i for i, (_, key) in enumerate(ranked[left_key]) if key == right_key)
            right_rank = next(i for i, (_, key) in enumerate(ranked[right_key]) if key == left_key)
            reciprocal = left_rank == 0 and right_rank == 0
            left_second = ranked[left_key][1][0] if left_rank == 0 and len(ranked[left_key]) > 1 else 0.0
            right_second = ranked[right_key][1][0] if right_rank == 0 and len(ranked[right_key]) > 1 else 0.0
            margin_left = score - left_second if left_rank == 0 else 0.0
            margin_right = score - right_second if right_rank == 0 else 0.0
            accept = score >= self.threshold and ((reciprocal and min(margin_left, margin_right) >= self.margin) or score >= self.strong)
            if not accept or self._conflict(uf, left_key, right_key, usable):
                continue
            if uf.union(left_key, right_key):
                relation = "same" if left.camera == right.camera else "cross"
                matches.append(Match(left_key, right_key, score, margin_left, margin_right, reciprocal, relation))

        # Now perform true global-gallery passes over the components created above.
        # This is what the old pairwise-only design lacked: a later tracklet can
        # match the accumulated identity gallery even when it is not close enough
        # to any single early tracklet.
        for _ in range(self.passes):
            groups: Dict[str, List[str]] = {}
            for key in keys:
                groups.setdefault(uf.find(key), []).append(key)
            changed = False
            components = list(groups.values())
            for key in keys:
                if any(key in members and len(members) > 1 for members in components):
                    continue
                query = usable[key]
                candidates = []
                for members in components:
                    if key in members:
                        continue
                    member_tracks = [usable[item] for item in members]
                    if any(query.camera == item.camera and self.overlap(query, item) for item in member_tracks):
                        continue
                    score = self.score_group(query, member_tracks)
                    shape = geometry_similarity(query.shape, self.group_shape(member_tracks))
                    candidates.append((score, shape, members))
                candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                if not candidates:
                    continue
                best, best_shape, members = candidates[0]
                second = candidates[1][0] if len(candidates) > 1 else 0.0
                margin = best - second
                tied = len(candidates) > 1 and abs(best - second) < self.margin
                if tied and best_shape < candidates[1][1]:
                    continue
                if best >= self.threshold and (margin >= self.margin or best >= self.strong):
                    root = members[0]
                    if not self._conflict(uf, key, root, usable) and uf.union(key, root):
                        relation = "same" if query.camera == usable[root].camera else "cross"
                        matches.append(Match(key, root, best, margin, margin, False, relation))
                        changed = True
            if not changed:
                break

        roots: Dict[str, int] = {}
        mapping = {}
        for key in keys:
            root = uf.find(key)
            roots.setdefault(root, len(roots) + 1)
            mapping[key] = f"G{roots[root]:06d}"
        return mapping, matches

    def gallery_for_mapping(self, tracklets: Dict[str, Tracklet], mapping: Dict[str, str], size: int) -> Dict[str, np.ndarray]:
        groups: Dict[str, List[Tracklet]] = {}
        for key, gid in mapping.items():
            groups.setdefault(gid, []).append(tracklets[key])
        out: Dict[str, np.ndarray] = {}
        for gid, members in groups.items():
            values = []
            scores = []
            for track in members:
                gallery = track.gallery(size)
                values.extend(gallery)
                scores.extend([1.0] * len(gallery))
            matrix = np.stack(values)
            out[gid] = diverse_select(matrix, np.asarray(scores, dtype=np.float32), min(size * 3, len(matrix)))
        return out

    def summarize_scores(self, tracklets: Dict[str, Tracklet]) -> Dict[str, float]:
        values = []
        keys = sorted(tracklets)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if self.allowed(tracklets[a], tracklets[b]):
                    values.append(self.score_tracks(tracklets[a], tracklets[b]))
        if not values:
            return {"count": 0.0, "min": 0.0, "mean": 0.0, "max": 0.0}
        v = np.asarray(values, dtype=np.float32)
        return {"count": float(len(v)), "min": float(v.min()), "mean": float(v.mean()), "max": float(v.max()), "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99))}


class OnlineGlobalGallery:
    def __init__(self, engine: GlobalIdentityEngine, min_embeddings: int, new_id_embeddings: int):
        self.engine = engine
        self.min_embeddings = int(min_embeddings)
        self.new_id_embeddings = max(self.min_embeddings, int(new_id_embeddings))
        self.tracklets: Dict[str, Tracklet] = {}
        self.track_to_gid: Dict[str, str] = {}
        self.gid_to_tracks: Dict[str, List[str]] = {}
        self.next_id = 1

    def _new_gid(self) -> str:
        gid = f"G{self.next_id:06d}"
        self.next_id += 1
        self.gid_to_tracks[gid] = []
        return gid

    def update(self, track: Tracklet) -> Tuple[str, str, float]:
        self.tracklets[track.key] = track
        if track.key in self.track_to_gid:
            return self.track_to_gid[track.key], "track", 1.0
        if track.count < self.min_embeddings:
            return "PENDING", "pending", 0.0

        candidates = []
        for gid, keys in self.gid_to_tracks.items():
            members = [self.tracklets[k] for k in keys]
            if not members:
                continue
            if any(track.camera == item.camera and self.engine.overlap(track, item) for item in members):
                continue
            score = self.engine.score_group(track, members)
            shape = geometry_similarity(track.shape, self.engine.group_shape(members))
            candidates.append((score, shape, gid))
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        if candidates:
            best, best_shape, gid = candidates[0]
            second = candidates[1][0] if len(candidates) > 1 else 0.0
            margin = best - second
            tied = len(candidates) > 1 and margin < self.engine.margin
            if tied and len(candidates) > 1 and best_shape < candidates[1][1]:
                return "PENDING", "pending", best
            if best >= self.engine.threshold and (margin >= self.engine.margin or best >= self.engine.strong):
                self.track_to_gid[track.key] = gid
                self.gid_to_tracks[gid].append(track.key)
                return gid, "reidentified", best

        if track.count >= self.new_id_embeddings:
            gid = self._new_gid()
            self.track_to_gid[track.key] = gid
            self.gid_to_tracks[gid].append(track.key)
            return gid, "new", 1.0
        return "PENDING", "pending", 0.0
