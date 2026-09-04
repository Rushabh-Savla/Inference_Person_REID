from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from rebuild.identity_body_v6 import GlobalIdentityBodyV6
from rebuild.identity_v3 import Feature, Tracklet


@dataclass
class LocalNode:
    key: str
    camera: str
    local_gid: str
    tracks: List[Tracklet] = field(default_factory=list)

    @property
    def start(self) -> float:
        return min((float(t.start) for t in self.tracks), default=0.0)

    @property
    def end(self) -> float:
        return max((float(t.end) for t in self.tracks), default=0.0)

    @property
    def features(self) -> List[Feature]:
        out: List[Feature] = []
        for track in self.tracks:
            out.extend(track.features)
        return out


class LocalGlobalResolver:
    """Cross-camera reconciliation over independent camera-local V6 identities.

    The critical invariant is that cross-camera reconciliation NEVER starts from
    global GIDs. Every camera first gets an independent V6 solution. The second
    stage only matches camera-local identities and creates a fresh global identity
    graph. Thus a bad GID in camera B cannot contaminate camera C before matching.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.score_engine = GlobalIdentityBodyV6(cfg)
        self.min_reid = float(cfg.get("cross_local_min_reid", 0.66))
        self.min_score = float(cfg.get("cross_local_min_score", 0.67))
        self.margin = float(cfg.get("cross_local_margin", 0.04))
        self.strong_reid = float(cfg.get("cross_local_strong_reid", 0.72))
        self.color_weight = float(cfg.get("cross_local_color_weight", 0.06))
        self.time_weight = float(cfg.get("cross_local_time_weight", 0.04))
        self.shape_weight = float(cfg.get("cross_local_shape_weight", 0.02))
        self.camera_offsets = {str(k): float(v) for k, v in (cfg.get("camera_time_offsets_sec", {}) or {}).items()}
        self.links: List[dict] = []

    @staticmethod
    def _unit(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        return value / (np.linalg.norm(value) + 1e-12)

    @classmethod
    def colour_similarity(cls, left: Tracklet, right: Tracklet) -> float:
        a = getattr(left, "colour_signature", None)
        b = getattr(right, "colour_signature", None)
        if a is None or b is None:
            return 0.5
        return float(np.clip(np.dot(cls._unit(a), cls._unit(b)), 0.0, 1.0))

    def _time_score(self, left: Tracklet, right: Tracklet) -> float:
        loff = self.camera_offsets.get(left.camera, 0.0)
        roff = self.camera_offsets.get(right.camera, 0.0)
        a0, a1 = left.start + loff, left.end + loff
        b0, b1 = right.start + roff, right.end + roff
        if not (a1 < b0 or b1 < a0):
            return 1.0
        gap = min(abs(a1 - b0), abs(b1 - a0))
        tol = float(self.cfg.get("cross_local_time_tolerance_sec", 12.0))
        return float(max(0.0, 1.0 - gap / max(tol, 1e-6)))

    @staticmethod
    def _shape_score(left: Tracklet, right: Tracklet) -> float:
        if left.shape <= 0 or right.shape <= 0:
            return 0.5
        ratio = min(left.shape, right.shape) / max(left.shape, right.shape)
        return float(max(0.0, min(1.0, ratio)))

    def _track_pair(self, left: Tracklet, right: Tracklet) -> dict:
        reid, support, partial = self.score_engine.body_score(left.features, right.features)
        colour = self.colour_similarity(left, right)
        time = self._time_score(left, right)
        shape = self._shape_score(left, right)
        score = (
            (1.0 - self.color_weight - self.time_weight - self.shape_weight) * reid
            + self.color_weight * colour
            + self.time_weight * time
            + self.shape_weight * shape
        )
        return {
            "score": float(score),
            "reid": float(reid),
            "support": int(support),
            "colour": float(colour),
            "time": float(time),
            "shape": float(shape),
            "partial": bool(partial),
        }

    def _node_score(self, left: LocalNode, right: LocalNode) -> dict:
        pairs = []
        for a in left.tracks:
            for b in right.tracks:
                item = self._track_pair(a, b)
                item["left_track"] = a.key
                item["right_track"] = b.key
                pairs.append(item)
        if not pairs:
            return {"score": 0.0, "reid": 0.0, "support": 0, "colour": 0.5, "time": 0.0, "shape": 0.5}
        pairs.sort(key=lambda x: x["score"], reverse=True)
        top = pairs[: min(3, len(pairs))]
        best = top[0]
        consensus = float(np.mean([x["score"] for x in top]))
        score = 0.70 * best["score"] + 0.30 * consensus
        return {
            "score": float(score),
            "reid": float(best["reid"]),
            "support": int(max(x["support"] for x in top)),
            "colour": float(best["colour"]),
            "time": float(best["time"]),
            "shape": float(best["shape"]),
            "best_pair": (best["left_track"], best["right_track"]),
        }

    @staticmethod
    def _hungarian(matrix: np.ndarray) -> List[Tuple[int, int]]:
        if matrix.size == 0:
            return []
        try:
            from scipy.optimize import linear_sum_assignment
            size = max(matrix.shape)
            padded = np.zeros((size, size), dtype=np.float64)
            padded[: matrix.shape[0], : matrix.shape[1]] = matrix
            rows, cols = linear_sum_assignment(-padded)
            return [
                (int(r), int(c))
                for r, c in zip(rows, cols)
                if r < matrix.shape[0] and c < matrix.shape[1] and matrix[r, c] > 0.0
            ]
        except Exception:
            edges = sorted(
                (float(matrix[i, j]), i, j)
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])
                if matrix[i, j] > 0.0
            )[::-1]
            used_r, used_c = set(), set()
            out = []
            for _, i, j in edges:
                if i in used_r or j in used_c:
                    continue
                used_r.add(i); used_c.add(j); out.append((i, j))
            return out

    def _pair_camera_nodes(self, left: List[LocalNode], right: List[LocalNode]) -> List[dict]:
        if not left or not right:
            return []
        matrix = np.zeros((len(left), len(right)), dtype=np.float64)
        details: Dict[Tuple[int, int], dict] = {}
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                item = self._node_score(a, b)
                details[(i, j)] = item
                if item["reid"] < self.min_reid or item["score"] < self.min_score:
                    continue
                if item["reid"] < self.strong_reid and item["colour"] < 0.55:
                    continue
                matrix[i, j] = item["score"]

        accepted: List[dict] = []
        for i, j in self._hungarian(matrix):
            item = details[(i, j)]
            row = np.sort(matrix[i][matrix[i] > 0.0])[::-1]
            col = np.sort(matrix[:, j][matrix[:, j] > 0.0])[::-1]
            rmargin = float(row[0] - row[1]) if len(row) > 1 else float("inf")
            cmargin = float(col[0] - col[1]) if len(col) > 1 else float("inf")
            if rmargin < self.margin or cmargin < self.margin:
                continue
            accepted.append({"left": left[i].key, "right": right[j].key, **item, "row_margin": rmargin, "col_margin": cmargin})
        return accepted

    @staticmethod
    def _union_find(nodes: List[str], edges: List[dict], node_camera: Dict[str, str]) -> List[List[str]]:
        parent = {x: x for x in nodes}
        members = {x: {x} for x in nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def cams(root: str) -> set[str]:
            return {node_camera[x] for x in members[root]}

        for edge in sorted(edges, key=lambda x: x["score"], reverse=True):
            a, b = edge["left"], edge["right"]
            ra, rb = find(a), find(b)
            if ra == rb or not cams(ra).isdisjoint(cams(rb)):
                continue
            parent[rb] = ra
            members[ra].update(members[rb])
            del members[rb]
        out: Dict[str, List[str]] = {}
        for node in nodes:
            out.setdefault(find(node), []).append(node)
        return list(out.values())

    def resolve(self, local_mapping: Dict[str, str], tracks: Dict[str, Tracklet], cameras: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]], List[dict]]:
        nodes: Dict[str, LocalNode] = {}
        for key, gid in local_mapping.items():
            track = tracks[key]
            nkey = f"{track.camera}::{gid}"
            node = nodes.setdefault(nkey, LocalNode(nkey, track.camera, gid))
            node.tracks.append(track)

        node_keys = sorted(nodes)
        node_camera = {key: nodes[key].camera for key in node_keys}
        edges: List[dict] = []
        sorted_cameras = sorted(cameras)
        for i, ca in enumerate(sorted_cameras):
            for cb in sorted_cameras[i + 1:]:
                left = [nodes[k] for k in node_keys if nodes[k].camera == ca]
                right = [nodes[k] for k in node_keys if nodes[k].camera == cb]
                edges.extend(self._pair_camera_nodes(left, right))

        components = self._union_find(node_keys, edges, node_camera)
        ordered = sorted(components, key=lambda comp: (min(nodes[x].start for x in comp), sorted(comp)))
        global_mapping: Dict[str, str] = {}
        component_gids: Dict[str, List[str]] = {}
        for idx, comp in enumerate(ordered, start=1):
            gid = f"G{idx:06d}"
            component_gids[gid] = sorted(comp)
            for nkey in comp:
                node = nodes[nkey]
                for track in node.tracks:
                    global_mapping[track.key] = gid

        self.links = edges
        return global_mapping, component_gids, edges


def colour_signature(image: np.ndarray) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if h < 40 or w < 20:
        return None
    x1, x2 = int(0.18 * w), int(0.82 * w)
    y1, y2 = int(0.12 * h), int(0.72 * h)
    torso = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    hue = hsv[..., 0].astype(np.float32) / 180.0
    hue_hist, _ = np.histogram(hue, bins=8, range=(0.0, 1.0), weights=sat + 0.05)
    neutral = sat < 0.28
    value_hist, _ = np.histogram(val[neutral], bins=4, range=(0.0, 1.0))
    descriptor = np.concatenate([hue_hist, value_hist]).astype(np.float32)
    descriptor /= np.linalg.norm(descriptor) + 1e-12
    return descriptor
