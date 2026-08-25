from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from rebuild.identity_body_v6 import GlobalIdentityBodyV6
from rebuild.identity_v3 import Identity, Tracklet


class GlobalIdentityBodyV6CameraGraph(GlobalIdentityBodyV6):
    """Known-good V6 matcher plus symmetric camera-level identity graph."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.graph_reid_weight = float(cfg.get("cross_graph_reid_weight", 0.78))
        self.graph_color_weight = float(cfg.get("cross_graph_color_weight", 0.12))
        self.graph_time_weight = float(cfg.get("cross_graph_time_weight", 0.06))
        self.graph_shape_weight = float(cfg.get("cross_graph_shape_weight", 0.04))
        self.graph_min_reid = float(cfg.get("cross_graph_min_reid", 0.50))
        self.graph_min_score = float(cfg.get("cross_graph_min_score", 0.58))
        self.graph_margin = float(cfg.get("cross_graph_margin", 0.035))
        self.graph_color_gate = float(cfg.get("cross_graph_color_gate", 0.62))
        self.graph_strong_reid = float(cfg.get("cross_graph_strong_reid", 0.68))
        self.graph_time_tolerance = float(cfg.get("cross_graph_time_tolerance_sec", 8.0))
        self.camera_offsets = {str(k): float(v) for k, v in (cfg.get("camera_time_offsets_sec", {}) or {}).items()}
        self.graph_links: List[dict] = []

    def _span(self, track: Tracklet) -> Tuple[float, float]:
        off = self.camera_offsets.get(track.camera, 0.0)
        return float(track.start + off), float(track.end + off)

    def _time_score(self, left: Tracklet, right: Tracklet) -> float:
        a0, a1 = self._span(left)
        b0, b1 = self._span(right)
        gap = 0.0 if not (a1 < b0 or b1 < a0) else min(abs(a1 - b0), abs(b1 - a0))
        if gap > self.graph_time_tolerance:
            return 0.0
        return float(max(0.0, 1.0 - gap / max(self.graph_time_tolerance, 1e-6)))

    @staticmethod
    def _shape_score(left: Tracklet, right: Tracklet) -> float:
        if left.shape <= 0 or right.shape <= 0:
            return 0.5
        ratio = min(float(left.shape), float(right.shape)) / max(float(left.shape), float(right.shape))
        return float(max(0.0, min(1.0, ratio)))

    @staticmethod
    def _normalise(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        return value / (np.linalg.norm(value) + 1e-12)

    @classmethod
    def colour_similarity(cls, left: Tracklet, right: Tracklet) -> float:
        a = getattr(left, "colour_signature", None)
        b = getattr(right, "colour_signature", None)
        if a is None or b is None:
            return 0.5
        return float(np.clip(np.dot(cls._normalise(a), cls._normalise(b)), 0.0, 1.0))

    def _pair_score(self, left: Tracklet, right: Tracklet) -> Tuple[float, float, float, float, float]:
        reid, _, _ = self.body_score(left.features, right.features)
        colour = self.colour_similarity(left, right)
        time = self._time_score(left, right)
        shape = self._shape_score(left, right)
        total = (
            self.graph_reid_weight * reid
            + self.graph_color_weight * colour
            + self.graph_time_weight * time
            + self.graph_shape_weight * shape
        )
        return float(total), float(reid), float(colour), float(time), float(shape)

    def _group_score(self, left: List[Tracklet], right: List[Tracklet]) -> dict:
        rows = []
        for a in left:
            for b in right:
                rows.append((*self._pair_score(a, b), a.key, b.key))
        if not rows:
            return {"score": 0.0, "reid": 0.0, "colour": 0.5, "time": 0.0, "shape": 0.5, "pair": None}
        rows.sort(key=lambda x: x[0], reverse=True)
        top = rows[: min(3, len(rows))]
        return {
            "score": float(0.72 * top[0][0] + 0.28 * np.mean([x[0] for x in top])),
            "reid": float(top[0][1]),
            "colour": float(np.mean([x[2] for x in top])),
            "time": float(max(x[3] for x in top)),
            "shape": float(np.mean([x[4] for x in top])),
            "pair": (top[0][5], top[0][6]),
        }

    @staticmethod
    def _hungarian(matrix: np.ndarray) -> List[Tuple[int, int]]:
        if matrix.size == 0:
            return []
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = matrix.shape
            size = max(rows, cols)
            padded = np.zeros((size, size), dtype=np.float64)
            padded[:rows, :cols] = matrix
            rr, cc = linear_sum_assignment(-padded)
            return [(int(r), int(c)) for r, c in zip(rr, cc) if r < rows and c < cols and matrix[r, c] > 0.0]
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

    def _camera_groups(self, mapping: Dict[str, str], tracks: Dict[str, Tracklet], camera: str) -> Dict[str, List[Tracklet]]:
        groups: Dict[str, List[Tracklet]] = {}
        for key, gid in mapping.items():
            if tracks[key].camera == camera:
                groups.setdefault(gid, []).append(tracks[key])
        return groups

    def _pair_camera_links(self, mapping: Dict[str, str], tracks: Dict[str, Tracklet], camera_a: str, camera_b: str) -> List[dict]:
        left = self._camera_groups(mapping, tracks, camera_a)
        right = self._camera_groups(mapping, tracks, camera_b)
        if not left or not right:
            return []
        gids_a, gids_b = sorted(left), sorted(right)
        matrix = np.zeros((len(gids_a), len(gids_b)), dtype=np.float64)
        details: Dict[Tuple[int, int], dict] = {}
        for i, gid_a in enumerate(gids_a):
            for j, gid_b in enumerate(gids_b):
                item = self._group_score(left[gid_a], right[gid_b])
                details[(i, j)] = item
                if item["reid"] < self.graph_min_reid or item["score"] < self.graph_min_score:
                    continue
                if item["reid"] < self.graph_strong_reid and item["colour"] < self.graph_color_gate:
                    continue
                matrix[i, j] = item["score"]

        out: List[dict] = []
        for i, j in self._hungarian(matrix):
            item = details[(i, j)]
            row_values = np.sort(matrix[i, :][matrix[i, :] > 0.0])[::-1]
            col_values = np.sort(matrix[:, j][matrix[:, j] > 0.0])[::-1]
            row_margin = float(row_values[0] - row_values[1]) if len(row_values) > 1 else float("inf")
            col_margin = float(col_values[0] - col_values[1]) if len(col_values) > 1 else float("inf")
            if row_margin < self.graph_margin or col_margin < self.graph_margin:
                continue
            out.append({
                "camera_a": camera_a, "camera_b": camera_b,
                "gid_a": gids_a[i], "gid_b": gids_b[j],
                "score": float(item["score"]), "reid": float(item["reid"]),
                "colour": float(item["colour"]), "time": float(item["time"]),
                "shape": float(item["shape"]), "row_margin": row_margin,
                "col_margin": col_margin, "pair": item["pair"],
            })
        return out

    @staticmethod
    def _components(edges: List[dict], nodes: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
        parent = {node: node for node in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def camera_set(root):
            return {cam for cam, gid in nodes if find((cam, gid)) == root}

        for edge in sorted(edges, key=lambda x: x["score"], reverse=True):
            a = (edge["camera_a"], edge["gid_a"])
            b = (edge["camera_b"], edge["gid_b"])
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            if camera_set(ra).isdisjoint(camera_set(rb)):
                parent[rb] = ra

        groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        for node in nodes:
            groups.setdefault(find(node), []).append(node)
        return list(groups.values())

    def _reconcile(self, mapping: Dict[str, str], tracks: Dict[str, Tracklet]):
        cameras = sorted({track.camera for track in tracks.values()})
        nodes = []
        for camera in cameras:
            gids = sorted({mapping[k] for k, t in tracks.items() if t.camera == camera and k in mapping})
            nodes.extend((camera, gid) for gid in gids)

        edges: List[dict] = []
        for i, camera_a in enumerate(cameras):
            for camera_b in cameras[i + 1:]:
                edges.extend(self._pair_camera_links(mapping, tracks, camera_a, camera_b))

        components = self._components(edges, nodes)
        used: set[str] = set()
        corrected = dict(mapping)
        ordered = sorted(
            components,
            key=lambda comp: min([int(g[1:]) for _c, g in comp if g.startswith("G") and g[1:].isdigit()] or [10**9]),
        )
        max_id = max([int(g[1:]) for _c, g in nodes if g.startswith("G") and g[1:].isdigit()] or [0])
        for component in ordered:
            candidates = sorted(
                {gid for _cam, gid in component},
                key=lambda g: int(g[1:]) if g.startswith("G") and g[1:].isdigit() else 10**9,
            )
            canonical = next((gid for gid in candidates if gid not in used), None)
            if canonical is None:
                max_id += 1
                canonical = f"G{max_id:06d}"
            used.add(canonical)
            nodeset = set(component)
            for key, gid in list(corrected.items()):
                track = tracks[key]
                if (track.camera, gid) in nodeset:
                    corrected[key] = canonical

        refreshed = []
        for item in self.decisions:
            gid = corrected.get(item.key, item.gid)
            reason = "cross_camera_graph" if gid != item.gid else item.reason
            refreshed.append(
                type(item)(item.key, gid, item.state, reason, item.score, item.margin, item.body,
                           item.temporal, item.spatial, item.support, item.camera,
                           item.provisional, item.merged)
            )
        self.decisions = refreshed
        self.graph_links = edges
        return corrected

    def _rebuild_state(self, mapping: Dict[str, str], tracks: Dict[str, Tracklet]) -> None:
        rebuilt: Dict[str, Identity] = {}
        for key, gid in mapping.items():
            track = tracks[key]
            identity = rebuilt.setdefault(gid, Identity(gid))
            identity.add(track, trusted=track.evidence() >= self.promote, bank=self.gallery, quality=self.promote)
        self.identities = rebuilt
        self.mapping = dict(mapping)

    def run(self, tracks: Dict[str, Tracklet]):
        base_mapping, _ = super().run(tracks)
        corrected = self._reconcile(base_mapping, tracks)
        self._rebuild_state(corrected, tracks)
        return corrected, list(self.decisions)

    def summary(self, tracks: Dict[str, Tracklet]) -> dict:
        result = super().summary(tracks)
        result["cross_graph_links"] = len(self.graph_links)
        return result
