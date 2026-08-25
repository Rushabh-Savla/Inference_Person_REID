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
                used_r.add(i)
                used_c.add(j)
                out.append((i, j))
            return out

    @staticmethod
    def _camera_groups(mapping: Dict[str, str], tracks: Dict[str, Tracklet], camera: str) -> Dict[str, List[Tracklet]]:
        groups: Dict[str, List[Tracklet]] = {}
        for key, gid in mapping.items():
            if tracks[key].camera == camera:
                groups.setdefault(gid, []).append(tracks[key])
        return groups

    def _track_matches(self, left: List[Tracklet], right: List[Tracklet]) -> List[dict]:
        """Match individual tracklets before any GID grouping."""
        if not left or not right:
            return []
        left = sorted(left, key=lambda item: item.key)
        right = sorted(right, key=lambda item: item.key)
        matrix = np.zeros((len(left), len(right)), dtype=np.float64)
        details: Dict[Tuple[int, int], dict] = {}
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                total, reid, colour, time, shape = self._pair_score(a, b)
                details[(i, j)] = {
                    "score": total,
                    "reid": reid,
                    "colour": colour,
                    "time": time,
                    "shape": shape,
                    "left": a.key,
                    "right": b.key,
                }
                if reid < self.graph_min_reid or total < self.graph_min_score:
                    continue
                if reid < self.graph_strong_reid and colour < self.graph_color_gate:
                    continue
                matrix[i, j] = total

        out: List[dict] = []
        for i, j in self._hungarian(matrix):
            item = details[(i, j)]
            row_values = np.sort(matrix[i, :][matrix[i, :] > 0.0])[::-1]
            col_values = np.sort(matrix[:, j][matrix[:, j] > 0.0])[::-1]
            row_margin = float(row_values[0] - row_values[1]) if len(row_values) > 1 else float("inf")
            col_margin = float(col_values[0] - col_values[1]) if len(col_values) > 1 else float("inf")
            if row_margin < self.graph_margin or col_margin < self.graph_margin:
                continue
            out.append({**item, "row_margin": row_margin, "col_margin": col_margin})
        return out

    def _pair_camera_links(self, mapping: Dict[str, str], tracks: Dict[str, Tracklet], camera_a: str, camera_b: str) -> List[dict]:
        """Build one-to-one cross-camera links from track-level evidence."""
        left_groups = self._camera_groups(mapping, tracks, camera_a)
        right_groups = self._camera_groups(mapping, tracks, camera_b)
        if not left_groups or not right_groups:
            return []

        left_tracks = [track for group in left_groups.values() for track in group]
        right_tracks = [track for group in right_groups.values() for track in group]
        track_matches = self._track_matches(left_tracks, right_tracks)
        if not track_matches:
            return []

        out: List[dict] = []
        for match in track_matches:
            out.append(
                {
                    "camera_a": camera_a,
                    "camera_b": camera_b,
                    "left": match["left"],
                    "right": match["right"],
                    "gid_a": mapping[match["left"]],
                    "gid_b": mapping[match["right"]],
                    "score": float(match["score"]),
                    "reid": float(match["reid"]),
                    "colour": float(match["colour"]),
                    "time": float(match["time"]),
                    "shape": float(match["shape"]),
                    "row_margin": float(match["row_margin"]),
                    "col_margin": float(match["col_margin"]),
                    "pair": (match["left"], match["right"]),
                }
            )
        return out

    @staticmethod
    def _track_components(edges: List[dict], tracks: Dict[str, Tracklet]) -> List[List[str]]:
        """Build identity components directly from cross-camera track matches."""
        keys = sorted(tracks)
        parent = {key: key for key in keys}
        members: Dict[str, set[str]] = {key: {key} for key in keys}

        def find(x: str) -> str:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != x:
                nxt = parent[x]
                parent[x] = root
                x = nxt
            return root

        def cameras(root: str) -> set[str]:
            return {tracks[key].camera for key in members[root]}

        for edge in sorted(edges, key=lambda item: item["score"], reverse=True):
            left = edge["left"]
            right = edge["right"]
            rl, rr = find(left), find(right)
            if rl == rr:
                continue
            if not cameras(rl).isdisjoint(cameras(rr)):
                continue
            parent[rr] = rl
            members[rl].update(members[rr])
            del members[rr]

        groups: Dict[str, List[str]] = {}
        for key in keys:
            groups.setdefault(find(key), []).append(key)
        return list(groups.values())

    @staticmethod
    def _components(edges: List[dict], nodes: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
        """Compatibility helper for existing tests/debugging."""
        parent = {node: node for node in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def camera_set(root):
            return {cam for cam, _gid in nodes if find((cam, _gid)) == root}

        for edge in sorted(edges, key=lambda x: x["score"], reverse=True):
            a = (edge["camera_a"], edge["gid_a"])
            b = (edge["camera_b"], edge["gid_b"])
            if a not in parent or b not in parent:
                continue
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
        """Correct cross-camera GIDs from track correspondences, not GID groups.

        Each connected track component represents one physical person. The GID
        anchor is taken from the earliest camera in the component's configured
        camera order, so a permutation in a later camera is corrected directly
        instead of merging the two existing GID labels together.
        """
        cameras = sorted({track.camera for track in tracks.values()})
        edges: List[dict] = []
        for i, camera_a in enumerate(cameras):
            for camera_b in cameras[i + 1:]:
                edges.extend(self._pair_camera_links(mapping, tracks, camera_a, camera_b))

        components = self._track_components(edges, tracks)
        corrected = dict(mapping)
        camera_rank = {camera: index for index, camera in enumerate(cameras)}

        for component in components:
            linked = [key for key in component if any(edge["left"] == key or edge["right"] == key for edge in edges)]
            if len(linked) < 2:
                continue
            anchor = min(
                linked,
                key=lambda key: (
                    camera_rank.get(tracks[key].camera, 10**9),
                    float(tracks[key].start),
                    key,
                ),
            )
            canonical = mapping[anchor]
            for key in linked:
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
