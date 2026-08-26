from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from rebuild.identity_body_v6 import GlobalIdentityBodyV6


@dataclass(frozen=True)
class PairEvidence:
    fused: float
    resnet: float
    swin: float
    solider: float
    colour: float
    shape: float
    temporal: float
    agreement: float
    support: int


class MultiModelLocalGlobalResolver:
    """Track-level multi-model cross-camera reconciliation.

    The proven V6 ResNet association remains the same-camera authority. Cross-camera
    matching is performed on individual tracklets, never on already-global IDs, so
    a bad local merge cannot contaminate another camera. NVIDIA Swin and SOLIDER are
    independent evidence spaces; all three must be healthy and at least two must
    support a link. Colour, body shape and soft time compatibility only break ties.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.resnet = GlobalIdentityBodyV6(cfg)
        self.fused_min = float(cfg.get("final_cross_fused_min", 0.69))
        self.strong = float(cfg.get("final_cross_strong", 0.72))
        self.margin = float(cfg.get("final_cross_margin", 0.045))
        self.conflict = float(cfg.get("final_cross_conflict", 0.32))
        self.time_tolerance = float(cfg.get("final_cross_time_tolerance_sec", 12.0))
        self.offsets = {str(k): float(v) for k, v in (cfg.get("camera_time_offsets_sec", {}) or {}).items()}
        self.default_weights = {
            "resnet": float(cfg.get("final_w_resnet", 0.28)),
            "swin": float(cfg.get("final_w_swin", 0.34)),
            "solider": float(cfg.get("final_w_solider", 0.30)),
            "colour": float(cfg.get("final_w_colour", 0.05)),
            "shape": float(cfg.get("final_w_shape", 0.03)),
        }
        self.pair_weights = cfg.get("final_pair_weights", {}) or {}

    @staticmethod
    def _unit(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, np.float32)
        return value / (np.linalg.norm(value) + 1e-12)

    @classmethod
    def _top_score(cls, left: Iterable[np.ndarray], right: Iterable[np.ndarray], k: int = 3) -> Tuple[float, int]:
        a = [cls._unit(x) for x in left if x is not None]
        b = [cls._unit(x) for x in right if x is not None]
        if not a or not b:
            return 0.0, 0
        matrix = np.asarray([[float(np.dot(x, y)) for y in b] for x in a], np.float32)
        flat = np.sort(matrix.reshape(-1))[::-1]
        take = flat[: min(k, len(flat))]
        return float(np.mean(take)), int((flat >= 0.60).sum())

    @staticmethod
    def _colour(left, right) -> float:
        a = getattr(left, "colour_signature", None)
        b = getattr(right, "colour_signature", None)
        if a is None or b is None:
            return 0.5
        return float(np.clip(np.dot(MultiModelLocalGlobalResolver._unit(a), MultiModelLocalGlobalResolver._unit(b)), 0.0, 1.0))

    @staticmethod
    def _shape(left, right) -> float:
        if left.shape <= 0 or right.shape <= 0:
            return 0.5
        return float(min(left.shape, right.shape) / max(left.shape, right.shape))

    def _time(self, left, right) -> float:
        a0 = float(left.start) + self.offsets.get(left.camera, 0.0)
        a1 = float(left.end) + self.offsets.get(left.camera, 0.0)
        b0 = float(right.start) + self.offsets.get(right.camera, 0.0)
        b1 = float(right.end) + self.offsets.get(right.camera, 0.0)
        if not (a1 < b0 or b1 < a0):
            return 1.0
        gap = min(abs(a1 - b0), abs(b1 - a0))
        return float(max(0.0, 1.0 - gap / max(self.time_tolerance, 1e-6)))

    def _weights(self, a: str, b: str) -> Dict[str, float]:
        key = "-".join(sorted((a, b)))
        data = dict(self.default_weights)
        data.update(self.pair_weights.get(key, {}) or {})
        total = sum(data.values())
        return {k: v / total for k, v in data.items()}

    def _pair(self, left, right) -> PairEvidence:
        lbank = getattr(left, "model_bank", {})
        rbank = getattr(right, "model_bank", {})
        resnet, rsupport = self.resnet.body_score(left.features, right.features) if left.features and right.features else (0.0, 0, False)
        swin, ssupport = self._top_score(lbank.get("swin", []), rbank.get("swin", []))
        solider, psupport = self._top_score(lbank.get("solider", []), rbank.get("solider", []))
        colour = self._colour(left, right)
        shape = self._shape(left, right)
        temporal = self._time(left, right)
        weights = self._weights(left.camera, right.camera)
        fused = (
            weights["resnet"] * resnet
            + weights["swin"] * swin
            + weights["solider"] * solider
            + weights["colour"] * colour
            + weights["shape"] * shape
        )
        agreement = float(min(resnet, swin, solider))
        support = int(rsupport + ssupport + psupport)
        return PairEvidence(float(fused), float(resnet), float(swin), float(solider), float(colour), float(shape), float(temporal), agreement, support)

    @staticmethod
    def _overlap(left, right) -> bool:
        return not (left.end < right.start or right.end < left.start)

    @staticmethod
    def _greedy_one_to_one(matrix: np.ndarray) -> List[Tuple[int, int]]:
        edges = sorted(
            ((float(matrix[i, j]), i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if matrix[i, j] > 0.0),
            reverse=True,
        )
        used_r, used_c, out = set(), set(), []
        for _, i, j in edges:
            if i in used_r or j in used_c:
                continue
            used_r.add(i); used_c.add(j); out.append((i, j))
        return out

    def _accepted(self, item: PairEvidence, row_margin: float, col_margin: float) -> bool:
        if item.fused < self.fused_min or row_margin < self.margin or col_margin < self.margin:
            return False
        scores = (item.resnet, item.swin, item.solider)
        strong = sum(x >= self.strong for x in scores)
        usable = sum(x >= 0.62 for x in scores)
        if usable < 2:
            return False
        if strong < 2 and item.agreement < 0.58:
            return False
        if min(scores) < self.conflict and strong < 3:
            return False
        return True

    def _cross_track_edges(self, left: List[object], right: List[object]) -> List[dict]:
        if not left or not right:
            return []
        matrix = np.zeros((len(left), len(right)), np.float32)
        detail: Dict[Tuple[int, int], PairEvidence] = {}
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                item = self._pair(a, b)
                detail[(i, j)] = item
                if item.fused >= self.fused_min:
                    matrix[i, j] = item.fused
        accepted: List[dict] = []
        for i, j in self._greedy_one_to_one(matrix):
            row = np.sort(matrix[i][matrix[i] > 0])[::-1]
            col = np.sort(matrix[:, j][matrix[:, j] > 0])[::-1]
            rm = float(row[0] - row[1]) if len(row) > 1 else 1.0
            cm = float(col[0] - col[1]) if len(col) > 1 else 1.0
            item = detail[(i, j)]
            if self._accepted(item, rm, cm):
                accepted.append({
                    "left": left[i].key,
                    "right": right[j].key,
                    "row_margin": rm,
                    "col_margin": cm,
                    "fused": item.fused,
                    "resnet": item.resnet,
                    "swin": item.swin,
                    "solider": item.solider,
                    "colour": item.colour,
                    "shape": item.shape,
                    "temporal": item.temporal,
                    "agreement": item.agreement,
                    "support": item.support,
                })
        return accepted

    def resolve(self, local_mapping: Dict[str, str], tracks: Dict[str, object], cameras: List[str]):
        # Local V6 labels are evidence only. Cross-camera matching is track-level.
        # This prevents a contaminated local identity from combining different
        # people before the multi-model matcher sees them.
        nodes = sorted(tracks.values(), key=lambda x: (x.start, x.camera, x.key))
        by_camera = {camera: [x for x in nodes if x.camera == camera] for camera in cameras}
        edges: List[dict] = []
        for idx, ca in enumerate(sorted(cameras)):
            for cb in sorted(cameras)[idx + 1:]:
                edges.extend(self._cross_track_edges(by_camera.get(ca, []), by_camera.get(cb, [])))

        parent = {x.key: x.key for x in nodes}
        members = {x.key: {x.key} for x in nodes}
        camera = {x.key: x.camera for x in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # First lock high-confidence cross-camera edges. Then connect same-camera
        # fragments only when they already share the protected local V6 identity
        # and do not overlap in time.
        for edge in sorted(edges, key=lambda x: x["fused"], reverse=True):
            a, b = find(edge["left"]), find(edge["right"])
            if a == b:
                continue
            if {camera[n] for n in members[a]} & {camera[n] for n in members[b]}:
                continue
            parent[b] = a
            members[a].update(members[b]); del members[b]

        by_local = defaultdict(list)
        for key, gid in local_mapping.items():
            if key in parent:
                by_local[(key.split(":", 1)[0], gid)].append(key)
        for (_, _gid), keys_local in by_local.items():
            for i, akey in enumerate(keys_local):
                for bkey in keys_local[i + 1:]:
                    a, b = find(akey), find(bkey)
                    if a == b or self._overlap(tracks[akey], tracks[bkey]):
                        continue
                    if {camera[n] for n in members[a]} & {camera[n] for n in members[b]}:
                        continue
                    # Same-camera local V6 continuity is allowed to reconnect a
                    # fragmented person, but never simultaneously visible tracks.
                    parent[b] = a
                    members[a].update(members[b]); del members[b]

        components = {}
        for node in nodes:
            components.setdefault(find(node.key), []).append(node.key)
        ordered = sorted(components.values(), key=lambda c: min((tracks[x].start, x) for x in c))
        global_mapping: Dict[str, str] = {}
        output_components: Dict[str, List[str]] = {}
        for index, component in enumerate(ordered, 1):
            gid = f"G{index:06d}"
            output_components[gid] = sorted(component)
            for key in component:
                global_mapping[key] = gid
        return global_mapping, output_components, edges
