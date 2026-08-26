from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from rebuild.identity_body_v6 import GlobalIdentityBodyV6
from rebuild.v6_local_global import LocalNode


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
    """Conservative camera-local -> global resolver using three ReID spaces.

    ResNet is the proven V6 baseline. NVIDIA Swin-Base and SOLIDER are independent
    cross-camera evidence sources. They are never concatenated: each model must
    agree independently before a cross-camera link can become global identity
    evidence. Clothing colour, body shape and soft temporal compatibility are
    tie-breakers only. All accepted camera-pair links are one-to-one.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.resnet = GlobalIdentityBodyV6(cfg)
        self.resnet_min = float(cfg.get("final_cross_resnet_min", 0.55))
        self.swin_min = float(cfg.get("final_cross_swin_min", 0.55))
        self.solider_min = float(cfg.get("final_cross_solider_min", 0.50))
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

    def _node_pair(self, left: LocalNode, right: LocalNode) -> dict:
        pairs = []
        for a in left.tracks:
            for b in right.tracks:
                ev = self._pair(a, b)
                pairs.append((ev, a.key, b.key))
        if not pairs:
            return {"score": 0.0}
        pairs.sort(key=lambda x: x[0].fused, reverse=True)
        top = pairs[0][0]
        supporting = [x[0] for x in pairs[1:4] if x[0].fused >= self.fused_min - 0.04]
        consensus = float(np.mean([x.fused for x in [top] + supporting])) if supporting else top.fused
        score = 0.78 * top.fused + 0.22 * consensus
        return {
            "score": float(score),
            "resnet": top.resnet,
            "swin": top.swin,
            "solider": top.solider,
            "colour": top.colour,
            "shape": top.shape,
            "temporal": top.temporal,
            "agreement": top.agreement,
            "support": top.support,
            "left_track": pairs[0][1],
            "right_track": pairs[0][2],
        }

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

    def _accepted(self, item: dict, row_margin: float, col_margin: float) -> bool:
        if item["score"] < self.fused_min or row_margin < self.margin or col_margin < self.margin:
            return False
        strong = sum(x >= self.strong for x in (item["resnet"], item["swin"], item["solider"]))
        usable = sum(x >= 0.62 for x in (item["resnet"], item["swin"], item["solider"]))
        if usable < 2:
            return False
        if strong < 2 and item["agreement"] < 0.58:
            return False
        if min(item["resnet"], item["swin"], item["solider"]) < self.conflict and strong < 3:
            return False
        return True

    def resolve(self, local_mapping: Dict[str, str], tracks: Dict[str, object], cameras: List[str]):
        # Split overlapping local IDs before building any cross-camera node.
        safe: Dict[str, str] = dict(local_mapping)
        grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for key, gid in safe.items():
            grouped[(tracks[key].camera, gid)].append(key)
        for (camera, gid), keys in grouped.items():
            if len(keys) < 2:
                continue
            overlaps = any(self._overlap(tracks[a], tracks[b]) for idx, a in enumerate(keys) for b in keys[idx + 1:])
            if overlaps:
                for idx, key in enumerate(sorted(keys), 1):
                    safe[key] = f"{gid}__split_{idx:03d}"

        nodes: Dict[str, LocalNode] = {}
        for key, gid in safe.items():
            track = tracks[key]
            nkey = f"{track.camera}::{gid}"
            nodes.setdefault(nkey, LocalNode(nkey, track.camera, gid)).tracks.append(track)
        keys = sorted(nodes)
        edges: List[dict] = []
        for idx, ca in enumerate(sorted(cameras)):
            for cb in sorted(cameras)[idx + 1:]:
                left = [nodes[k] for k in keys if nodes[k].camera == ca]
                right = [nodes[k] for k in keys if nodes[k].camera == cb]
                if not left or not right:
                    continue
                matrix = np.zeros((len(left), len(right)), np.float32)
                details = {}
                for i, a in enumerate(left):
                    for j, b in enumerate(right):
                        item = self._node_pair(a, b)
                        details[(i, j)] = item
                        if item.get("score", 0.0) >= self.fused_min:
                            matrix[i, j] = item["score"]
                for i, j in self._greedy_one_to_one(matrix):
                    row = np.sort(matrix[i][matrix[i] > 0])[::-1]
                    col = np.sort(matrix[:, j][matrix[:, j] > 0])[::-1]
                    rm = float(row[0] - row[1]) if len(row) > 1 else 1.0
                    cm = float(col[0] - col[1]) if len(col) > 1 else 1.0
                    item = details[(i, j)]
                    if self._accepted(item, rm, cm):
                        edges.append({"left": left[i].key, "right": right[j].key, "row_margin": rm, "col_margin": cm, **item})

        parent = {k: k for k in keys}
        cams = {k: nodes[k].camera for k in keys}
        members = {k: {k} for k in keys}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for edge in sorted(edges, key=lambda x: x["score"], reverse=True):
            a, b = find(edge["left"]), find(edge["right"])
            if a == b:
                continue
            if {cams[n] for n in members[a]} & {cams[n] for n in members[b]}:
                continue
            parent[b] = a
            members[a].update(members[b]); del members[b]

        components = {}
        for key in keys:
            components.setdefault(find(key), []).append(key)
        ordered = sorted(components.values(), key=lambda c: min((nodes[x].start, x) for x in c))
        global_mapping = {}
        output_components = {}
        for index, component in enumerate(ordered, 1):
            gid = f"G{index:06d}"
            output_components[gid] = sorted(component)
            for node_key in component:
                for track in nodes[node_key].tracks:
                    global_mapping[track.key] = gid
        return global_mapping, output_components, edges
