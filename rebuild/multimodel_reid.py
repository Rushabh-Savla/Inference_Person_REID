from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

from rebuild.identity_body_v6 import GlobalIdentityBodyV6
from live.persistent_multimodel import PersistentMultimodelRegistry


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
    """Production MTMC reconciliation over independent camera-local tracks.

    Cross-camera matching is built directly from track observations, never from
    already-corrupted global IDs. A valid edge requires temporal compatibility,
    two-model appearance support, reciprocal one-to-one preference and a margin.
    Persistent multimodel identity memory is optional for tests and mandatory in
    the final deployment pipeline.
    """

    MODELS = ("resnet", "swin", "solider")

    def __init__(self, cfg: dict, registry: PersistentMultimodelRegistry | None = None):
        self.cfg = cfg
        self.resnet = GlobalIdentityBodyV6(cfg)
        self.registry = registry
        self.fused_min = float(cfg.get("final_cross_fused_min", 0.69))
        self.strong = float(cfg.get("final_cross_strong", 0.72))
        self.margin = float(cfg.get("final_cross_margin", 0.045))
        self.time_tolerance = float(cfg.get("final_cross_time_tolerance_sec", 12.0))
        self.offsets = {str(k): float(v) for k, v in (cfg.get("camera_time_offsets_sec", {}) or {}).items()}
        self.default_weights = {
            "resnet": float(cfg.get("final_w_resnet", 0.28)),
            "swin": float(cfg.get("final_w_swin", 0.34)),
            "solider": float(cfg.get("final_w_solider", 0.30)),
            "colour": float(cfg.get("final_w_colour", 0.05)),
            "shape": float(cfg.get("final_w_shape", 0.03)),
            "temporal": float(cfg.get("final_w_temporal", 0.04)),
        }
        self.pair_weights = cfg.get("final_pair_weights", {}) or {}
        self.model_min = {
            "resnet": float(cfg.get("final_cross_resnet_min", 0.55)),
            "swin": float(cfg.get("final_cross_swin_min", 0.55)),
            "solider": float(cfg.get("final_cross_solider_min", 0.50)),
        }
        self.gallery_min = float(cfg.get("final_gallery_match_min", 0.72))
        self.gallery_margin = float(cfg.get("final_gallery_margin", 0.045))
        self.max_gap_without_overlap = float(cfg.get("final_cross_max_gap_without_overlap_sec", 4.0))

    @staticmethod
    def _unit(value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, np.float32).reshape(-1)
        return arr / (np.linalg.norm(arr) + 1e-12)

    @classmethod
    def _top_score(cls, left: Iterable[np.ndarray], right: Iterable[np.ndarray], k: int = 5) -> Tuple[float, int]:
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

    def _pair_allowed(self, left, right, evidence: PairEvidence) -> bool:
        temporal = evidence.temporal
        if temporal <= 0.0:
            a0 = float(left.end) + self.offsets.get(left.camera, 0.0)
            b0 = float(right.start) + self.offsets.get(right.camera, 0.0)
            b1 = float(right.end) + self.offsets.get(right.camera, 0.0)
            a1 = float(left.start) + self.offsets.get(left.camera, 0.0)
            gap = min(abs(a0 - b1), abs(b0 - a1))
            if gap > self.max_gap_without_overlap and max(evidence.resnet, evidence.swin, evidence.solider) < 0.90:
                return False
        return True

    def _weights(self, a: str, b: str) -> Dict[str, float]:
        key = "-".join(sorted((a, b)))
        data = dict(self.default_weights)
        data.update(self.pair_weights.get(key, {}) or {})
        total = sum(data.values())
        return {k: v / total for k, v in data.items()}

    def _pair(self, left, right) -> PairEvidence:
        lbank = getattr(left, "model_bank", {})
        rbank = getattr(right, "model_bank", {})
        resnet, rsupport, _ = self.resnet.body_score(left.features, right.features) if left.features and right.features else (0.0, 0, False)
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
            + weights["temporal"] * temporal
        )
        scores = sorted((resnet, swin, solider), reverse=True)
        agreement = float((scores[0] + scores[1]) * 0.5)
        support = int(rsupport + ssupport + psupport)
        return PairEvidence(float(fused), float(resnet), float(swin), float(solider), float(colour), float(shape), float(temporal), agreement, support)

    def _accepted(self, item: PairEvidence, row_margin: float, col_margin: float) -> bool:
        if item.fused < self.fused_min or row_margin < self.margin or col_margin < self.margin:
            return False
        scores = (item.resnet, item.swin, item.solider)
        usable = sum(x >= self.model_min[name] for x, name in zip(scores, self.MODELS))
        strong = sum(x >= self.strong for x in scores)
        if usable < 2:
            return False
        if strong == 0 and item.agreement < 0.68:
            return False
        if item.temporal < 0.35 and max(scores) < 0.88:
            return False
        return True

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
            used_r.add(i)
            used_c.add(j)
            out.append((i, j))
        return out

    def _cross_track_edges(self, left: List[object], right: List[object]) -> List[dict]:
        if not left or not right:
            return []
        matrix = np.zeros((len(left), len(right)), np.float32)
        detail: Dict[Tuple[int, int], PairEvidence] = {}
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                item = self._pair(a, b)
                detail[(i, j)] = item
                if self._pair_allowed(a, b, item) and item.fused >= self.fused_min:
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

    @staticmethod
    def _component_cameras(component: Iterable[str], tracks: Mapping[str, object]) -> set[str]:
        return {tracks[key].camera for key in component}

    @staticmethod
    def _component_models(component: Iterable[str], tracks: Mapping[str, object]) -> Dict[str, List[np.ndarray]]:
        banks: Dict[str, List[np.ndarray]] = defaultdict(list)
        for key in component:
            track = tracks[key]
            for feat in track.features:
                banks["resnet"].append(np.asarray(feat.value, np.float32))
            for model_name, values in getattr(track, "model_bank", {}).items():
                banks[model_name].extend(np.asarray(v, np.float32) for v in values)
        return banks

    def _gallery_score(self, component: Iterable[str], tracks: Mapping[str, object], gallery: Mapping[int, Mapping[str, List[np.ndarray]]]) -> Dict[int, float]:
        current = self._component_models(component, tracks)
        scores: Dict[int, float] = {}
        weights = {"resnet": 0.28, "swin": 0.34, "solider": 0.30}
        for gid, model_bank in gallery.items():
            vals = []
            for model in self.MODELS:
                left = current.get(model, [])
                right = model_bank.get(model, [])
                if not left or not right:
                    continue
                score, _ = self._top_score(left, right)
                vals.append((weights[model], score))
            if len(vals) >= 2:
                total = sum(w for w, _ in vals)
                scores[int(gid)] = sum(w * s for w, s in vals) / total
        return scores

    def _assign_persistent_gids(self, components: List[List[str]], tracks: Mapping[str, object]) -> Dict[str, str]:
        if self.registry is None:
            return {}
        gallery = self.registry.load_gallery()
        used: set[int] = set()
        result: Dict[str, str] = {}
        for component in components:
            ranked = sorted(self._gallery_score(component, tracks, gallery).items(), key=lambda x: x[1], reverse=True)
            gid: int | None = None
            if ranked:
                best_gid, best_score = ranked[0]
                second = ranked[1][1] if len(ranked) > 1 else 0.0
                if best_score >= self.gallery_min and (best_score - second) >= self.gallery_margin and best_gid not in used:
                    gid = best_gid
            if gid is None:
                gid = self.registry.allocate_gid()
            used.add(gid)
            banks = self._component_models(component, tracks)
            cams = self._component_cameras(component, tracks)
            last_ts = max(float(tracks[key].end) for key in component)
            obs = sum(len(tracks[key].features) for key in component)
            self.registry.save_component(gid, model_banks=banks, cameras=cams, last_ts=last_ts, obs=obs)
            gallery = self.registry.load_gallery()
            for key in component:
                result[key] = f"G{gid:06d}"
        return result

    def resolve(self, local_mapping: Dict[str, str], tracks: Dict[str, object], cameras: List[str]):
        nodes = sorted(tracks.values(), key=lambda x: (x.start, x.camera, x.key))
        by_camera = {camera: [x for x in nodes if x.camera == camera] for camera in cameras}
        edges: List[dict] = []
        ordered_cameras = sorted(cameras)
        for idx, ca in enumerate(ordered_cameras):
            for cb in ordered_cameras[idx + 1:]:
                edges.extend(self._cross_track_edges(by_camera.get(ca, []), by_camera.get(cb, [])))

        parent = {x.key: x.key for x in nodes}
        members = {x.key: {x.key} for x in nodes}
        camera = {x.key: x.camera for x in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for edge in sorted(edges, key=lambda x: x["fused"], reverse=True):
            a, b = find(edge["left"]), find(edge["right"])
            if a == b:
                continue
            if {camera[n] for n in members[a]} & {camera[n] for n in members[b]}:
                continue
            parent[b] = a
            members[a].update(members[b])
            del members[b]

        components = {}
        for node in nodes:
            components.setdefault(find(node.key), []).append(node.key)
        ordered = sorted(components.values(), key=lambda c: min((tracks[x].start, x) for x in c))

        persistent = self._assign_persistent_gids(ordered, tracks) if self.registry is not None else None
        if persistent:
            global_mapping = persistent
        else:
            global_mapping = {}
            for index, component in enumerate(ordered, 1):
                gid = f"G{index:06d}"
                for key in component:
                    global_mapping[key] = gid

        output_components = {global_mapping[component[0]]: sorted(component) for component in ordered}
        return global_mapping, output_components, edges
