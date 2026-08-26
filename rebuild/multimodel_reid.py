from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


@dataclass
class LocalGroup:
    key: str
    camera: str
    local_gid: str
    members: List[str] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    shape: float = 0.0
    features: List[object] = field(default_factory=list)
    model_bank: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    colour_signature: np.ndarray | None = None


class MultiModelLocalGlobalResolver:
    """Production MTMC reconciliation over camera-local V6 identities.

    V6 remains the same-camera authority. Its local mapping is converted into
    non-overlapping camera-local groups so sitting/standing/walking fragments keep
    one identity while accidentally overlapping local IDs are forcibly separated.
    Cross-camera matching is then performed on these groups, not on already-global
    IDs. Final GIDs come from a durable multimodel registry.
    """

    MODELS = ("resnet", "swin", "solider")

    def __init__(self, cfg: dict, registry: PersistentMultimodelRegistry | None = None):
        self.cfg = cfg
        self.resnet = GlobalIdentityBodyV6(cfg)
        self.registry = registry
        self.fused_min = float(cfg.get("final_cross_fused_min", 0.70))
        self.strong = float(cfg.get("final_cross_strong", 0.72))
        self.margin = float(cfg.get("final_cross_margin", 0.050))
        self.time_tolerance = float(cfg.get("final_cross_time_tolerance_sec", 10.0))
        self.offsets = {str(k): float(v) for k, v in (cfg.get("camera_time_offsets_sec", {}) or {}).items()}
        self.default_weights = {
            "resnet": float(cfg.get("final_w_resnet", 0.27)),
            "swin": float(cfg.get("final_w_swin", 0.34)),
            "solider": float(cfg.get("final_w_solider", 0.30)),
            "colour": float(cfg.get("final_w_colour", 0.04)),
            "shape": float(cfg.get("final_w_shape", 0.02)),
            "temporal": float(cfg.get("final_w_temporal", 0.03)),
        }
        self.pair_weights = cfg.get("final_pair_weights", {}) or {}
        self.model_min = {
            "resnet": float(cfg.get("final_cross_resnet_min", 0.55)),
            "swin": float(cfg.get("final_cross_swin_min", 0.55)),
            "solider": float(cfg.get("final_cross_solider_min", 0.50)),
        }
        self.gallery_min = float(cfg.get("final_gallery_match_min", 0.73))
        self.gallery_margin = float(cfg.get("final_gallery_margin", 0.050))
        self.max_gap_without_overlap = float(cfg.get("final_cross_max_gap_without_overlap_sec", 3.0))

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
        if evidence.temporal > 0.0:
            return True
        a0 = float(left.end) + self.offsets.get(left.camera, 0.0)
        b0 = float(right.start) + self.offsets.get(right.camera, 0.0)
        b1 = float(right.end) + self.offsets.get(right.camera, 0.0)
        a1 = float(left.start) + self.offsets.get(left.camera, 0.0)
        gap = min(abs(a0 - b1), abs(b0 - a1))
        return gap <= self.max_gap_without_overlap or max(evidence.resnet, evidence.swin, evidence.solider) >= 0.90

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

    @staticmethod
    def _build_local_groups(local_mapping: Mapping[str, str], tracks: Mapping[str, object], cameras: List[str]) -> List[LocalGroup]:
        grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for key, gid in local_mapping.items():
            if key in tracks:
                grouped[(tracks[key].camera, str(gid))].append(key)

        groups: List[LocalGroup] = []
        for (camera, gid), keys in grouped.items():
            lanes: List[Tuple[float, List[str]]] = []
            for key in sorted(keys, key=lambda k: (tracks[k].start, tracks[k].end, k)):
                start, end = float(tracks[key].start), float(tracks[key].end)
                placed = False
                for lane_index, (lane_end, members) in enumerate(lanes):
                    if start >= lane_end:
                        members.append(key)
                        lanes[lane_index] = (max(lane_end, end), members)
                        placed = True
                        break
                if not placed:
                    lanes.append((end, [key]))
            for lane_index, members in enumerate(lanes):
                feature_list: List[object] = []
                model_bank: Dict[str, List[np.ndarray]] = defaultdict(list)
                colours: List[np.ndarray] = []
                shapes: List[float] = []
                for key in members:
                    track = tracks[key]
                    feature_list.extend(track.features)
                    for model_name, values in getattr(track, "model_bank", {}).items():
                        model_bank[model_name].extend(np.asarray(v, np.float32) for v in values)
                    if getattr(track, "colour_signature", None) is not None:
                        colours.append(np.asarray(track.colour_signature, np.float32))
                    if float(getattr(track, "shape", 0.0)) > 0:
                        shapes.append(float(track.shape))
                colour = None
                if colours:
                    colour = np.mean(np.stack(colours), axis=0)
                    colour /= np.linalg.norm(colour) + 1e-12
                groups.append(
                    LocalGroup(
                        key=f"{camera}:{gid}:lane{lane_index}",
                        camera=camera,
                        local_gid=str(gid),
                        members=sorted(members),
                        start=min(float(tracks[k].start) for k in members),
                        end=max(float(tracks[k].end) for k in members),
                        shape=float(np.median(shapes)) if shapes else 0.0,
                        features=feature_list,
                        model_bank=dict(model_bank),
                        colour_signature=colour,
                    )
                )
        return sorted(groups, key=lambda g: (g.camera, g.start, g.key))

    def _cross_group_edges(self, left: List[LocalGroup], right: List[LocalGroup]) -> List[dict]:
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
        accepted = []
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
                    "left_members": left[i].members,
                    "right_members": right[j].members,
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
    def _component_models(component: Iterable[LocalGroup]) -> Dict[str, List[np.ndarray]]:
        banks: Dict[str, List[np.ndarray]] = defaultdict(list)
        for group in component:
            for feat in group.features:
                banks["resnet"].append(np.asarray(feat.value, np.float32))
            for model_name, values in group.model_bank.items():
                banks[model_name].extend(np.asarray(v, np.float32) for v in values)
        return banks

    def _gallery_score(self, component: List[LocalGroup], gallery: Mapping[int, Mapping[str, List[np.ndarray]]]) -> Dict[int, float]:
        current = self._component_models(component)
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

    def _assign_persistent_gids(self, components: List[List[LocalGroup]]) -> Dict[str, str]:
        if self.registry is None:
            return {}
        gallery = self.registry.load_gallery()
        used: set[int] = set()
        result: Dict[str, str] = {}
        for component in components:
            ranked = sorted(self._gallery_score(component, gallery).items(), key=lambda x: x[1], reverse=True)
            gid: int | None = None
            if ranked:
                best_gid, best_score = ranked[0]
                second = ranked[1][1] if len(ranked) > 1 else 0.0
                if best_score >= self.gallery_min and (best_score - second) >= self.gallery_margin and best_gid not in used:
                    gid = best_gid
            if gid is None:
                gid = self.registry.allocate_gid()
            used.add(gid)
            banks = self._component_models(component)
            cams = {group.camera for group in component}
            last_ts = max(group.end for group in component)
            obs = sum(len(group.features) for group in component)
            self.registry.save_component(gid, model_banks=banks, cameras=cams, last_ts=last_ts, obs=obs)
            gallery = self.registry.load_gallery()
            gid_text = f"G{gid:06d}"
            for group in component:
                for key in group.members:
                    result[key] = gid_text
        return result

    def resolve(self, local_mapping: Dict[str, str], tracks: Dict[str, object], cameras: List[str]):
        groups = self._build_local_groups(local_mapping, tracks, cameras)
        by_camera = {camera: [x for x in groups if x.camera == camera] for camera in cameras}
        edges: List[dict] = []
        ordered_cameras = sorted(cameras)
        for index, camera_a in enumerate(ordered_cameras):
            for camera_b in ordered_cameras[index + 1:]:
                edges.extend(self._cross_group_edges(by_camera.get(camera_a, []), by_camera.get(camera_b, [])))

        parent = {group.key: group.key for group in groups}
        members = {group.key: {group.key} for group in groups}
        group_by_key = {group.key: group for group in groups}

        def find(key: str) -> str:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for edge in sorted(edges, key=lambda x: x["fused"], reverse=True):
            a, b = find(edge["left"]), find(edge["right"])
            if a == b:
                continue
            cams_a = {group_by_key[node].camera for node in members[a]}
            cams_b = {group_by_key[node].camera for node in members[b]}
            if cams_a & cams_b:
                continue
            parent[b] = a
            members[a].update(members[b])
            del members[b]

        component_groups: Dict[str, List[LocalGroup]] = defaultdict(list)
        for group in groups:
            component_groups[find(group.key)].append(group)
        ordered_components = sorted(component_groups.values(), key=lambda comp: min((g.start, g.key) for g in comp))

        persistent = self._assign_persistent_gids(ordered_components)
        if persistent:
            global_mapping = persistent
        else:
            global_mapping = {}
            for index, component in enumerate(ordered_components, 1):
                gid = f"G{index:06d}"
                for group in component:
                    for key in group.members:
                        global_mapping[key] = gid

        output_components: Dict[str, List[str]] = {}
        for component in ordered_components:
            sample_key = component[0].members[0]
            gid = global_mapping[sample_key]
            output_components[gid] = sorted(key for group in component for key in group.members)
        return global_mapping, output_components, edges
