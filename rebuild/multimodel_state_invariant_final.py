from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

from live.persistent_multimodel import PersistentMultimodelRegistry


@dataclass(frozen=True)
class PairEvidence:
    fused: float
    resnet: float
    swin: float
    solider: float
    colour: float
    geometry: float
    temporal: float
    continuity: float
    agreement: float
    model_support: int
    mutual_models: int
    view_support: int
    state_transition: bool


@dataclass
class LocalGroup:
    key: str
    camera: str
    local_gid: str
    members: List[str] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    aspect: float = 0.0
    state_type: str = "unknown"
    state_bank: Dict[str, Dict[str, List[np.ndarray]]] = field(default_factory=dict)
    colour_signature: np.ndarray | None = None
    start_center: Tuple[float, float] | None = None
    end_center: Tuple[float, float] | None = None
    end_height: float = 1.0


class StateInvariantFinalResolver:
    """State-invariant MTMC resolver.

    Local tracker/GID output is evidence, not identity truth. Tracklets are first
    repaired within each camera using temporal/spatial continuity plus agreement
    across ResNet, NVIDIA Swin and SOLIDER. Only then are cross-camera links built.
    """

    MODELS = ("resnet", "swin", "solider")
    VIEWS = ("full", "upper", "torso", "lower")
    TRANSITION_VIEWS = (
        ("full", "upper"), ("upper", "full"),
        ("full", "torso"), ("torso", "full"),
        ("upper", "torso"), ("torso", "upper"),
        ("full", "lower"), ("lower", "full"),
        ("torso", "lower"), ("lower", "torso"),
    )

    def __init__(self, cfg: Mapping[str, object], registry: PersistentMultimodelRegistry | None = None):
        self.cfg = cfg
        self.registry = registry
        self.cross_min = float(cfg.get("state_cross_fused_min", 0.58))
        self.partial_min = float(cfg.get("state_cross_partial_fused_min", 0.54))
        self.cross_margin = float(cfg.get("state_cross_margin", 0.025))
        self.partial_margin = float(cfg.get("state_cross_partial_margin", 0.02))
        self.same_min = float(cfg.get("state_same_fused_min", 0.56))
        self.same_margin = float(cfg.get("state_same_margin", 0.02))
        self.strong = float(cfg.get("state_cross_strong", 0.76))
        self.model_min = {
            "resnet": float(cfg.get("state_cross_resnet_min", 0.48)),
            "swin": float(cfg.get("state_cross_swin_min", 0.48)),
            "solider": float(cfg.get("state_cross_solider_min", 0.46)),
        }
        self.same_max_gap = float(cfg.get("state_same_max_gap_sec", 30.0))
        self.cross_max_gap = float(cfg.get("state_cross_max_gap_sec", 30.0))
        self.same_spatial_min = float(cfg.get("state_same_spatial_min", 0.04))
        self.gallery_min = float(cfg.get("state_gallery_match_min", 0.66))
        self.gallery_margin = float(cfg.get("state_gallery_margin", 0.035))
        self.offsets = {str(k): float(v) for k, v in (cfg.get("camera_time_offsets_sec", {}) or {}).items()}

    @staticmethod
    def _unit(value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if arr.size == 0 or not np.isfinite(norm) or norm <= 0:
            raise ValueError("invalid embedding")
        return arr / norm

    @classmethod
    def _matrix(cls, left: Iterable[np.ndarray], right: Iterable[np.ndarray]) -> np.ndarray:
        a = [cls._unit(x) for x in left if x is not None]
        b = [cls._unit(x) for x in right if x is not None]
        if not a or not b:
            return np.empty((0, 0), np.float32)
        return np.asarray([[float(np.dot(x, y)) for y in b] for x in a], dtype=np.float32)

    @classmethod
    def _topmean(cls, matrix: np.ndarray, k: int = 5) -> float:
        if matrix.size == 0:
            return 0.0
        values = np.sort(matrix.reshape(-1))[::-1]
        return float(np.mean(values[:min(k, values.size)]))

    @classmethod
    def _best_state_score(cls, left: Mapping[str, List[np.ndarray]], right: Mapping[str, List[np.ndarray]]) -> Tuple[float, int, bool]:
        direct: List[float] = []
        transition: List[float] = []
        support = 0
        for view in cls.VIEWS:
            if left.get(view) and right.get(view):
                matrix = cls._matrix(left[view], right[view])
                if matrix.size:
                    direct.append(cls._topmean(matrix))
                    support += int(np.sum(matrix >= 0.55))
        for first, second in cls.TRANSITION_VIEWS:
            if left.get(first) and right.get(second):
                matrix = cls._matrix(left[first], right[second])
                if matrix.size:
                    transition.append(cls._topmean(matrix))
                    support += int(np.sum(matrix >= 0.55))
        base = float(np.mean(sorted(direct, reverse=True)[:2])) if direct else 0.0
        trans = max(transition) if transition else 0.0
        if direct and transition:
            value = max(base, 0.80 * trans + 0.20 * base)
        elif transition:
            value = trans
        else:
            value = base
        return float(value), int(support), bool(transition and trans >= base + 0.01)

    @staticmethod
    def _colour(left: LocalGroup, right: LocalGroup) -> float:
        if left.colour_signature is None or right.colour_signature is None:
            return 0.5
        return float(np.clip(np.dot(StateInvariantFinalResolver._unit(left.colour_signature), StateInvariantFinalResolver._unit(right.colour_signature)), 0.0, 1.0))

    def _temporal(self, left: LocalGroup, right: LocalGroup) -> float:
        a0 = left.start + self.offsets.get(left.camera, 0.0)
        a1 = left.end + self.offsets.get(left.camera, 0.0)
        b0 = right.start + self.offsets.get(right.camera, 0.0)
        b1 = right.end + self.offsets.get(right.camera, 0.0)
        if not (a1 < b0 or b1 < a0):
            return 0.5
        gap = min(abs(a1 - b0), abs(b1 - a0))
        return float(max(0.0, 1.0 - gap / max(self.cross_max_gap, 1e-6)))

    @staticmethod
    def _distance(left: LocalGroup, right: LocalGroup) -> float:
        left_end = getattr(left, "end_center", None)
        right_start = getattr(right, "start_center", None)
        if left_end is None or right_start is None:
            return 0.5
        ax, ay = left_end
        bx, by = right_start
        scale = max(float(getattr(left, "end_height", 1.0)), 1.0)
        distance = float(np.hypot(ax - bx, ay - by) / scale)
        return float(max(0.0, 1.0 - distance / 8.0))

    @staticmethod
    def _continuity(left: LocalGroup, right: LocalGroup) -> float:
        if left.camera != right.camera:
            return 0.5
        if StateInvariantFinalResolver._overlap(left, right):
            return 0.0
        gap = right.start - left.end if right.start >= left.end else left.start - right.end
        gap = max(0.0, gap)
        temporal = max(0.0, 1.0 - gap / 30.0)
        spatial = StateInvariantFinalResolver._distance(left, right)
        return float(0.65 * temporal + 0.35 * spatial)

    @staticmethod
    def _weights(camera_a: str, camera_b: str) -> Dict[str, float]:
        pair = "-".join(sorted((camera_a, camera_b)))
        if pair == "cam_222-cam_224":
            return {"resnet": 0.30, "swin": 0.35, "solider": 0.32, "colour": 0.01, "geometry": 0.01, "temporal": 0.01}
        return {"resnet": 0.31, "swin": 0.34, "solider": 0.32, "colour": 0.01, "geometry": 0.01, "temporal": 0.01}

    def _model_score(self, left: LocalGroup, right: LocalGroup, model: str) -> float:
        l = left.state_bank.get(model, {})
        r = right.state_bank.get(model, {})
        values: List[float] = []
        for view in self.VIEWS:
            if l.get(view) and r.get(view):
                matrix = self._matrix(l[view], r[view])
                if matrix.size:
                    values.append(self._topmean(matrix))
        for first, second in self.TRANSITION_VIEWS:
            if l.get(first) and r.get(second):
                matrix = self._matrix(l[first], r[second])
                if matrix.size:
                    values.append(self._topmean(matrix))
        return max(values) if values else 0.0

    @staticmethod
    def _overlap(left: LocalGroup, right: LocalGroup) -> bool:
        return not (left.end < right.start or right.end < left.start)

    def _mutual_models(self, left: LocalGroup, right: LocalGroup, left_pool: List[LocalGroup], right_pool: List[LocalGroup]) -> int:
        count = 0
        for model in self.MODELS:
            row_candidates = [c for c in right_pool if c.key != left.key and not (c.camera == left.camera and self._overlap(left, c))]
            col_candidates = [c for c in left_pool if c.key != right.key and not (c.camera == right.camera and self._overlap(right, c))]
            if not row_candidates:
                row_candidates = [right]
            if not col_candidates:
                col_candidates = [left]
            row_scores = [self._model_score(left, c, model) for c in row_candidates]
            col_scores = [self._model_score(c, right, model) for c in col_candidates]
            if not row_scores or not col_scores:
                continue
            row_idx = int(np.argmax(row_scores)); col_idx = int(np.argmax(col_scores)); minimum = self.model_min[model]
            if row_scores[row_idx] >= minimum and col_scores[col_idx] >= minimum and row_candidates[row_idx].key == right.key and col_candidates[col_idx].key == left.key:
                count += 1
        return count

    def pair(self, left: LocalGroup, right: LocalGroup, left_pool: List[LocalGroup], right_pool: List[LocalGroup]) -> PairEvidence:
        scores = {model: self._model_score(left, right, model) for model in self.MODELS}
        ordered = sorted(scores.values(), reverse=True)
        agreement = float(np.mean(ordered[:2]))
        support = sum(score >= self.model_min[model] for model, score in scores.items())
        mutual = self._mutual_models(left, right, left_pool, right_pool)
        colour = self._colour(left, right); geometry = self._distance(left, right); temporal = self._temporal(left, right); continuity = self._continuity(left, right)
        weights = self._weights(left.camera, right.camera)
        fused = (weights["resnet"] * scores["resnet"] + weights["swin"] * scores["swin"] + weights["solider"] * scores["solider"] + weights["colour"] * colour + weights["geometry"] * geometry + weights["temporal"] * temporal)
        state_change = left.state_type != "unknown" and right.state_type != "unknown" and left.state_type != right.state_type
        _, left_support, left_transition = self._best_state_score(left.state_bank.get("swin", {}), right.state_bank.get("swin", {}))
        _, right_support, right_transition = self._best_state_score(right.state_bank.get("swin", {}), left.state_bank.get("swin", {}))
        state_change = state_change or left_transition or right_transition
        return PairEvidence(float(fused), float(scores["resnet"]), float(scores["swin"]), float(scores["solider"]), float(colour), float(geometry), float(temporal), float(continuity), float(agreement), int(support), int(mutual), int(min(left_support, right_support)), bool(state_change))

    @classmethod
    def _state_type(cls, aspect: float) -> str:
        if aspect <= 0.0:
            return "unknown"
        return "upright" if aspect >= 1.35 else "compact"

    @staticmethod
    def _track_centres(track):
        rows = getattr(track, "observations", []) or []
        if not rows:
            return None, None, 1.0
        ordered = sorted(rows, key=lambda row: float(row.get("timestamp", 0.0)))
        def centre(row):
            box = row.get("bbox") or [0, 0, 0, 0]; x1, y1, x2, y2 = [float(v) for v in box]; return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        first = centre(ordered[0]); last = centre(ordered[-1]); box = ordered[-1].get("bbox") or [0, 0, 0, 0]; height = max(1.0, float(box[3]) - float(box[1]))
        return first, last, height

    @classmethod
    def _group_from_track(cls, key: str, track, local_gid: str) -> LocalGroup:
        first, last, height = cls._track_centres(track)
        return LocalGroup(key=key, camera=str(track.camera), local_gid=str(local_gid), members=[key], start=float(track.start), end=float(track.end), aspect=float(getattr(track, "shape", 0.0)), state_type=cls._state_type(float(getattr(track, "shape", 0.0))), state_bank={model: {view: list(values) for view, values in views.items() if view in cls.VIEWS} for model, views in getattr(track, "state_bank", {}).items()}, colour_signature=np.asarray(track.colour_signature, np.float32) if getattr(track, "colour_signature", None) is not None else None, start_center=first, end_center=last, end_height=height)

    @staticmethod
    def _merge_groups(left: LocalGroup, right: LocalGroup) -> LocalGroup:
        first, second = (left, right) if left.start <= right.start else (right, left)
        bank: Dict[str, Dict[str, List[np.ndarray]]] = {}
        for model in set(left.state_bank) | set(right.state_bank):
            bank[model] = {}
            for view in set(left.state_bank.get(model, {})) | set(right.state_bank.get(model, {})):
                values = list(left.state_bank.get(model, {}).get(view, [])); values.extend(right.state_bank.get(model, {}).get(view, [])); bank[model][view] = values[-64:]
        colours = []
        if left.colour_signature is not None: colours.append(np.asarray(left.colour_signature, np.float32))
        if right.colour_signature is not None: colours.append(np.asarray(right.colour_signature, np.float32))
        colour = None
        if colours:
            colour = np.mean(np.stack(colours), axis=0); colour /= np.linalg.norm(colour) + 1e-12
        shapes = [x for x in (left.aspect, right.aspect) if x > 0]; aspect = float(np.median(shapes)) if shapes else 0.0
        return LocalGroup(key=f"{first.key}+{second.key}", camera=first.camera, local_gid=first.local_gid, members=first.members + second.members, start=min(first.start, second.start), end=max(first.end, second.end), aspect=aspect, state_type=first.state_type if first.state_type == second.state_type else "mixed", state_bank=bank, colour_signature=colour, start_center=first.start_center, end_center=second.end_center, end_height=second.end_height)

    @classmethod
    def build_groups(cls, local_mapping: Mapping[str, str], tracks: Mapping[str, object]) -> List[LocalGroup]:
        groups = []
        for key, track in tracks.items():
            groups.append(cls._group_from_track(key, track, str(local_mapping.get(key, "UNKNOWN"))))
        return sorted(groups, key=lambda g: (g.camera, g.start, g.key))

    def _same_accept(self, evidence: PairEvidence, row_margin: float, col_margin: float) -> bool:
        if evidence.continuity < self.same_spatial_min:
            return False
        threshold = self.partial_min if evidence.state_transition else self.same_min
        margin = self.partial_margin if evidence.state_transition else self.same_margin
        if evidence.fused < threshold or row_margin < margin or col_margin < margin:
            return False
        if evidence.model_support < 2 or evidence.mutual_models < 2 or evidence.agreement < 0.55:
            return False
        return True

    def same_camera_edges(self, groups: List[LocalGroup]) -> List[dict]:
        if len(groups) < 2:
            return []
        ordered = sorted(groups, key=lambda g: (g.start, g.end, g.key)); matrix = {}; evidence = {}
        for i, left in enumerate(ordered):
            for j in range(i + 1, len(ordered)):
                right = ordered[j]
                if right.start <= left.end:
                    continue
                if right.start - left.end > self.same_max_gap:
                    break
                item = self.pair(left, right, ordered, ordered); threshold = self.partial_min if item.state_transition else self.same_min
                if item.fused >= threshold:
                    matrix[(i, j)] = item.fused; evidence[(i, j)] = item
        by_left: Dict[int, List[Tuple[int, float]]] = defaultdict(list); by_right: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for key, score in matrix.items(): by_left[key[0]].append((key[1], score)); by_right[key[1]].append((key[0], score))
        edges = []
        for (i, j), _score in sorted(matrix.items(), key=lambda item: item[1], reverse=True):
            row = sorted(by_left[i], key=lambda x: x[1], reverse=True); col = sorted(by_right[j], key=lambda x: x[1], reverse=True)
            row_margin = row[0][1] - row[1][1] if len(row) > 1 else 1.0; col_margin = col[0][1] - col[1][1] if len(col) > 1 else 1.0
            item = evidence[(i, j)]
            if not self._same_accept(item, float(row_margin), float(col_margin)): continue
            edges.append({"left": ordered[i].key, "right": ordered[j].key, "fused": item.fused, "resnet": item.resnet, "swin": item.swin, "solider": item.solider, "continuity": item.continuity, "state_transition": item.state_transition})
        return edges

    @staticmethod
    def _stitch(groups: List[LocalGroup], edges: List[dict]) -> List[LocalGroup]:
        if not groups:
            return []
        lookup = {group.key: group for group in groups}; parent = {group.key: group.key for group in groups}
        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]; key = parent[key]
            return key
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra
        for edge in sorted(edges, key=lambda item: item["fused"], reverse=True):
            if edge["left"] not in parent or edge["right"] not in parent: continue
            a, b = find(edge["left"]), find(edge["right"])
            if a == b or StateInvariantFinalResolver._overlap(lookup[a], lookup[b]): continue
            union(a, b)
        clusters = defaultdict(list)
        for group in groups: clusters[find(group.key)].append(group)
        output = []
        for cluster in clusters.values():
            merged = sorted(cluster, key=lambda item: item.start)[0]
            for group in sorted(cluster[1:], key=lambda item: item.start):
                if not StateInvariantFinalResolver._overlap(merged, group): merged = StateInvariantFinalResolver._merge_groups(merged, group)
            output.append(merged)
        return sorted(output, key=lambda g: (g.camera, g.start, g.key))

    def _cross_accept(self, evidence: PairEvidence, row_margin: float, col_margin: float) -> bool:
        threshold = self.partial_min if evidence.state_transition else self.cross_min; margin = self.partial_margin if evidence.state_transition else self.cross_margin
        if evidence.fused < threshold or row_margin < margin or col_margin < margin: return False
        if evidence.model_support < 2 or evidence.mutual_models < 2 or evidence.agreement < 0.55: return False
        strongest = max(evidence.resnet, evidence.swin, evidence.solider)
        if evidence.state_transition and evidence.view_support < 3 and strongest < self.strong: return False
        return True

    def cross_edges(self, left: List[LocalGroup], right: List[LocalGroup]) -> List[dict]:
        if not left or not right: return []
        matrix = {}; details = {}
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                item = self.pair(a, b, left, right); details[(i, j)] = item; threshold = self.partial_min if item.state_transition else self.cross_min
                if item.fused >= threshold: matrix[(i, j)] = item.fused
        by_left: Dict[int, List[Tuple[int, float]]] = defaultdict(list); by_right: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for key, score in matrix.items(): by_left[key[0]].append((key[1], score)); by_right[key[1]].append((key[0], score))
        result = []; used_left, used_right = set(), set()
        for (i, j), _ in sorted(matrix.items(), key=lambda item: item[1], reverse=True):
            if i in used_left or j in used_right: continue
            row = sorted(by_left[i], key=lambda x: x[1], reverse=True); col = sorted(by_right[j], key=lambda x: x[1], reverse=True)
            row_margin = row[0][1] - row[1][1] if len(row) > 1 else 1.0; col_margin = col[0][1] - col[1][1] if len(col) > 1 else 1.0
            item = details[(i, j)]
            if not self._cross_accept(item, float(row_margin), float(col_margin)): continue
            used_left.add(i); used_right.add(j)
            result.append({"left": left[i].key, "right": right[j].key, "left_members": list(left[i].members), "right_members": list(right[j].members), "fused": item.fused, "resnet": item.resnet, "swin": item.swin, "solider": item.solider, "colour": item.colour, "geometry": item.geometry, "temporal": item.temporal, "continuity": item.continuity, "agreement": item.agreement, "model_support": item.model_support, "mutual_models": item.mutual_models, "view_support": item.view_support, "state_transition": item.state_transition, "row_margin": float(row_margin), "col_margin": float(col_margin)})
        return result

    @staticmethod
    def _components(groups: List[LocalGroup], edges: List[dict]):
        parent = {group.key: group.key for group in groups}; members = {group.key: {group.key} for group in groups}; lookup = {group.key: group for group in groups}
        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]; key = parent[key]
            return key
        for edge in sorted(edges, key=lambda item: item["fused"], reverse=True):
            a_key, b_key = edge["left"], edge["right"]
            if a_key not in parent or b_key not in parent: continue
            a, b = find(a_key), find(b_key)
            if a == b: continue
            cameras_a = {lookup[node].camera for node in members[a]}; cameras_b = {lookup[node].camera for node in members[b]}
            if cameras_a & cameras_b: continue
            parent[b] = a; members[a].update(members[b]); members.pop(b, None)
        result = defaultdict(list)
        for group in groups: result[find(group.key)].append(group)
        return sorted(result.values(), key=lambda component: min((g.start, g.key) for g in component))

    @staticmethod
    def _flat_gallery(component: List[LocalGroup]):
        result = defaultdict(list)
        for group in component:
            for model, views in group.state_bank.items():
                for values in views.values(): result[model].extend(values)
        return result

    def _gallery_score(self, component, gallery):
        current = self._flat_gallery(component); weights = {"resnet": 0.30, "swin": 0.37, "solider": 0.33}; result = {}
        for gid, stored in gallery.items():
            scores = []
            for model in self.MODELS:
                if current.get(model) and stored.get(model):
                    matrix = self._matrix(current[model], stored[model])
                    if matrix.size: scores.append((weights[model], self._topmean(matrix)))
            if len(scores) >= 2:
                total = sum(weight for weight, _ in scores); result[int(gid)] = sum(weight * score for weight, score in scores) / total
        return result

    def _assign(self, components):
        if self.registry is None:
            result = {}
            for index, component in enumerate(components, 1):
                gid = f"G{index:06d}"
                for group in component:
                    for key in group.members: result[key] = gid
            return result
        gallery = self.registry.load_gallery(); result, used = {}, set()
        for component in components:
            ranked = sorted(self._gallery_score(component, gallery).items(), key=lambda item: item[1], reverse=True); gid = None
            if ranked:
                best, score = ranked[0]; second = ranked[1][1] if len(ranked) > 1 else 0.0
                if score >= self.gallery_min and score - second >= self.gallery_margin and best not in used: gid = int(best)
            if gid is None: gid = self.registry.allocate_gid()
            used.add(gid); self.registry.save_component(gid, model_banks=self._flat_gallery(component), cameras={group.camera for group in component}, last_ts=max(group.end for group in component), obs=len([key for group in component for key in group.members])); gallery = self.registry.load_gallery(); text = f"G{gid:06d}"
            for group in component:
                for key in group.members: result[key] = text
        return result

    def resolve(self, local_mapping: Mapping[str, str], tracks: Mapping[str, object], cameras: List[str]):
        raw = self.build_groups(local_mapping, tracks); by_camera = {camera: [group for group in raw if group.camera == camera] for camera in cameras}; stitched = {}; same_edges = []
        for camera in cameras:
            source = by_camera.get(camera, []); edges = self.same_camera_edges(source); same_edges.extend(edges); stitched[camera] = self._stitch(source, edges)
        repaired = [group for camera in cameras for group in stitched.get(camera, [])]; repaired_by_camera = {camera: [group for group in repaired if group.camera == camera] for camera in cameras}; cross_edges = []; ordered = list(cameras)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1:]: cross_edges.extend(self.cross_edges(repaired_by_camera.get(first, []), repaired_by_camera.get(second, [])))
        components = self._components(repaired, cross_edges); mapping = self._assign(components); component_output = {}
        for component in components:
            gid = mapping[component[0].members[0]]; component_output[gid] = sorted(key for group in component for key in group.members)
        return mapping, component_output, same_edges + cross_edges
