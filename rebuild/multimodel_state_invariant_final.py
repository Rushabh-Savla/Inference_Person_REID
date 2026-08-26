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
    """Final state-invariant MTMC resolver.

    The resolver deliberately does not trust a local GID. It repairs tracker
    fragmentation inside each camera, then performs global one-to-one matching
    using the three independent ReID spaces plus appearance and temporal evidence.
    The previous version over-gated candidates with reciprocal-best and top-two
    margins; that caused true matches to disappear when one view/model was weak.
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
        self.cross_min = float(cfg.get("state_cross_fused_min", 0.50))
        self.partial_min = float(cfg.get("state_cross_partial_fused_min", 0.47))
        self.cross_margin = float(cfg.get("state_cross_margin", 0.0))
        self.partial_margin = float(cfg.get("state_cross_partial_margin", 0.0))
        self.same_min = float(cfg.get("state_same_fused_min", 0.48))
        self.same_margin = float(cfg.get("state_same_margin", 0.0))
        self.strong = float(cfg.get("state_cross_strong", 0.80))
        self.model_min = {
            "resnet": float(cfg.get("state_cross_resnet_min", 0.44)),
            "swin": float(cfg.get("state_cross_swin_min", 0.44)),
            "solider": float(cfg.get("state_cross_solider_min", 0.42)),
        }
        self.same_model_min = float(cfg.get("state_same_model_min", 0.42))
        self.same_max_gap = float(cfg.get("state_same_max_gap_sec", 30.0))
        self.same_spatial_min = float(cfg.get("state_same_camera_continuity_min", 0.22))
        self.gallery_min = float(cfg.get("state_gallery_match_min", 0.62))
        self.gallery_margin = float(cfg.get("state_gallery_margin", 0.02))
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
        return float(np.mean(values[: min(k, values.size)]))

    @classmethod
    def _best_view_score(cls, left: Mapping[str, List[np.ndarray]], right: Mapping[str, List[np.ndarray]]) -> Tuple[float, int, bool]:
        direct: List[float] = []
        transition: List[float] = []
        support = 0
        for view in cls.VIEWS:
            if left.get(view) and right.get(view):
                matrix = cls._matrix(left[view], right[view])
                if matrix.size:
                    direct.append(cls._topmean(matrix))
                    support += int(np.sum(matrix >= 0.50))
        for first, second in cls.TRANSITION_VIEWS:
            if left.get(first) and right.get(second):
                matrix = cls._matrix(left[first], right[second])
                if matrix.size:
                    transition.append(cls._topmean(matrix))
                    support += int(np.sum(matrix >= 0.50))
        direct_best = max(direct) if direct else 0.0
        direct_top2 = float(np.mean(sorted(direct, reverse=True)[:2])) if direct else 0.0
        trans_best = max(transition) if transition else 0.0
        base = max(direct_best, direct_top2)
        if transition:
            value = max(base, 0.85 * trans_best + 0.15 * base)
        else:
            value = base
        return float(value), int(support), bool(transition and trans_best >= base + 0.01)

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
        return float(max(0.0, 1.0 - gap / 30.0))

    @staticmethod
    def _distance(left: LocalGroup, right: LocalGroup) -> float:
        if left.end_center is None or right.start_center is None:
            return 0.5
        ax, ay = left.end_center
        bx, by = right.start_center
        scale = max(float(left.end_height), 1.0)
        distance = float(np.hypot(ax - bx, ay - by) / scale)
        return float(max(0.0, 1.0 - distance / 8.0))

    @staticmethod
    def _overlap(left: LocalGroup, right: LocalGroup) -> bool:
        return not (left.end < right.start or right.end < left.start)

    def _model_score(self, left: LocalGroup, right: LocalGroup, model: str) -> float:
        l = left.state_bank.get(model, {})
        r = right.state_bank.get(model, {})
        score, _, _ = self._best_view_score(l, r)
        return score

    def _mutual_models(self, left: LocalGroup, right: LocalGroup, left_pool: List[LocalGroup], right_pool: List[LocalGroup]) -> int:
        count = 0
        for model in self.MODELS:
            right_scores = [(candidate, self._model_score(left, candidate, model)) for candidate in right_pool]
            left_scores = [(candidate, self._model_score(candidate, right, model)) for candidate in left_pool]
            if not right_scores or not left_scores:
                continue
            right_best = max(right_scores, key=lambda item: item[1])
            left_best = max(left_scores, key=lambda item: item[1])
            minimum = self.model_min[model]
            if right_best[0].key == right.key and left_best[0].key == left.key and right_best[1] >= minimum and left_best[1] >= minimum:
                count += 1
        return count

    def pair(self, left: LocalGroup, right: LocalGroup, left_pool: List[LocalGroup], right_pool: List[LocalGroup]) -> PairEvidence:
        scores = {model: self._model_score(left, right, model) for model in self.MODELS}
        ordered = sorted(scores.values(), reverse=True)
        agreement = float(np.mean(ordered[:2])) if ordered else 0.0
        support = sum(score >= self.model_min[model] for model, score in scores.items())
        mutual = self._mutual_models(left, right, left_pool, right_pool)
        colour = self._colour(left, right)
        geometry = self._distance(left, right)
        temporal = self._temporal(left, right)
        if left.camera == right.camera:
            continuity = 0.0 if self._overlap(left, right) else float(0.65 * max(0.0, 1.0 - min(abs(right.start-left.end), abs(left.start-right.end)) / 30.0) + 0.35 * geometry)
        else:
            continuity = 0.5
        state_left = left.state_type
        state_right = right.state_type
        state_change = state_left not in ("unknown", "mixed") and state_right not in ("unknown", "mixed") and state_left != state_right
        _, left_view_support, left_transition = self._best_view_score(left.state_bank.get("swin", {}), right.state_bank.get("swin", {}))
        _, right_view_support, right_transition = self._best_view_score(right.state_bank.get("swin", {}), left.state_bank.get("swin", {}))
        state_change = state_change or left_transition or right_transition
        deep_top2 = float(np.mean(ordered[:2])) if len(ordered) >= 2 else (ordered[0] if ordered else 0.0)
        strongest = ordered[0] if ordered else 0.0
        if left.camera == right.camera:
            fused = 0.78 * deep_top2 + 0.12 * colour + 0.10 * continuity
        else:
            fused = 0.70 * deep_top2 + 0.12 * strongest + 0.10 * colour + 0.05 * temporal + 0.03 * geometry
        return PairEvidence(float(fused), float(scores["resnet"]), float(scores["swin"]), float(scores["solider"]), float(colour), float(geometry), float(temporal), float(continuity), float(agreement), int(support), int(mutual), int(min(left_view_support, right_view_support)), bool(state_change))

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
            box = row.get("bbox") or [0, 0, 0, 0]
            x1, y1, x2, y2 = [float(v) for v in box]
            return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        first = centre(ordered[0]); last = centre(ordered[-1])
        box = ordered[-1].get("bbox") or [0, 0, 0, 0]
        height = max(1.0, float(box[3]) - float(box[1]))
        return first, last, height

    @classmethod
    def _group_from_track(cls, key: str, track, local_gid: str) -> LocalGroup:
        first, last, height = cls._track_centres(track)
        return LocalGroup(
            key=key,
            camera=str(track.camera),
            local_gid=str(local_gid),
            members=[key],
            start=float(track.start),
            end=float(track.end),
            aspect=float(getattr(track, "shape", 0.0)),
            state_type=cls._state_type(float(getattr(track, "shape", 0.0))),
            state_bank={model: {view: list(values) for view, values in views.items() if view in cls.VIEWS} for model, views in getattr(track, "state_bank", {}).items()},
            colour_signature=np.asarray(track.colour_signature, np.float32) if getattr(track, "colour_signature", None) is not None else None,
            start_center=first,
            end_center=last,
            end_height=height,
        )

    @staticmethod
    def _merge_groups(left: LocalGroup, right: LocalGroup) -> LocalGroup:
        first, second = (left, right) if left.start <= right.start else (right, left)
        bank: Dict[str, Dict[str, List[np.ndarray]]] = {}
        for model in set(left.state_bank) | set(right.state_bank):
            bank[model] = {}
            for view in set(left.state_bank.get(model, {})) | set(right.state_bank.get(model, {})):
                values = list(left.state_bank.get(model, {}).get(view, []))
                values.extend(right.state_bank.get(model, {}).get(view, []))
                bank[model][view] = values[-64:]
        colours = [x for x in (left.colour_signature, right.colour_signature) if x is not None]
        colour = None
        if colours:
            colour = np.mean(np.stack(colours), axis=0)
            colour /= np.linalg.norm(colour) + 1e-12
        shapes = [x for x in (left.aspect, right.aspect) if x > 0]
        aspect = float(np.median(shapes)) if shapes else 0.0
        return LocalGroup(
            key=f"{first.key}+{second.key}",
            camera=first.camera,
            local_gid=first.local_gid,
            members=first.members + second.members,
            start=min(first.start, second.start),
            end=max(first.end, second.end),
            aspect=aspect,
            state_type=first.state_type if first.state_type == second.state_type else "mixed",
            state_bank=bank,
            colour_signature=colour,
            start_center=first.start_center,
            end_center=second.end_center,
            end_height=second.end_height,
        )

    @classmethod
    def build_groups(cls, local_mapping: Mapping[str, str], tracks: Mapping[str, object]) -> List[LocalGroup]:
        return sorted((cls._group_from_track(key, track, str(local_mapping.get(key, "UNKNOWN"))) for key, track in tracks.items()), key=lambda group: (group.camera, group.start, group.key))

    def _same_accept(self, evidence: PairEvidence) -> bool:
        threshold = self.partial_min if evidence.state_transition else self.same_min
        if evidence.fused < threshold or evidence.continuity < self.same_spatial_min:
            return False
        if evidence.model_support >= 2 and evidence.agreement >= 0.48:
            return True
        strongest = max(evidence.resnet, evidence.swin, evidence.solider)
        return strongest >= 0.72 and evidence.colour >= 0.72

    def same_camera_edges(self, groups: List[LocalGroup]) -> List[dict]:
        ordered = sorted(groups, key=lambda group: (group.start, group.end, group.key))
        candidates = []
        for i, left in enumerate(ordered):
            for j in range(i + 1, len(ordered)):
                right = ordered[j]
                if right.start <= left.end:
                    continue
                if right.start - left.end > self.same_max_gap:
                    break
                evidence = self.pair(left, right, ordered, ordered)
                if self._same_accept(evidence):
                    candidates.append((evidence.fused, left, right, evidence))
        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = []
        groups_used: set[str] = set()
        for score, left, right, evidence in candidates:
            if left.key in groups_used or right.key in groups_used:
                continue
            groups_used.add(left.key)
            groups_used.add(right.key)
            chosen.append({"left": left.key, "right": right.key, "fused": score, "resnet": evidence.resnet, "swin": evidence.swin, "solider": evidence.solider, "colour": evidence.colour, "continuity": evidence.continuity, "state_transition": evidence.state_transition})
        return chosen

    @staticmethod
    def _stitch(groups: List[LocalGroup], edges: List[dict]) -> List[LocalGroup]:
        if not groups:
            return []
        lookup = {group.key: group for group in groups}
        parent = {group.key: group.key for group in groups}
        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for edge in sorted(edges, key=lambda item: item["fused"], reverse=True):
            a, b = edge["left"], edge["right"]
            if a not in parent or b not in parent:
                continue
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            if StateInvariantFinalResolver._overlap(lookup[ra], lookup[rb]):
                continue
            union(ra, rb)
        clusters = defaultdict(list)
        for group in groups:
            clusters[find(group.key)].append(group)
        result = []
        for cluster in clusters.values():
            merged = sorted(cluster, key=lambda item: item.start)[0]
            for group in sorted(cluster[1:], key=lambda item: item.start):
                if not StateInvariantFinalResolver._overlap(merged, group):
                    merged = StateInvariantFinalResolver._merge_groups(merged, group)
            result.append(merged)
        return sorted(result, key=lambda group: (group.camera, group.start, group.key))

    def _cross_accept(self, evidence: PairEvidence) -> bool:
        threshold = self.partial_min if evidence.state_transition else self.cross_min
        if evidence.fused < threshold:
            return False
        if evidence.model_support >= 2:
            return True
        ordered = sorted((evidence.resnet, evidence.swin, evidence.solider), reverse=True)
        if ordered[0] >= self.strong and ordered[1] >= 0.46 and evidence.colour >= 0.65:
            return True
        return False

    def cross_edges(self, left: List[LocalGroup], right: List[LocalGroup]) -> List[dict]:
        if not left or not right:
            return []
        candidates = []
        for a in left:
            for b in right:
                evidence = self.pair(a, b, left, right)
                if self._cross_accept(evidence):
                    candidates.append((evidence.fused, a, b, evidence))
        candidates.sort(key=lambda item: item[0], reverse=True)
        used_left: set[str] = set()
        used_right: set[str] = set()
        result = []
        for score, a, b, evidence in candidates:
            if a.key in used_left or b.key in used_right:
                continue
            used_left.add(a.key)
            used_right.add(b.key)
            result.append({
                "left": a.key,
                "right": b.key,
                "left_members": list(a.members),
                "right_members": list(b.members),
                "fused": float(score),
                "resnet": evidence.resnet,
                "swin": evidence.swin,
                "solider": evidence.solider,
                "colour": evidence.colour,
                "geometry": evidence.geometry,
                "temporal": evidence.temporal,
                "continuity": evidence.continuity,
                "agreement": evidence.agreement,
                "model_support": evidence.model_support,
                "mutual_models": evidence.mutual_models,
                "view_support": evidence.view_support,
                "state_transition": evidence.state_transition,
            })
        return result

    @staticmethod
    def _components(groups: List[LocalGroup], edges: List[dict]):
        parent = {group.key: group.key for group in groups}
        members = {group.key: {group.key} for group in groups}
        lookup = {group.key: group for group in groups}
        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key
        for edge in sorted(edges, key=lambda item: item["fused"], reverse=True):
            a_key, b_key = edge["left"], edge["right"]
            if a_key not in parent or b_key not in parent:
                continue
            a, b = find(a_key), find(b_key)
            if a == b:
                continue
            cameras_a = {lookup[node].camera for node in members[a]}
            cameras_b = {lookup[node].camera for node in members[b]}
            if cameras_a & cameras_b:
                continue
            parent[b] = a
            members[a].update(members[b])
            members.pop(b, None)
        result = defaultdict(list)
        for group in groups:
            result[find(group.key)].append(group)
        return sorted(result.values(), key=lambda component: min((g.start, g.key) for g in component))

    @staticmethod
    def _flat_gallery(component: List[LocalGroup]):
        result = defaultdict(list)
        for group in component:
            for model, views in group.state_bank.items():
                for values in views.values():
                    result[model].extend(values)
        return result

    def _gallery_score(self, component, gallery):
        current = self._flat_gallery(component)
        weights = {"resnet": 0.30, "swin": 0.37, "solider": 0.33}
        result = {}
        for gid, stored in gallery.items():
            scores = []
            for model in self.MODELS:
                if current.get(model) and stored.get(model):
                    matrix = self._matrix(current[model], stored[model])
                    if matrix.size:
                        scores.append((weights[model], self._topmean(matrix)))
            if len(scores) >= 2:
                total = sum(weight for weight, _ in scores)
                result[int(gid)] = sum(weight * score for weight, score in scores) / total
        return result

    def _assign(self, components):
        if self.registry is None:
            result = {}
            for index, component in enumerate(components, 1):
                gid = f"G{index:06d}"
                for group in component:
                    for key in group.members:
                        result[key] = gid
            return result
        gallery = self.registry.load_gallery()
        result, used = {}, set()
        for component in components:
            ranked = sorted(self._gallery_score(component, gallery).items(), key=lambda item: item[1], reverse=True)
            gid = None
            if ranked:
                best, score = ranked[0]
                second = ranked[1][1] if len(ranked) > 1 else 0.0
                if score >= self.gallery_min and score - second >= self.gallery_margin and best not in used:
                    gid = int(best)
            if gid is None:
                gid = self.registry.allocate_gid()
            used.add(gid)
            self.registry.save_component(
                gid,
                model_banks=self._flat_gallery(component),
                cameras={group.camera for group in component},
                last_ts=max(group.end for group in component),
                obs=len([key for group in component for key in group.members]),
            )
            gallery = self.registry.load_gallery()
            text = f"G{gid:06d}"
            for group in component:
                for key in group.members:
                    result[key] = text
        return result

    def resolve(self, local_mapping: Mapping[str, str], tracks: Mapping[str, object], cameras: List[str]):
        raw = self.build_groups(local_mapping, tracks)
        by_camera = {camera: [group for group in raw if group.camera == camera] for camera in cameras}
        stitched = {}
        same_edges: List[dict] = []
        for camera in cameras:
            source = by_camera.get(camera, [])
            edges = self.same_camera_edges(source)
            same_edges.extend(edges)
            stitched[camera] = self._stitch(source, edges)
        repaired = [group for camera in cameras for group in stitched.get(camera, [])]
        repaired_by_camera = {camera: [group for group in repaired if group.camera == camera] for camera in cameras}
        cross_edges: List[dict] = []
        for index, first in enumerate(cameras):
            for second in cameras[index + 1:]:
                cross_edges.extend(self.cross_edges(repaired_by_camera.get(first, []), repaired_by_camera.get(second, [])))
        components = self._components(repaired, cross_edges)
        mapping = self._assign(components)
        component_output = {}
        for component in components:
            gid = mapping[component[0].members[0]]
            component_output[gid] = sorted(key for group in component for key in group.members)
        return mapping, component_output, same_edges + cross_edges