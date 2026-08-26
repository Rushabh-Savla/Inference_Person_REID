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
    temporal: float
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


class StateInvariantFinalResolver:
    """Conservative cross-camera resolver with sitting/standing invariance.

    Identity evidence comes from three independent ReID spaces and multiple body
    views. Upper/torso evidence bridges seated and standing observations; full-body
    and lower-body evidence remain supporting views. A cross-camera edge must be
    supported by at least two model spaces and, normally, mutual-best agreement in
    those spaces. Same-camera uniqueness is a hard component constraint.
    """

    MODELS = ("resnet", "swin", "solider")
    VIEWS = ("full", "upper", "torso", "lower")
    TRANSITION_VIEWS = (
        ("full", "upper"), ("upper", "full"),
        ("full", "torso"), ("torso", "full"),
        ("upper", "torso"), ("torso", "upper"),
        ("full", "lower"), ("lower", "full"),
    )

    def __init__(self, cfg: Mapping[str, object], registry: PersistentMultimodelRegistry | None = None):
        self.cfg = cfg
        self.registry = registry
        self.fused_min = float(cfg.get("state_cross_fused_min", 0.66))
        self.partial_fused_min = float(cfg.get("state_cross_partial_fused_min", 0.62))
        self.strong = float(cfg.get("state_cross_strong", 0.70))
        self.margin = float(cfg.get("state_cross_margin", 0.045))
        self.partial_margin = float(cfg.get("state_cross_partial_margin", 0.035))
        self.model_min = {
            "resnet": float(cfg.get("state_cross_resnet_min", 0.52)),
            "swin": float(cfg.get("state_cross_swin_min", 0.52)),
            "solider": float(cfg.get("state_cross_solider_min", 0.50)),
        }
        self.max_gap = float(cfg.get("state_cross_max_gap_sec", 8.0))
        self.gallery_min = float(cfg.get("state_gallery_match_min", 0.70))
        self.gallery_margin = float(cfg.get("state_gallery_margin", 0.045))
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
        return np.asarray([[float(np.dot(x, y)) for y in b] for x in a], np.float32)

    @classmethod
    def _topmean_matrix(cls, matrix: np.ndarray, k: int = 5) -> float:
        if matrix.size == 0:
            return 0.0
        flat = np.sort(matrix.reshape(-1))[::-1]
        return float(np.mean(flat[: min(k, flat.size)]))

    @classmethod
    def _best_state_score(cls, left: Mapping[str, List[np.ndarray]], right: Mapping[str, List[np.ndarray]]) -> Tuple[float, int, bool]:
        direct_scores: List[float] = []
        transition_scores: List[float] = []
        support = 0
        for view in cls.VIEWS:
            if left.get(view) and right.get(view):
                matrix = cls._matrix(left[view], right[view])
                direct_scores.append(cls._topmean_matrix(matrix))
                support += int(np.sum(matrix >= 0.60))
        for a, b in cls.TRANSITION_VIEWS:
            if left.get(a) and right.get(b):
                matrix = cls._matrix(left[a], right[b])
                transition_scores.append(cls._topmean_matrix(matrix))
                support += int(np.sum(matrix >= 0.60))
        same = float(np.mean(sorted(direct_scores, reverse=True)[:2])) if direct_scores else 0.0
        transition = max(transition_scores) if transition_scores else 0.0
        if transition_scores and direct_scores:
            value = max(same, 0.75 * transition + 0.25 * same)
        else:
            value = transition if transition_scores else same
        return float(value), support, bool(transition_scores and transition > same + 0.01)

    @staticmethod
    def _colour(left: LocalGroup, right: LocalGroup) -> float:
        if left.colour_signature is None or right.colour_signature is None:
            return 0.5
        return float(np.clip(np.dot(left.colour_signature, right.colour_signature), 0.0, 1.0))

    def _temporal(self, left: LocalGroup, right: LocalGroup) -> float:
        a0 = left.start + self.offsets.get(left.camera, 0.0)
        a1 = left.end + self.offsets.get(left.camera, 0.0)
        b0 = right.start + self.offsets.get(right.camera, 0.0)
        b1 = right.end + self.offsets.get(right.camera, 0.0)
        if not (a1 < b0 or b1 < a0):
            return 0.5
        gap = min(abs(a1 - b0), abs(b1 - a0))
        return float(max(0.0, 1.0 - gap / max(self.max_gap, 1e-6)))

    @staticmethod
    def _weights(a: str, b: str) -> Dict[str, float]:
        pair = "-".join(sorted((a, b)))
        if pair == "cam_222-cam_224":
            raw = {"resnet": 0.29, "swin": 0.35, "solider": 0.31, "colour": 0.03, "temporal": 0.02}
        else:
            raw = {"resnet": 0.31, "swin": 0.34, "solider": 0.30, "colour": 0.03, "temporal": 0.02}
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    def _model_matrices(self, left: LocalGroup, right: LocalGroup, model: str) -> Tuple[Dict[Tuple[str, str], np.ndarray], List[str]]:
        l = left.state_bank.get(model, {})
        r = right.state_bank.get(model, {})
        result: Dict[Tuple[str, str], np.ndarray] = {}
        present: List[str] = []
        for view in self.VIEWS:
            if l.get(view) and r.get(view):
                result[(view, view)] = self._matrix(l[view], r[view])
                present.append(view)
        for a, b in self.TRANSITION_VIEWS:
            if l.get(a) and r.get(b):
                result[(a, b)] = self._matrix(l[a], r[b])
        return result, present

    def _model_score_for_pair(self, left: LocalGroup, right: LocalGroup, model: str) -> float:
        matrices, _ = self._model_matrices(left, right, model)
        values = [self._topmean_matrix(matrix) for matrix in matrices.values() if matrix.size]
        return max(values) if values else 0.0

    def _mutual_model_count(self, left: LocalGroup, right: LocalGroup, left_pool: List[LocalGroup], right_pool: List[LocalGroup]) -> int:
        count = 0
        for model in self.MODELS:
            scores_row = [self._model_score_for_pair(left, candidate, model) for candidate in right_pool]
            scores_col = [self._model_score_for_pair(candidate, right, model) for candidate in left_pool]
            if not scores_row or not scores_col:
                continue
            row_best = int(np.argmax(scores_row))
            col_best = int(np.argmax(scores_col))
            if right_pool[row_best].key == right.key and left_pool[col_best].key == left.key:
                count += 1
        return count

    def pair(self, left: LocalGroup, right: LocalGroup, left_pool: List[LocalGroup], right_pool: List[LocalGroup]) -> PairEvidence:
        scores: Dict[str, float] = {}
        view_support = 0
        transition_votes = 0
        for model in self.MODELS:
            score, support, transition = self._best_state_score(left.state_bank.get(model, {}), right.state_bank.get(model, {}))
            scores[model] = score
            view_support += min(support, 6)
            transition_votes += int(transition)
        ordered = sorted(scores.values(), reverse=True)
        agreement = float(np.mean(ordered[:2]))
        model_support = sum(score >= self.model_min[name] for name, score in scores.items())
        mutual = self._mutual_model_count(left, right, left_pool, right_pool)
        colour = self._colour(left, right)
        temporal = self._temporal(left, right)
        weights = self._weights(left.camera, right.camera)
        fused = (
            weights["resnet"] * scores["resnet"]
            + weights["swin"] * scores["swin"]
            + weights["solider"] * scores["solider"]
            + weights["colour"] * colour
            + weights["temporal"] * temporal
        )
        actual_state_change = (
            left.state_type != "unknown"
            and right.state_type != "unknown"
            and left.state_type != right.state_type
        )
        transition = actual_state_change or transition_votes >= 2
        return PairEvidence(
            fused=float(fused),
            resnet=float(scores["resnet"]),
            swin=float(scores["swin"]),
            solider=float(scores["solider"]),
            colour=float(colour),
            temporal=float(temporal),
            agreement=agreement,
            model_support=int(model_support),
            mutual_models=int(mutual),
            view_support=int(view_support),
            state_transition=bool(transition),
        )

    @classmethod
    def _state_type(cls, aspect: float) -> str:
        if aspect <= 0:
            return "unknown"
        return "upright" if aspect >= 1.35 else "compact"

    @classmethod
    def build_groups(cls, local_mapping: Mapping[str, str], tracks: Mapping[str, object]) -> List[LocalGroup]:
        grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for key, gid in local_mapping.items():
            if key in tracks:
                grouped[(tracks[key].camera, str(gid))].append(key)

        groups: List[LocalGroup] = []
        for (camera, gid), keys in grouped.items():
            lanes: List[Tuple[float, List[str]]] = []
            for key in sorted(keys, key=lambda k: (float(tracks[k].start), float(tracks[k].end), k)):
                start, end = float(tracks[key].start), float(tracks[key].end)
                for lane_index, (lane_end, members) in enumerate(lanes):
                    if start >= lane_end:
                        members.append(key)
                        lanes[lane_index] = (max(lane_end, end), members)
                        break
                else:
                    lanes.append((end, [key]))
            for lane_index, (_lane_end, members) in enumerate(lanes):
                state_bank: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
                colours: List[np.ndarray] = []
                aspects: List[float] = []
                for key in members:
                    track = tracks[key]
                    for model, views in getattr(track, "state_bank", {}).items():
                        for view, values in views.items():
                            state_bank[model][view].extend(np.asarray(v, np.float32) for v in values)
                    if getattr(track, "colour_signature", None) is not None:
                        colours.append(np.asarray(track.colour_signature, np.float32))
                    ratio = float(getattr(track, "shape", 0.0))
                    if ratio > 0:
                        aspects.append(ratio)
                colour = None
                if colours:
                    colour = np.mean(np.stack(colours), axis=0)
                    colour /= np.linalg.norm(colour) + 1e-12
                aspect = float(np.median(aspects)) if aspects else 0.0
                groups.append(LocalGroup(
                    key=f"{camera}:{gid}:lane{lane_index}",
                    camera=camera,
                    local_gid=str(gid),
                    members=sorted(members),
                    start=min(float(tracks[k].start) for k in members),
                    end=max(float(tracks[k].end) for k in members),
                    aspect=aspect,
                    state_type=cls._state_type(aspect),
                    state_bank={model: dict(views) for model, views in state_bank.items()},
                    colour_signature=colour,
                ))
        return sorted(groups, key=lambda g: (g.camera, g.start, g.key))

    def _acceptable(self, evidence: PairEvidence, row_margin: float, col_margin: float) -> bool:
        threshold = self.partial_fused_min if evidence.state_transition else self.fused_min
        margin = self.partial_margin if evidence.state_transition else self.margin
        if evidence.fused < threshold or row_margin < margin or col_margin < margin:
            return False
        if evidence.model_support < 2:
            return False
        if evidence.mutual_models < 2 and max(evidence.resnet, evidence.swin, evidence.solider) < 0.84:
            return False
        if evidence.agreement < 0.62:
            return False
        if evidence.state_transition and evidence.view_support < 4 and evidence.mutual_models < 3:
            return False
        if evidence.temporal < 0.20 and max(evidence.resnet, evidence.swin, evidence.solider) < 0.90:
            return False
        return True

    def cross_edges(self, left: List[LocalGroup], right: List[LocalGroup]) -> List[dict]:
        if not left or not right:
            return []
        matrix = np.zeros((len(left), len(right)), np.float32)
        details: Dict[Tuple[int, int], PairEvidence] = {}
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                item = self.pair(a, b, left, right)
                details[(i, j)] = item
                threshold = self.partial_fused_min if item.state_transition else self.fused_min
                if item.fused >= threshold:
                    matrix[i, j] = item.fused
        order = sorted(
            ((float(matrix[i, j]), i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if matrix[i, j] > 0),
            reverse=True,
        )
        used_left: set[int] = set()
        used_right: set[int] = set()
        result: List[dict] = []
        for _score, i, j in order:
            if i in used_left or j in used_right:
                continue
            row = np.sort(matrix[i][matrix[i] > 0])[::-1]
            col = np.sort(matrix[:, j][matrix[:, j] > 0])[::-1]
            row_margin = float(row[0] - row[1]) if row.size > 1 else 1.0
            col_margin = float(col[0] - col[1]) if col.size > 1 else 1.0
            evidence = details[(i, j)]
            if not self._acceptable(evidence, row_margin, col_margin):
                continue
            used_left.add(i)
            used_right.add(j)
            result.append({
                "left": left[i].key,
                "right": right[j].key,
                "left_members": left[i].members,
                "right_members": right[j].members,
                "fused": evidence.fused,
                "resnet": evidence.resnet,
                "swin": evidence.swin,
                "solider": evidence.solider,
                "colour": evidence.colour,
                "temporal": evidence.temporal,
                "agreement": evidence.agreement,
                "model_support": evidence.model_support,
                "mutual_models": evidence.mutual_models,
                "view_support": evidence.view_support,
                "state_transition": evidence.state_transition,
                "row_margin": row_margin,
                "col_margin": col_margin,
            })
        return result

    @staticmethod
    def _components(groups: List[LocalGroup], edges: List[dict]) -> List[List[LocalGroup]]:
        parent = {g.key: g.key for g in groups}
        members = {g.key: {g.key} for g in groups}
        lookup = {g.key: g for g in groups}

        def find(key: str) -> str:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for edge in sorted(edges, key=lambda item: item["fused"], reverse=True):
            a, b = find(edge["left"]), find(edge["right"])
            if a == b:
                continue
            if {lookup[node].camera for node in members[a]} & {lookup[node].camera for node in members[b]}:
                continue
            parent[b] = a
            members[a].update(members[b])
            del members[b]

        result: Dict[str, List[LocalGroup]] = defaultdict(list)
        for group in groups:
            result[find(group.key)].append(group)
        return sorted(result.values(), key=lambda component: min((g.start, g.key) for g in component))

    @staticmethod
    def _flat_gallery(component: List[LocalGroup]) -> Dict[str, List[np.ndarray]]:
        result: Dict[str, List[np.ndarray]] = defaultdict(list)
        for group in component:
            for model, views in group.state_bank.items():
                for values in views.values():
                    result[model].extend(values)
        return result

    def _gallery_score(self, component: List[LocalGroup], gallery: Mapping[int, Mapping[str, List[np.ndarray]]]) -> Dict[int, float]:
        current = self._flat_gallery(component)
        weights = {"resnet": 0.31, "swin": 0.35, "solider": 0.31}
        result: Dict[int, float] = {}
        for gid, stored in gallery.items():
            values = []
            for model in self.MODELS:
                if not current.get(model) or not stored.get(model):
                    continue
                matrix = self._matrix(current[model], stored[model])
                if matrix.size:
                    values.append((weights[model], self._topmean_matrix(matrix)))
            if len(values) >= 2:
                total = sum(weight for weight, _ in values)
                result[int(gid)] = sum(weight * score for weight, score in values) / total
        return result

    def _assign(self, components: List[List[LocalGroup]]) -> Dict[str, str]:
        if self.registry is None:
            output: Dict[str, str] = {}
            for index, component in enumerate(components, 1):
                gid = f"G{index:06d}"
                for group in component:
                    for key in group.members:
                        output[key] = gid
            return output
        gallery = self.registry.load_gallery()
        result: Dict[str, str] = {}
        used: set[int] = set()
        for component in components:
            ranked = sorted(self._gallery_score(component, gallery).items(), key=lambda item: item[1], reverse=True)
            gid: int | None = None
            if ranked:
                best, best_score = ranked[0]
                second = ranked[1][1] if len(ranked) > 1 else 0.0
                if best_score >= self.gallery_min and best_score - second >= self.gallery_margin and best not in used:
                    gid = int(best)
            if gid is None:
                gid = self.registry.allocate_gid()
            used.add(gid)
            banks = self._flat_gallery(component)
            self.registry.save_component(
                gid,
                model_banks=banks,
                cameras={group.camera for group in component},
                last_ts=max(group.end for group in component),
                obs=sum(len(group.members) for group in component),
            )
            gallery = self.registry.load_gallery()
            text = f"G{gid:06d}"
            for group in component:
                for key in group.members:
                    result[key] = text
        return result

    def resolve(self, local_mapping: Mapping[str, str], tracks: Mapping[str, object], cameras: List[str]):
        groups = self.build_groups(local_mapping, tracks)
        by_camera = {camera: [group for group in groups if group.camera == camera] for camera in cameras}
        edges: List[dict] = []
        ordered = sorted(cameras)
        for index, camera_a in enumerate(ordered):
            for camera_b in ordered[index + 1:]:
                edges.extend(self.cross_edges(by_camera.get(camera_a, []), by_camera.get(camera_b, [])))
        components = self._components(groups, edges)
        mapping = self._assign(components)
        component_output: Dict[str, List[str]] = {}
        for component in components:
            gid = mapping[component[0].members[0]]
            component_output[gid] = sorted(key for group in component for key in group.members)
        return mapping, component_output, edges
