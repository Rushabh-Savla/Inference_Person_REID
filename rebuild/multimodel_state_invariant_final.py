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
    geometry: float
    agreement: float
    model_support: int
    mutual_models: int
    view_support: int
    state_transition: bool
    continuity: float


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

        self.cross_min = float(cfg.get("state_cross_fused_min", 0.58))
        self.cross_partial_min = float(cfg.get("state_cross_partial_fused_min", 0.54))
        self.cross_margin = float(cfg.get("state_cross_margin", 0.025))
        self.cross_partial_margin = float(cfg.get("state_cross_partial_margin", 0.020))
        self.cross_strong = float(cfg.get("state_cross_strong", 0.82))

        self.model_min = {
            "resnet": float(cfg.get("state_cross_resnet_min", 0.50)),
            "swin": float(cfg.get("state_cross_swin_min", 0.48)),
            "solider": float(cfg.get("state_cross_solider_min", 0.48)),
        }
        self.required_models = int(cfg.get("state_cross_required_models", 2))
        self.required_mutual = int(cfg.get("state_cross_required_mutual_models", 2))
        self.required_views = int(cfg.get("state_cross_required_views", 2))

        self.same_gap = float(cfg.get("state_same_camera_gap_sec", 20.0))
        self.same_fused_min = float(cfg.get("state_same_camera_fused_min", 0.58))
        self.same_continuity_min = float(cfg.get("state_same_camera_continuity_min", 0.32))
        self.same_strong = float(cfg.get("state_same_camera_strong", 0.72))

        self.gallery_min = float(cfg.get("state_gallery_match_min", 0.66))
        self.gallery_margin = float(cfg.get("state_gallery_margin", 0.03))
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
    def _best_state_score(
        cls,
        left: Mapping[str, List[np.ndarray]],
        right: Mapping[str, List[np.ndarray]],
    ) -> Tuple[float, int, bool]:
        direct: List[Tuple[str, float]] = []
        transition: List[Tuple[str, float]] = []

        for view in cls.VIEWS:
            if left.get(view) and right.get(view):
                matrix = cls._matrix(left[view], right[view])
                if matrix.size:
                    direct.append((view, cls._topmean_matrix(matrix, 3)))

        for left_view, right_view in cls.TRANSITION_VIEWS:
            if left.get(left_view) and right.get(right_view):
                matrix = cls._matrix(left[left_view], right[right_view])
                if matrix.size:
                    transition.append(
                        (f"{left_view}>{right_view}", cls._topmean_matrix(matrix, 3))
                    )

        direct_sorted = sorted(direct, key=lambda item: item[1], reverse=True)
        trans_sorted = sorted(transition, key=lambda item: item[1], reverse=True)

        direct_score = (
            float(np.mean([score for _, score in direct_sorted[:2]]))
            if direct_sorted else 0.0
        )
        trans_score = float(trans_sorted[0][1]) if trans_sorted else 0.0

        if direct_sorted and trans_sorted:
            score = max(direct_score, 0.80 * trans_score + 0.20 * direct_score)
        elif trans_sorted:
            score = trans_score
        else:
            score = direct_score

        view_support = sum(score_value >= 0.60 for _, score_value in direct_sorted)
        view_support += sum(score_value >= 0.60 for _, score_value in trans_sorted)
        state_transition = bool(trans_score >= direct_score + 0.01)
        return float(score), int(view_support), state_transition

    @staticmethod
    def _colour(left: LocalGroup, right: LocalGroup) -> float:
        if left.colour_signature is None or right.colour_signature is None:
            return 0.5
        return float(np.clip(np.dot(left.colour_signature, right.colour_signature), 0.0, 1.0))

    def _temporal(self, left: LocalGroup, right: LocalGroup) -> float:
        # Raw cross-camera clip timestamps are not an identity signal unless
        # camera offsets are explicitly calibrated. Without offsets, keep this
        # neutral so recording start-time differences cannot block true matches.
        if left.camera != right.camera and not self.offsets:
            return 0.5

        left_end = left.end + self.offsets.get(left.camera, 0.0)
        right_start = right.start + self.offsets.get(right.camera, 0.0)
        left_start = left.start + self.offsets.get(left.camera, 0.0)
        right_end = right.end + self.offsets.get(right.camera, 0.0)

        if left.camera == right.camera:
            if left.end > right.start:
                return 0.0
            gap = right_start - left_end
            return float(max(0.0, 1.0 - gap / max(self.same_gap, 1e-6)))

        if not (left_end < right_start or right_end < left_start):
            return 0.5
        gap = min(abs(left_end - right_start), abs(right_end - left_start))
        limit = float(self.cfg.get("state_cross_max_gap_sec", 8.0))
        return float(max(0.0, 1.0 - gap / max(limit, 1e-6)))

    @staticmethod
    def _continuity(left: LocalGroup, right: LocalGroup) -> float:
        if left.camera != right.camera:
            return 0.5
        if left.end > right.start:
            return 0.0
        if left.end_center is None or right.start_center is None:
            return 0.0
        height = max(1.0, float(left.end_height))
        dx = float(right.start_center[0] - left.end_center[0])
        dy = float(right.start_center[1] - left.end_center[1])
        distance = float(np.hypot(dx, dy) / height)
        return float(max(0.0, 1.0 - distance / 6.0))

    @staticmethod
    def _geometry(left: LocalGroup, right: LocalGroup) -> float:
        if left.aspect <= 0 or right.aspect <= 0:
            return 0.5
        ratio = max(left.aspect, right.aspect) / max(1e-6, min(left.aspect, right.aspect))
        return float(max(0.0, min(1.0, 1.0 / ratio)))

    @staticmethod
    def _weights(left: LocalGroup, right: LocalGroup) -> Dict[str, float]:
        if left.camera == right.camera:
            raw = {
                "resnet": 0.31,
                "swin": 0.33,
                "solider": 0.28,
                "colour": 0.02,
                "temporal": 0.03,
                "geometry": 0.03,
            }
        elif {left.camera, right.camera} == {"cam_222", "cam_224"}:
            raw = {
                "resnet": 0.28,
                "swin": 0.36,
                "solider": 0.31,
                "colour": 0.02,
                "temporal": 0.01,
                "geometry": 0.02,
            }
        else:
            raw = {
                "resnet": 0.30,
                "swin": 0.36,
                "solider": 0.30,
                "colour": 0.02,
                "temporal": 0.01,
                "geometry": 0.01,
            }
        total = sum(raw.values())
        return {key: value / total for key, value in raw.items()}

    def _model_matrices(
        self, left: LocalGroup, right: LocalGroup, model: str
    ) -> Tuple[Dict[Tuple[str, str], np.ndarray], List[str]]:
        left_views = left.state_bank.get(model, {})
        right_views = right.state_bank.get(model, {})
        result: Dict[Tuple[str, str], np.ndarray] = {}
        direct: List[str] = []

        for view in self.VIEWS:
            if left_views.get(view) and right_views.get(view):
                matrix = self._matrix(left_views[view], right_views[view])
                if matrix.size:
                    result[(view, view)] = matrix
                    direct.append(view)

        for left_view, right_view in self.TRANSITION_VIEWS:
            if left_views.get(left_view) and right_views.get(right_view):
                matrix = self._matrix(left_views[left_view], right_views[right_view])
                if matrix.size:
                    result[(left_view, right_view)] = matrix

        return result, direct

    def _model_score_for_pair(self, left: LocalGroup, right: LocalGroup, model: str) -> float:
        matrices, _ = self._model_matrices(left, right, model)
        values = [self._topmean_matrix(matrix, 3) for matrix in matrices.values() if matrix.size]
        return max(values) if values else 0.0

    def _mutual_model_count(
        self,
        left: LocalGroup,
        right: LocalGroup,
        left_pool: List[LocalGroup],
        right_pool: List[LocalGroup],
    ) -> int:
        count = 0
        for model in self.MODELS:
            row_scores = [self._model_score_for_pair(left, candidate, model) for candidate in right_pool]
            col_scores = [self._model_score_for_pair(candidate, right, model) for candidate in left_pool]
            if not row_scores or not col_scores:
                continue
            row_index = int(np.argmax(row_scores))
            col_index = int(np.argmax(col_scores))
            row_score = float(row_scores[row_index])
            col_score = float(col_scores[col_index])
            minimum = self.model_min[model]
            if (
                row_score >= minimum
                and col_score >= minimum
                and right_pool[row_index].key == right.key
                and left_pool[col_index].key == left.key
            ):
                count += 1
        return count

    def pair(
        self,
        left: LocalGroup,
        right: LocalGroup,
        left_pool: List[LocalGroup],
        right_pool: List[LocalGroup],
    ) -> PairEvidence:
        scores: Dict[str, float] = {}
        view_support = 0
        transition_votes = 0

        for model in self.MODELS:
            score, support, transition = self._best_state_score(
                left.state_bank.get(model, {}),
                right.state_bank.get(model, {}),
            )
            scores[model] = score
            view_support += support
            transition_votes += int(transition)

        ordered = sorted(scores.values(), reverse=True)
        agreement = float(np.mean(ordered[:2]))
        model_support = sum(score >= self.model_min[name] for name, score in scores.items())
        mutual_models = self._mutual_model_count(left, right, left_pool, right_pool)
        colour = self._colour(left, right)
        temporal = self._temporal(left, right)
        geometry = self._geometry(left, right)
        continuity = self._continuity(left, right)
        weights = self._weights(left, right)

        fused = (
            weights["resnet"] * scores["resnet"]
            + weights["swin"] * scores["swin"]
            + weights["solider"] * scores["solider"]
            + weights["colour"] * colour
            + weights["temporal"] * temporal
            + weights["geometry"] * geometry
        )

        actual_state_change = (
            left.state_type != "unknown"
            and right.state_type != "unknown"
            and left.state_type != right.state_type
        )
        state_transition = actual_state_change or transition_votes >= 2

        return PairEvidence(
            float(fused),
            float(scores["resnet"]),
            float(scores["swin"]),
            float(scores["solider"]),
            float(colour),
            float(temporal),
            float(geometry),
            agreement,
            int(model_support),
            int(mutual_models),
            int(view_support),
            bool(state_transition),
            float(continuity),
        )

    @classmethod
    def _state_type(cls, aspect: float) -> str:
        if aspect <= 0:
            return "unknown"
        return "upright" if aspect >= 1.35 else "compact"

    @staticmethod
    def _endpoint(track: object, last: bool) -> tuple[tuple[float, float] | None, float]:
        observations = list(getattr(track, "observations", []) or [])
        if not observations:
            return None, 1.0
        observations.sort(key=lambda item: float(item.get("timestamp", 0.0)))
        row = observations[-1] if last else observations[0]
        box = row.get("bbox")
        if not box or len(box) != 4:
            return None, 1.0
        x1, y1, x2, y2 = [float(value) for value in box]
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2)), max(1.0, y2 - y1)

    @classmethod
    def build_groups(
        cls,
        local_mapping: Mapping[str, str],
        tracks: Mapping[str, object],
    ) -> List[LocalGroup]:
        grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for key, gid in local_mapping.items():
            if key in tracks:
                grouped[(tracks[key].camera, str(gid))].append(key)

        groups: List[LocalGroup] = []
        for (camera, gid), keys in grouped.items():
            lanes: List[Tuple[float, List[str]]] = []
            for key in sorted(keys, key=lambda value: (float(tracks[value].start), float(tracks[value].end), value)):
                start = float(tracks[key].start)
                end = float(tracks[key].end)
                for lane_index, (lane_end, members) in enumerate(lanes):
                    if start > lane_end:
                        members.append(key)
                        lanes[lane_index] = (max(lane_end, end), members)
                        break
                else:
                    lanes.append((end, [key]))

            for lane_index, (_lane_end, members) in enumerate(lanes):
                state_bank: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
                colours: List[np.ndarray] = []
                aspects: List[float] = []
                start_center = None
                end_center = None
                end_height = 1.0
                ordered_members = sorted(members, key=lambda value: (float(tracks[value].start), value))

                for member_index, key in enumerate(ordered_members):
                    track = tracks[key]
                    for model, views in getattr(track, "state_bank", {}).items():
                        for view, values in views.items():
                            state_bank[model][view].extend(np.asarray(value, np.float32) for value in values)
                    if getattr(track, "colour_signature", None) is not None:
                        colours.append(np.asarray(track.colour_signature, np.float32))
                    ratio = float(getattr(track, "shape", 0.0))
                    if ratio > 0:
                        aspects.append(ratio)

                    begin_center, _ = cls._endpoint(track, False)
                    finish_center, finish_height = cls._endpoint(track, True)
                    if member_index == 0:
                        start_center = begin_center
                    if member_index == len(ordered_members) - 1:
                        end_center = finish_center
                        end_height = finish_height

                colour = None
                if colours:
                    colour = np.mean(np.stack(colours), axis=0)
                    colour /= np.linalg.norm(colour) + 1e-12

                aspect = float(np.median(aspects)) if aspects else 0.0
                groups.append(
                    LocalGroup(
                        key=f"{camera}:{gid}:lane{lane_index}",
                        camera=camera,
                        local_gid=str(gid),
                        members=sorted(members),
                        start=min(float(tracks[key].start) for key in members),
                        end=max(float(tracks[key].end) for key in members),
                        aspect=aspect,
                        state_type=cls._state_type(aspect),
                        state_bank={model: dict(views) for model, views in state_bank.items()},
                        colour_signature=colour,
                        start_center=start_center,
                        end_center=end_center,
                        end_height=end_height,
                    )
                )

        return sorted(groups, key=lambda group: (group.camera, group.start, group.key))

    def _acceptable_cross(self, evidence: PairEvidence, row_margin: float, col_margin: float) -> bool:
        threshold = self.cross_partial_min if evidence.state_transition else self.cross_min
        margin = self.cross_partial_margin if evidence.state_transition else self.cross_margin
        if evidence.fused < threshold:
            return False
        if evidence.model_support < self.required_models:
            return False
        if evidence.mutual_models < self.required_mutual and max(evidence.resnet, evidence.swin, evidence.solider) < self.cross_strong:
            return False
        if evidence.agreement < 0.58:
            return False
        if evidence.view_support < self.required_views:
            return False
        if row_margin < margin or col_margin < margin:
            return False
        if self.offsets and evidence.temporal < 0.05 and max(evidence.resnet, evidence.swin, evidence.solider) < 0.90:
            return False
        return True

    def cross_edges(self, left: List[LocalGroup], right: List[LocalGroup]) -> List[dict]:
        if not left or not right:
            return []

        matrix = np.zeros((len(left), len(right)), np.float32)
        details: Dict[Tuple[int, int], PairEvidence] = {}
        for i, left_group in enumerate(left):
            for j, right_group in enumerate(right):
                evidence = self.pair(left_group, right_group, left, right)
                details[(i, j)] = evidence
                threshold = self.cross_partial_min if evidence.state_transition else self.cross_min
                if evidence.fused >= threshold:
                    matrix[i, j] = evidence.fused

        order = sorted(
            (
                (float(matrix[i, j]), i, j)
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])
                if matrix[i, j] > 0
            ),
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
            if not self._acceptable_cross(evidence, row_margin, col_margin):
                continue

            used_left.add(i)
            used_right.add(j)
            result.append(
                {
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
                    "geometry": evidence.geometry,
                    "continuity": evidence.continuity,
                    "agreement": evidence.agreement,
                    "model_support": evidence.model_support,
                    "mutual_models": evidence.mutual_models,
                    "view_support": evidence.view_support,
                    "state_transition": evidence.state_transition,
                    "row_margin": row_margin,
                    "col_margin": col_margin,
                }
            )
        return result

    def _acceptable_same(self, evidence: PairEvidence, gap: float) -> bool:
        if gap < 0 or gap > self.same_gap:
            return False
        strong_appearance = evidence.fused >= self.same_strong and evidence.model_support >= 2 and evidence.mutual_models >= 2
        continuity_appearance = evidence.fused >= self.same_fused_min and evidence.model_support >= 2 and evidence.continuity >= self.same_continuity_min and evidence.temporal >= 0.15
        return bool(strong_appearance or continuity_appearance)

    def same_camera_edges(self, groups: List[LocalGroup]) -> List[dict]:
        by_camera: Dict[str, List[LocalGroup]] = defaultdict(list)
        for group in groups:
            by_camera[group.camera].append(group)

        result: List[dict] = []
        for camera_groups in by_camera.values():
            ordered = sorted(camera_groups, key=lambda group: (group.start, group.end, group.key))
            for right_index, right_group in enumerate(ordered):
                candidates = []
                for left_index in range(right_index - 1, -1, -1):
                    left_group = ordered[left_index]
                    gap = right_group.start - left_group.end
                    if gap < 0:
                        continue
                    if gap > self.same_gap:
                        break
                    evidence = self.pair(left_group, right_group, ordered, ordered)
                    if self._acceptable_same(evidence, gap):
                        candidates.append((evidence.fused + 0.08 * evidence.continuity, left_group, evidence, gap))

                if not candidates:
                    continue
                candidates.sort(key=lambda item: item[0], reverse=True)
                chosen = candidates[0]
                if len(candidates) > 1 and chosen[0] - candidates[1][0] < 0.04 and chosen[2].continuity < 0.70:
                    continue

                _, left_group, evidence, gap = chosen
                result.append(
                    {
                        "left": left_group.key,
                        "right": right_group.key,
                        "left_members": left_group.members,
                        "right_members": right_group.members,
                        "fused": evidence.fused,
                        "resnet": evidence.resnet,
                        "swin": evidence.swin,
                        "solider": evidence.solider,
                        "continuity": evidence.continuity,
                        "temporal": evidence.temporal,
                        "geometry": evidence.geometry,
                        "model_support": evidence.model_support,
                        "mutual_models": evidence.mutual_models,
                        "state_transition": evidence.state_transition,
                        "gap": float(gap),
                        "same_camera": True,
                    }
                )
        return result

    @staticmethod
    def _merge_groups(groups: List[LocalGroup], edges: List[dict]) -> List[LocalGroup]:
        lookup = {group.key: group for group in groups}
        parent = {group.key: group.key for group in groups}
        members: Dict[str, List[str]] = {group.key: [group.key] for group in groups}

        def find(key: str) -> str:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for edge in sorted(edges, key=lambda item: item["fused"], reverse=True):
            if edge["left"] not in parent or edge["right"] not in parent:
                continue
            left_root, right_root = find(edge["left"]), find(edge["right"])
            if left_root == right_root:
                continue

            left_groups = [lookup[key] for key in members[left_root]]
            right_groups = [lookup[key] for key in members[right_root]]
            if {group.camera for group in left_groups} != {group.camera for group in right_groups}:
                continue

            combined = left_groups + right_groups
            overlaps = False
            for index_a, group_a in enumerate(combined):
                for group_b in combined[index_a + 1:]:
                    if group_a.end > group_b.start and group_b.end > group_a.start:
                        overlaps = True
                        break
                if overlaps:
                    break
            if overlaps:
                continue

            parent[right_root] = left_root
            members[left_root].extend(members[right_root])
            del members[right_root]

        merged: List[LocalGroup] = []
        for grouped in members.values():
            ordered = sorted((lookup[key] for key in grouped), key=lambda group: (group.start, group.end, group.key))
            first = ordered[0]
            state_bank: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
            colours: List[np.ndarray] = []
            aspects: List[float] = []
            for group in ordered:
                for model, views in group.state_bank.items():
                    for view, values in views.items():
                        state_bank[model][view].extend(values)
                if group.colour_signature is not None:
                    colours.append(group.colour_signature)
                if group.aspect > 0:
                    aspects.append(group.aspect)

            colour = None
            if colours:
                colour = np.mean(np.stack(colours), axis=0)
                colour /= np.linalg.norm(colour) + 1e-12

            merged.append(
                LocalGroup(
                    key="+".join(sorted(group.key for group in ordered)),
                    camera=first.camera,
                    local_gid=first.local_gid,
                    members=sorted(member for group in ordered for member in group.members),
                    start=min(group.start for group in ordered),
                    end=max(group.end for group in ordered),
                    aspect=float(np.median(aspects)) if aspects else first.aspect,
                    state_type="upright" if float(np.median(aspects or [first.aspect])) >= 1.35 else "compact",
                    state_bank={model: dict(views) for model, views in state_bank.items()},
                    colour_signature=colour,
                    start_center=ordered[0].start_center,
                    end_center=ordered[-1].end_center,
                    end_height=ordered[-1].end_height,
                )
            )
        return sorted(merged, key=lambda group: (group.camera, group.start, group.key))

    @staticmethod
    def _components(groups: List[LocalGroup], edges: List[dict]) -> List[List[LocalGroup]]:
        parent = {group.key: group.key for group in groups}
        members = {group.key: {group.key} for group in groups}
        lookup = {group.key: group for group in groups}

        def find(key: str) -> str:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for edge in sorted(edges, key=lambda item: item["fused"], reverse=True):
            left_key = edge["left"]
            right_key = edge["right"]
            if left_key not in parent or right_key not in parent:
                continue
            left_root, right_root = find(left_key), find(right_key)
            if left_root == right_root:
                continue
            cameras_left = {lookup[key].camera for key in members[left_root]}
            cameras_right = {lookup[key].camera for key in members[right_root]}
            if cameras_left & cameras_right:
                continue
            parent[right_root] = left_root
            members[left_root].update(members[right_root])
            del members[right_root]

        result: Dict[str, List[LocalGroup]] = defaultdict(list)
        for group in groups:
            result[find(group.key)].append(group)
        return sorted(result.values(), key=lambda component: min((group.start, group.key) for group in component))

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
        weights = {"resnet": 0.30, "swin": 0.38, "solider": 0.30}
        result: Dict[int, float] = {}
        for gid, stored in gallery.items():
            values = []
            for model in self.MODELS:
                if not current.get(model) or not stored.get(model):
                    continue
                matrix = self._matrix(current[model], stored[model])
                if matrix.size:
                    values.append((weights[model], self._topmean_matrix(matrix, 5)))
            if len(values) >= self.required_models:
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
                best_gid, best_score = ranked[0]
                second_score = ranked[1][1] if len(ranked) > 1 else 0.0
                if best_score >= self.gallery_min and best_score - second_score >= self.gallery_margin and best_gid not in used:
                    gid = int(best_gid)
            if gid is None:
                gid = self.registry.allocate_gid()
            used.add(gid)
            self.registry.save_component(
                gid,
                model_banks=self._flat_gallery(component),
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

        # First repair same-camera track fragmentation. The previous design
        # explicitly forbade same-camera edges during component formation, so a
        # fragmented person could never recover their identity inside that camera.
        same_camera_edges = self.same_camera_edges(groups)
        stitched = self._merge_groups(groups, same_camera_edges)

        by_camera = {camera: [group for group in stitched if group.camera == camera] for camera in cameras}
        cross_edges: List[dict] = []
        ordered = sorted(cameras)
        for index, camera_a in enumerate(ordered):
            for camera_b in ordered[index + 1:]:
                cross_edges.extend(self.cross_edges(by_camera.get(camera_a, []), by_camera.get(camera_b, [])))

        components = self._components(stitched, cross_edges)
        mapping = self._assign(components)
        component_output: Dict[str, List[str]] = {}
        for component in components:
            gid = mapping[component[0].members[0]]
            component_output[gid] = sorted(member for group in component for member in group.members)

        return mapping, component_output, cross_edges
''