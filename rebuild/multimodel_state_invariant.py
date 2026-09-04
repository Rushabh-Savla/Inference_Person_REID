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
    state_bank: Dict[str, Dict[str, List[np.ndarray]]] = field(default_factory=dict)
    colour_signature: np.ndarray | None = None


class StateInvariantResolver:
    """Final cross-camera resolver for standing/standing and sitting/standing ReID.

    V6 remains the same-camera authority. Cross-camera matching never trusts an
    already-global label. Instead it compares camera-local groups using three
    independent embedding spaces across multiple crop states, especially upper/
    torso evidence that survives desk occlusion and sitting-to-standing changes.
    """

    MODELS = ("resnet", "swin", "solider")
    DIRECT_VIEWS = ("full", "upper", "torso", "lower")
    TRANSITION_VIEWS = (
        ("full", "upper"),
        ("upper", "full"),
        ("full", "torso"),
        ("torso", "full"),
        ("upper", "torso"),
        ("torso", "upper"),
        ("full", "lower"),
        ("lower", "full"),
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
        self.transition_min = int(cfg.get("state_transition_model_support", 2))
        self.transition_view_support = int(cfg.get("state_transition_view_support", 2))
        self.max_time_gap = float(cfg.get("state_cross_max_gap_sec", 8.0))
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
    def _topmean(cls, left: Iterable[np.ndarray], right: Iterable[np.ndarray], k: int = 5) -> Tuple[float, int]:
        a = [cls._unit(x) for x in left if x is not None]
        b = [cls._unit(x) for x in right if x is not None]
        if not a or not b:
            return 0.0, 0
        matrix = np.asarray([[float(np.dot(x, y)) for y in b] for x in a], np.float32)
        flat = np.sort(matrix.reshape(-1))[::-1]
        take = flat[: min(k, flat.size)]
        return float(np.mean(take)), int((flat >= 0.60).sum())

    @classmethod
    def _model_score(cls, left: Mapping[str, List[np.ndarray]], right: Mapping[str, List[np.ndarray]]) -> Tuple[float, int, bool]:
        direct: List[float] = []
        support = 0
        for view in cls.DIRECT_VIEWS:
            score, count = cls._topmean(left.get(view, []), right.get(view, []))
            if left.get(view) and right.get(view):
                direct.append(score)
                support += min(count, 3)

        transition: List[float] = []
        for a, b in cls.TRANSITION_VIEWS:
            if not left.get(a) or not right.get(b):
                continue
            score, count = cls._topmean(left[a], right[b])
            transition.append(score)
            support += min(count, 2)

        same = float(np.mean(sorted(direct, reverse=True)[:2])) if direct else 0.0
        trans = max(transition) if transition else 0.0
        if transition and direct:
            value = max(same, 0.72 * trans + 0.28 * same)
        else:
            value = trans if transition else same
        return float(value), support, bool(transition and trans >= same)

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
        return float(max(0.0, 1.0 - gap / max(self.max_time_gap, 1e-6)))

    @staticmethod
    def _weights(camera_a: str, camera_b: str) -> Dict[str, float]:
        pair = "-".join(sorted((camera_a, camera_b)))
        if pair == "cam_222-cam_224":
            data = {"resnet": 0.29, "swin": 0.34, "solider": 0.31, "colour": 0.04, "temporal": 0.02}
        else:
            data = {"resnet": 0.31, "swin": 0.34, "solider": 0.30, "colour": 0.03, "temporal": 0.02}
        total = sum(data.values())
        return {k: v / total for k, v in data.items()}

    def _pair(self, left: LocalGroup, right: LocalGroup) -> PairEvidence:
        scores: Dict[str, float] = {}
        supports = 0
        view_support = 0
        transition = False
        for model in self.MODELS:
            value, support, is_transition = self._model_score(
                left.state_bank.get(model, {}), right.state_bank.get(model, {})
            )
            scores[model] = value
            supports += support
            transition = transition or is_transition
        for a, b in self.TRANSITION_VIEWS:
            if left.state_bank.get("swin", {}).get(a) and right.state_bank.get("swin", {}).get(b):
                view_support += 1
            if left.state_bank.get("solider", {}).get(a) and right.state_bank.get("solider", {}).get(b):
                view_support += 1
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
        ordered = sorted(scores.values(), reverse=True)
        agreement = float(np.mean(ordered[:2])) if len(ordered) >= 2 else ordered[0]
        model_support = sum(score >= self.model_min[name] for name, score in scores.items())
        state_transition = transition and view_support >= self.transition_view_support
        return PairEvidence(
            fused=float(fused),
            resnet=float(scores["resnet"]),
            swin=float(scores["swin"]),
            solider=float(scores["solider"]),
            colour=float(colour),
            temporal=float(temporal),
            agreement=agreement,
            model_support=int(model_support),
            view_support=int(view_support),
            state_transition=state_transition,
        )

    @staticmethod
    def _accept(evidence: PairEvidence, row_margin: float, col_margin: float, fused_min: float, margin: float) -> bool:
        if evidence.fused < fused_min or row_margin < margin or col_margin < margin:
            return False
        if evidence.model_support < 2:
            return False
        if max(evidence.resnet, evidence.swin, evidence.solider) < 0.62:
            return False
        if evidence.agreement < 0.60:
            return False
        if evidence.temporal < 0.25 and max(evidence.resnet, evidence.swin, evidence.solider) < 0.88:
            return False
        return True

    @staticmethod
    def _greedy(matrix: np.ndarray) -> List[Tuple[int, int]]:
        edges = sorted(
            ((float(matrix[i, j]), i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if matrix[i, j] > 0),
            reverse=True,
        )
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        result: List[Tuple[int, int]] = []
        for _score, i, j in edges:
            if i in used_rows or j in used_cols:
                continue
            used_rows.add(i)
            used_cols.add(j)
            result.append((i, j))
        return result

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
                start = float(tracks[key].start)
                end = float(tracks[key].end)
                placed = False
                for index, (lane_end, members) in enumerate(lanes):
                    if start >= lane_end:
                        members.append(key)
                        lanes[index] = (max(lane_end, end), members)
                        placed = True
                        break
                if not placed:
                    lanes.append((end, [key]))

            for lane_index, _lane in enumerate(lanes):
                members = lanes[lane_index][1]
                state_bank: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
                colours: List[np.ndarray] = []
                for key in members:
                    track = tracks[key]
                    bank = getattr(track, "state_bank", {})
                    for model, views in bank.items():
                        for view, values in views.items():
                            state_bank[model][view].extend(np.asarray(v, np.float32) for v in values)
                    if getattr(track, "colour_signature", None) is not None:
                        colours.append(np.asarray(track.colour_signature, np.float32))
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
                        state_bank={m: dict(v) for m, v in state_bank.items()},
                        colour_signature=colour,
                    )
                )
        return sorted(groups, key=lambda g: (g.camera, g.start, g.key))

    def _pair_groups(self, left: List[LocalGroup], right: List[LocalGroup]) -> List[dict]:
        if not left or not right:
            return []
        matrix = np.zeros((len(left), len(right)), np.float32)
        evidence: Dict[Tuple[int, int], PairEvidence] = {}
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                item = self._pair(a, b)
                evidence[(i, j)] = item
                threshold = self.partial_fused_min if item.state_transition else self.fused_min
                if item.fused >= threshold:
                    matrix[i, j] = item.fused

        result: List[dict] = []
        for i, j in self._greedy(matrix):
            item = evidence[(i, j)]
            row = np.sort(matrix[i][matrix[i] > 0])[::-1]
            col = np.sort(matrix[:, j][matrix[:, j] > 0])[::-1]
            row_margin = float(row[0] - row[1]) if row.size > 1 else 1.0
            col_margin = float(col[0] - col[1]) if col.size > 1 else 1.0
            threshold = self.partial_fused_min if item.state_transition else self.fused_min
            margin = self.partial_margin if item.state_transition else self.margin
            if self._accept(item, row_margin, col_margin, threshold, margin):
                result.append({
                    "left": left[i].key,
                    "right": right[j].key,
                    "fused": item.fused,
                    "resnet": item.resnet,
                    "swin": item.swin,
                    "solider": item.solider,
                    "colour": item.colour,
                    "temporal": item.temporal,
                    "agreement": item.agreement,
                    "model_support": item.model_support,
                    "view_support": item.view_support,
                    "state_transition": item.state_transition,
                    "row_margin": row_margin,
                    "col_margin": col_margin,
                    "left_members": left[i].members,
                    "right_members": right[j].members,
                })
        return result

    @staticmethod
    def _union(groups: List[LocalGroup], edges: List[dict]) -> List[List[LocalGroup]]:
        parent = {g.key: g.key for g in groups}
        members = {g.key: {g.key} for g in groups}
        lookup = {g.key: g for g in groups}

        def find(key: str) -> str:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for edge in sorted(edges, key=lambda x: x["fused"], reverse=True):
            a, b = find(edge["left"]), find(edge["right"])
            if a == b:
                continue
            cams_a = {lookup[node].camera for node in members[a]}
            cams_b = {lookup[node].camera for node in members[b]}
            if cams_a & cams_b:
                continue
            parent[b] = a
            members[a].update(members[b])
            del members[b]

        result: Dict[str, List[LocalGroup]] = defaultdict(list)
        for group in groups:
            result[find(group.key)].append(group)
        return sorted(result.values(), key=lambda comp: min((g.start, g.key) for g in comp))

    def _gallery_score(self, component: List[LocalGroup], gallery: Mapping[int, Mapping[str, List[np.ndarray]]]) -> Dict[int, float]:
        # Persisted gallery has no crop-state labels by design. Compare all stored
        # exemplars to all current state views and require two model spaces.
        current: Dict[str, List[np.ndarray]] = defaultdict(list)
        for group in component:
            for model, views in group.state_bank.items():
                for values in views.values():
                    current[model].extend(values)
        scores: Dict[int, float] = {}
        weights = {"resnet": 0.31, "swin": 0.35, "solider": 0.31}
        for gid, stored in gallery.items():
            parts = []
            for model in self.MODELS:
                if not current.get(model) or not stored.get(model):
                    continue
                score, _ = self._topmean(current[model], stored[model])
                parts.append((weights[model], score))
            if len(parts) >= 2:
                total = sum(weight for weight, _ in parts)
                scores[int(gid)] = sum(weight * score for weight, score in parts) / total
        return scores

    def _persistent_assign(self, components: List[List[LocalGroup]]) -> Dict[str, str]:
        if self.registry is None:
            out: Dict[str, str] = {}
            for index, component in enumerate(components, 1):
                gid = f"G{index:06d}"
                for group in component:
                    for key in group.members:
                        out[key] = gid
            return out

        gallery = self.registry.load_gallery()
        result: Dict[str, str] = {}
        used: set[int] = set()
        for component in components:
            ranked = sorted(self._gallery_score(component, gallery).items(), key=lambda item: item[1], reverse=True)
            gid: int | None = None
            if ranked:
                best_gid, best = ranked[0]
                second = ranked[1][1] if len(ranked) > 1 else 0.0
                if best >= self.gallery_min and (best - second) >= self.gallery_margin and best_gid not in used:
                    gid = int(best_gid)
            if gid is None:
                gid = self.registry.allocate_gid()
            used.add(gid)
            banks: Dict[str, List[np.ndarray]] = defaultdict(list)
            cameras: set[str] = set()
            last_ts = 0.0
            obs = 0
            for group in component:
                cameras.add(group.camera)
                last_ts = max(last_ts, group.end)
                obs += len(group.members)
                for model, views in group.state_bank.items():
                    for values in views.values():
                        banks[model].extend(values)
            self.registry.save_component(gid, model_banks=banks, cameras=cameras, last_ts=last_ts, obs=obs)
            gallery = self.registry.load_gallery()
            text = f"G{gid:06d}"
            for group in component:
                for key in group.members:
                    result[key] = text
        return result

    def resolve(self, local_mapping: Mapping[str, str], tracks: Mapping[str, object], cameras: List[str]):
        groups = self.build_groups(local_mapping, tracks)
        by_camera = {camera: [g for g in groups if g.camera == camera] for camera in cameras}
        edges: List[dict] = []
        ordered = sorted(cameras)
        for index, camera_a in enumerate(ordered):
            for camera_b in ordered[index + 1:]:
                edges.extend(self._pair_groups(by_camera.get(camera_a, []), by_camera.get(camera_b, [])))
        components = self._union(groups, edges)
        mapping = self._persistent_assign(components)
        output_components: Dict[str, List[str]] = {}
        for component in components:
            sample = component[0].members[0]
            gid = mapping[sample]
            output_components[gid] = sorted(key for group in component for key in group.members)
        return mapping, output_components, edges
