from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from rebuild.multimodel_state_invariant_final import PairEvidence
from rebuild.multimodel_state_invariant_fast import StateInvariantFinalResolverFast


class AttributeAwareResolver(StateInvariantFinalResolverFast):
    """Primary 3-model ReID with bounded clothing/face evidence and overlap recovery gates."""

    VIEWS = ("full", "upper", "torso", "lower", "attributes")

    def __init__(self, cfg, registry=None):
        super().__init__(cfg, registry=registry)
        self.attr_color = float(cfg.get("attribute_color_min", 0.84))
        self.attr_pattern = float(cfg.get("attribute_pattern_min", 0.64))
        self.attr_detail = float(cfg.get("attribute_detail_min", 0.72))
        self.face_weight = float(cfg.get("face_match_weight", 0.40))
        self.clothing_weight = float(cfg.get("clothing_match_weight", 0.12))
        self.lower_weight = float(cfg.get("lower_match_weight", 0.10))
        self.lower_conflict_penalty = float(cfg.get("lower_conflict_penalty", 0.08))
        self.lower_conflict_floor = float(cfg.get("lower_conflict_floor", 0.38))
        self.face_match_min = float(cfg.get("face_match_min", 0.72))
        self.face_strong = float(cfg.get("face_strong", 0.82))
        self._meta: dict[tuple[str, str], dict] = {}

    @staticmethod
    def _unit(value):
        arr = np.asarray(value, np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if arr.size == 0 or not np.isfinite(norm) or norm <= 0.0:
            return None
        return arr / norm

    @classmethod
    def _best(cls, left: Iterable[np.ndarray], right: Iterable[np.ndarray]) -> float:
        vals = []
        for first in left:
            a = cls._unit(first)
            if a is None:
                continue
            for second in right:
                b = cls._unit(second)
                if b is None or a.shape != b.shape:
                    continue
                vals.append(float(np.dot(a, b)))
        if not vals:
            return 0.5
        vals.sort(reverse=True)
        return float(np.mean(vals[: min(3, len(vals))]))

    @staticmethod
    def _attr_bank(group):
        values = getattr(group, "attribute_bank", None)
        if values:
            return values
        return group.state_bank.get("resnet", {}).get("attributes", [])

    @classmethod
    def _attrs(cls, left, right):
        first = [np.asarray(x, np.float32).reshape(-1) for x in cls._attr_bank(left)]
        second = [np.asarray(x, np.float32).reshape(-1) for x in cls._attr_bank(right)]
        first = [x for x in first if x.size == 112 and np.isfinite(x).all()]
        second = [x for x in second if x.size == 112 and np.isfinite(x).all()]
        if not first or not second:
            return {"ready": False}
        a, b = np.stack(first), np.stack(second)
        upper_color = cls._best(a[:, 0:20], b[:, 0:20])
        lower_color = cls._best(a[:, 20:40], b[:, 20:40])
        upper_pattern = cls._best(a[:, 40:54], b[:, 40:54])
        lower_pattern = cls._best(a[:, 54:68], b[:, 54:68])
        head = cls._best(a[:, 68:102], b[:, 68:102])
        eye = cls._best(a[:, 102:108], b[:, 102:108])
        upper_visible = bool(np.mean(a[:, 108] > 0.0) >= 0.5 and np.mean(b[:, 108] > 0.0) >= 0.5)
        lower_visible = bool(np.mean(a[:, 109] > 0.0) >= 0.5 and np.mean(b[:, 109] > 0.0) >= 0.5)
        head_visible = bool(np.mean(a[:, 110] > 0.0) >= 0.5 and np.mean(b[:, 110] > 0.0) >= 0.5)
        eye_visible = bool(np.mean(a[:, 111] > 0.0) >= 0.5 and np.mean(b[:, 111] > 0.0) >= 0.5)
        upper = 0.62 * upper_color + 0.38 * upper_pattern
        lower = 0.60 * lower_color + 0.40 * lower_pattern
        pattern = 0.50 * upper_pattern + 0.50 * lower_pattern
        clothing = 0.34 * upper + 0.52 * lower + 0.14 * pattern
        conflict = lower_visible and upper >= 0.78 and lower < cls.lower_conflict_floor
        return {
            "ready": True,
            "upper": float(upper), "lower": float(lower), "pattern": float(pattern), "clothing": float(clothing),
            "upper_color": float(upper_color), "lower_color": float(lower_color),
            "upper_pattern": float(upper_pattern), "lower_pattern": float(lower_pattern),
            "head": float(head), "eye": float(eye),
            "upper_visible": upper_visible, "lower_visible": lower_visible,
            "head_visible": head_visible, "eye_visible": eye_visible, "conflict": bool(conflict),
        }

    @classmethod
    def _face_bank(cls, group):
        return [x for x in (getattr(group, "face_bank", []) or []) if isinstance(x, dict) and x.get("valid")]

    @classmethod
    def _face(cls, left, right):
        first, second = cls._face_bank(left), cls._face_bank(right)
        if not first or not second:
            return {"valid": False, "score": 0.0, "quality": 0.0}
        pairs = []
        for a in first:
            va = cls._unit(a.get("vector"))
            if va is None:
                continue
            for b in second:
                vb = cls._unit(b.get("vector"))
                if vb is None or va.shape != vb.shape:
                    continue
                pairs.append((float(np.dot(va, vb)), min(float(a.get("quality", 0.0)), float(b.get("quality", 0.0)))))
        if not pairs:
            return {"valid": False, "score": 0.0, "quality": 0.0}
        pairs.sort(reverse=True)
        top = pairs[: min(3, len(pairs))]
        return {"valid": True, "score": float(np.mean([x[0] for x in top])), "quality": float(np.mean([x[1] for x in top]))}

    @classmethod
    def _best_view_score(cls, left, right):
        left = {key: value for key, value in left.items() if key != "attributes"}
        right = {key: value for key, value in right.items() if key != "attributes"}
        return super()._best_view_score(left, right)

    @staticmethod
    def _merge_groups_preserve(left, right):
        merged = StateInvariantFinalResolverFast._merge_groups(left, right)
        attrs = list(getattr(left, "attribute_bank", []) or []) + list(getattr(right, "attribute_bank", []) or [])
        faces = list(getattr(left, "face_bank", []) or []) + list(getattr(right, "face_bank", []) or [])
        setattr(merged, "attribute_bank", attrs[-64:])
        setattr(merged, "face_bank", faces[-32:])
        setattr(merged, "overlap_recovery", bool(getattr(left, "overlap_recovery", False) or getattr(right, "overlap_recovery", False)))
        setattr(merged, "recovery_sources", sorted(set(getattr(left, "recovery_sources", []) or []) | set(getattr(right, "recovery_sources", []) or [])))
        return merged

    @classmethod
    def _group_from_track(cls, key, track, local_gid):
        group = super()._group_from_track(key, track, local_gid)
        setattr(group, "attribute_bank", list(getattr(track, "state_bank", {}).get("resnet", {}).get("attributes", []) or [])[-64:])
        setattr(group, "face_bank", list(getattr(track, "face_bank", []) or [])[-32:])
        setattr(group, "overlap_recovery", bool(getattr(track, "overlap_recovery", False)))
        setattr(group, "recovery_sources", list(getattr(track, "recovery_sources", []) or []))
        return group

    @staticmethod
    def _merge_groups(left, right):
        return AttributeAwareResolver._merge_groups_preserve(left, right)

    def pair(self, left, right, left_pool, right_pool):
        evidence = super().pair(left, right, left_pool, right_pool)
        attrs = self._attrs(left, right)
        face = self._face(left, right)
        fused = float(evidence.fused)
        if attrs.get("ready"):
            if attrs["lower_visible"]:
                fused = 0.88 * fused + 0.08 * attrs["upper"] + 0.04 * attrs["lower"]
                fused = 0.90 * fused + 0.10 * attrs["lower"]
            else:
                fused = 0.94 * fused + 0.06 * attrs["upper"]
            if attrs["conflict"]:
                fused -= self.lower_conflict_penalty
        if face.get("valid") and face.get("quality", 0.0) >= 0.50 and face.get("score", 0.0) >= self.face_match_min:
            fused = (1.0 - self.face_weight) * fused + self.face_weight * float(face["score"])
        fused = float(np.clip(fused, 0.0, 0.99))
        self._meta[tuple(sorted((left.key, right.key)))] = {"attributes": attrs, "face": face}
        return PairEvidence(
            fused, evidence.resnet, evidence.swin, evidence.solider, evidence.colour,
            evidence.geometry, evidence.temporal, evidence.continuity, evidence.agreement,
            evidence.model_support, evidence.mutual_models, evidence.view_support, evidence.state_transition,
        )

    def _accept(self, evidence, left, right, same=False):
        meta = self._meta.get(tuple(sorted((left.key, right.key))), {})
        attrs = meta.get("attributes", {})
        face = meta.get("face", {})
        threshold = self.partial_min if evidence.state_transition else (self.same_min if same else self.cross_min)
        if evidence.fused < threshold:
            return False
        if attrs.get("conflict"):
            # A strong shirt match plus a visibly incompatible lower body is not
            # identity proof. Only very strong 3-model + face evidence can override it.
            if not (evidence.model_support == 3 and evidence.agreement >= 0.78 and face.get("valid") and face.get("score", 0.0) >= self.face_strong):
                return False
        if face.get("valid") and face.get("score", 0.0) >= self.face_strong and evidence.model_support >= 2 and evidence.agreement < 0.43:
            return False
        if evidence.model_support >= 2 and evidence.agreement >= (0.50 if same else 0.48):
            return (not same) or evidence.continuity >= self.same_spatial_min
        if not same and max(evidence.resnet, evidence.swin, evidence.solider) >= self.strong:
            return bool(face.get("valid") and face.get("score", 0.0) >= self.face_match_min)
        return False

    def same_camera_edges(self, groups):
        ordered = sorted(groups, key=lambda group: (group.start, group.end, group.key))
        candidates = []
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                if right.start <= left.end:
                    continue
                if right.start - left.end > self.same_max_gap:
                    break
                recovery = bool(getattr(left, "overlap_recovery", False) or getattr(right, "overlap_recovery", False))
                sources = set(getattr(left, "recovery_sources", []) or []) | set(getattr(right, "recovery_sources", []) or [])
                if recovery and sources and left.key not in sources and right.key not in sources:
                    continue
                evidence = self.pair(left, right, ordered, ordered)
                if self._accept(evidence, left, right, True):
                    candidates.append((1 if recovery else 0, float(evidence.fused), left, right, evidence))
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        incoming, outgoing, result = set(), set(), []
        for priority, score, left, right, evidence in candidates:
            if left.key in outgoing or right.key in incoming:
                continue
            outgoing.add(left.key); incoming.add(right.key)
            meta = self._meta.get(tuple(sorted((left.key, right.key))), {})
            attrs = meta.get("attributes", {}); face = meta.get("face", {})
            result.append({
                "left": left.key, "right": right.key, "fused": score,
                "resnet": evidence.resnet, "swin": evidence.swin, "solider": evidence.solider,
                "colour": evidence.colour, "geometry": evidence.geometry, "temporal": evidence.temporal,
                "continuity": evidence.continuity, "agreement": evidence.agreement,
                "model_support": evidence.model_support, "view_support": evidence.view_support,
                "state_transition": evidence.state_transition, "recovery_priority": priority,
                "post_overlap_recovery": bool(priority),
                "recovery_source_match": bool(left.key in (getattr(right, "recovery_sources", []) or []) or right.key in (getattr(left, "recovery_sources", []) or [])),
                "upper_clothing": attrs.get("upper", 0.0), "lower_clothing": attrs.get("lower", 0.0),
                "upper_pattern": attrs.get("upper_pattern", 0.0), "lower_pattern": attrs.get("lower_pattern", 0.0),
                "face": face.get("score", 0.0), "face_valid": bool(face.get("valid", False)),
            })
        return result

    def cross_edges(self, left, right):
        if not left or not right:
            return []
        candidates = []
        for first in left:
            for second in right:
                evidence = self.pair(first, second, left, right)
                if self._accept(evidence, first, second, False):
                    meta = self._meta.get(tuple(sorted((first.key, second.key))), {})
                    attrs = meta.get("attributes", {}); face = meta.get("face", {})
                    candidates.append((float(evidence.fused), first, second, evidence, attrs, face))
        candidates.sort(key=lambda x: x[0], reverse=True)
        used_left, used_right, result = set(), set(), []
        for score, first, second, evidence, attrs, face in candidates:
            if first.key in used_left or second.key in used_right:
                continue
            used_left.add(first.key); used_right.add(second.key)
            result.append({
                "left": first.key, "right": second.key, "left_members": list(first.members), "right_members": list(second.members),
                "fused": score, "resnet": evidence.resnet, "swin": evidence.swin, "solider": evidence.solider,
                "colour": evidence.colour, "geometry": evidence.geometry, "temporal": evidence.temporal,
                "continuity": evidence.continuity, "agreement": evidence.agreement, "model_support": evidence.model_support,
                "mutual_models": evidence.mutual_models, "view_support": evidence.view_support,
                "state_transition": evidence.state_transition, "upper_clothing": attrs.get("upper", 0.0),
                "lower_clothing": attrs.get("lower", 0.0), "upper_pattern": attrs.get("upper_pattern", 0.0),
                "lower_pattern": attrs.get("lower_pattern", 0.0), "face": face.get("score", 0.0),
                "face_valid": bool(face.get("valid", False)),
                "post_overlap_recovery": bool(getattr(first, "overlap_recovery", False) or getattr(second, "overlap_recovery", False)),
            })
        return result

    def _assign(self, components):
        if self.registry is None:
            result = {}
            next_id = 1
            for component in components:
                recovery_only = len(component) == 1 and bool(getattr(component[0], "overlap_recovery", False))
                gid = "PENDING" if recovery_only else f"G{next_id:06d}"
                if gid != "PENDING":
                    next_id += 1
                for group in component:
                    for key in group.members:
                        result[key] = gid
            return result
        gallery = self.registry.load_gallery()
        result, used = {}, set()
        for component in components:
            ranked = sorted(self._gallery_score(component, gallery).items(), key=lambda item: item[1], reverse=True)
            recovery_only = len(component) == 1 and bool(getattr(component[0], "overlap_recovery", False))
            gid = None
            if ranked:
                best, score = ranked[0]
                second = ranked[1][1] if len(ranked) > 1 else 0.0
                if score >= self.gallery_min and score - second >= self.gallery_margin and best not in used:
                    gid = int(best)
            if gid is None and recovery_only:
                for group in component:
                    for key in group.members:
                        result[key] = "PENDING"
                continue
            if gid is None:
                gid = self.registry.allocate_gid()
            used.add(gid)
            self.registry.save_component(gid, model_banks=self._flat_gallery(component), cameras={g.camera for g in component}, last_ts=max(g.end for g in component), obs=len([key for g in component for key in g.members]))
            gallery = self.registry.load_gallery()
            text = f"G{gid:06d}"
            for group in component:
                for key in group.members:
                    result[key] = text
        return result

    def resolve(self, local_mapping, tracks, cameras):
        self._meta.clear()
        return super().resolve(local_mapping, tracks, cameras)


__all__ = ["AttributeAwareResolver"]
