from __future__ import annotations

from typing import Iterable

import numpy as np

from rebuild.multimodel_state_invariant_fast import StateInvariantFinalResolverFast


class AttributeAwareResolver(StateInvariantFinalResolverFast):
    """High-confidence V6 resolver with attributes as supporting evidence.

    Deep ReID remains the primary identity signal. Clothing/head/detail
    attributes can reinforce a strong multimodel decision, but they can never
    create an identity by themselves. Same-camera fragment repair also permits
    sequential chains instead of limiting every tracklet to one repair edge.
    """

    VIEWS = ("full", "upper", "torso", "lower", "attributes")

    def __init__(self, cfg, registry=None):
        super().__init__(cfg, registry=registry)
        self.attr_color = float(cfg.get("attribute_color_min", 0.84))
        self.attr_pattern = float(cfg.get("attribute_pattern_min", 0.64))
        self.attr_detail = float(cfg.get("attribute_detail_min", 0.72))
        self.same_chain_min = float(cfg.get("state_same_chain_min", 0.53))
        self.same_chain_support = int(cfg.get("state_same_chain_support", 2))
        self.same_chain_continuity = float(cfg.get("state_same_chain_continuity_min", 0.28))
        self._pending = {"ready": False}

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

    @classmethod
    def _pack(cls, group):
        values = list(group.state_bank.get("resnet", {}).get("attributes", []))
        arr = [np.asarray(value, np.float32).reshape(-1) for value in values if value is not None]
        arr = [value for value in arr if value.size == 112 and np.isfinite(value).all()]
        return np.stack(arr) if arr else None

    @classmethod
    def _attrs(cls, left, right):
        a = cls._pack(left)
        b = cls._pack(right)
        if a is None or b is None:
            return {"ready": False}

        au, al = a[:, 0:20], a[:, 20:40]
        ap, aq = a[:, 40:54], a[:, 54:68]
        ah, ae = a[:, 68:102], a[:, 102:108]
        av = a[:, 108:112]
        bu, bl = b[:, 0:20], b[:, 20:40]
        bp, bq = b[:, 40:54], b[:, 54:68]
        bh, be = b[:, 68:102], b[:, 102:108]
        bv = b[:, 108:112]

        upvis = float(np.mean(av[:, 0]) > 0.15 and np.mean(bv[:, 0]) > 0.15)
        lowvis = float(np.mean(av[:, 1]) > 0.15 and np.mean(bv[:, 1]) > 0.15)
        headvis = float(np.mean(av[:, 2]) > 0.15 and np.mean(bv[:, 2]) > 0.15)
        eyevis = float(np.mean(av[:, 3]) > 0.15 and np.mean(bv[:, 3]) > 0.15)
        colorup = cls._best(au, bu)
        colorlow = cls._best(al, bl)
        patternup = cls._best(ap, bp)
        patternlow = cls._best(aq, bq)
        headscore = cls._best(ah, bh)
        eyescore = cls._best(ae, be)
        upright = bool(float(getattr(left, "aspect", 0.0)) >= 1.35 and float(getattr(right, "aspect", 0.0)) >= 1.35)
        return {
            "ready": True,
            "full": bool(upvis and lowvis and upright),
            "upper": colorup,
            "lower": colorlow,
            "upperpattern": patternup,
            "lowerpattern": patternlow,
            "head": headscore,
            "eye": eyescore,
            "headvis": bool(headvis),
            "eyevis": bool(eyevis),
            "detail": max(patternup, patternlow, headscore, eyescore),
            "cloth": min(colorup, colorlow),
        }

    @classmethod
    def _best_view_score(cls, left, right):
        l = {key: value for key, value in left.items() if key != "attributes"}
        r = {key: value for key, value in right.items() if key != "attributes"}
        return super()._best_view_score(l, r)

    @staticmethod
    def _flat_gallery(component):
        result = {}
        for group in component:
            for model, views in group.state_bank.items():
                for view, values in views.items():
                    if view != "attributes":
                        result.setdefault(model, []).extend(values)
        return result

    def pair(self, left, right, left_pool, right_pool):
        evidence = super().pair(left, right, left_pool, right_pool)
        self._pending = self._attrs(left, right)
        attr = self._pending
        # Attributes are deliberately capped reinforcement. They never replace
        # the three-model ReID decision and can add at most 0.025 to the fused
        # score when both upper and lower evidence are actually visible.
        if attr.get("ready") and evidence.model_support >= 2:
            bonus = 0.0
            if attr["full"] and attr["cloth"] >= self.attr_color and min(attr["upperpattern"], attr["lowerpattern"]) >= self.attr_pattern:
                bonus = 0.025
            elif not attr["full"] and attr["detail"] >= self.attr_detail and (attr["headvis"] or attr["eyevis"] or attr["upperpattern"] >= self.attr_pattern or attr["lowerpattern"] >= self.attr_pattern):
                bonus = 0.015
            if bonus:
                from rebuild.multimodel_state_invariant_final import PairEvidence
                evidence = PairEvidence(
                    min(0.99, float(evidence.fused + bonus)),
                    evidence.resnet,
                    evidence.swin,
                    evidence.solider,
                    evidence.colour,
                    evidence.geometry,
                    evidence.temporal,
                    evidence.continuity,
                    evidence.agreement,
                    evidence.model_support,
                    evidence.mutual_models,
                    evidence.view_support,
                    evidence.state_transition,
                )
        return evidence

    def _accept(self, evidence, same=False):
        # Strong assignment comes from at least two independent ReID models.
        # Attribute evidence only reinforces an already-credible match.
        threshold = self.partial_min if evidence.state_transition else (self.same_min if same else self.cross_min)
        if evidence.fused < threshold:
            return False
        if evidence.model_support >= 2 and evidence.agreement >= (0.50 if same else 0.48):
            if same:
                return evidence.continuity >= self.same_spatial_min
            return True
        return False

    def _same_accept(self, evidence):
        return self._accept(evidence, True)

    def _cross_accept(self, evidence):
        return self._accept(evidence, False)

    def same_camera_edges(self, groups):
        """Build sequential same-camera repair links, allowing chains.

        A tracklet may have one incoming and one outgoing repair edge, which
        lets A->B->C represent one person after repeated tracker resets while
        still preventing unrelated overlapping intervals from being joined.
        """
        ordered = sorted(groups, key=lambda group: (group.start, group.end, group.key))
        candidates = []
        for i, left in enumerate(ordered):
            for j in range(i + 1, len(ordered)):
                right = ordered[j]
                if right.start <= left.end:
                    continue
                gap = right.start - left.end
                if gap > self.same_max_gap:
                    break
                evidence = self.pair(left, right, ordered, ordered)
                strong = evidence.fused >= self.same_chain_min and evidence.model_support >= self.same_chain_support and evidence.continuity >= self.same_chain_continuity and evidence.agreement >= 0.52
                if self._same_accept(evidence) or strong:
                    candidates.append((float(evidence.fused), left, right, evidence))

        candidates.sort(key=lambda item: item[0], reverse=True)
        incoming = set()
        outgoing = set()
        chosen = []
        for score, left, right, evidence in candidates:
            if left.key in outgoing or right.key in incoming:
                continue
            # Do not create a same-camera edge if the two tracklets overlap in
            # time; overlap is a physical two-person constraint, not a stitch.
            if left.end >= right.start:
                continue
            outgoing.add(left.key)
            incoming.add(right.key)
            chosen.append({
                "left": left.key,
                "right": right.key,
                "fused": score,
                "resnet": evidence.resnet,
                "swin": evidence.swin,
                "solider": evidence.solider,
                "colour": evidence.colour,
                "continuity": evidence.continuity,
                "geometry": evidence.geometry,
                "temporal": evidence.temporal,
                "agreement": evidence.agreement,
                "model_support": evidence.model_support,
                "state_transition": evidence.state_transition,
            })
        return chosen
