from __future__ import annotations

from typing import Iterable

import numpy as np

from rebuild.multimodel_state_invariant_fast import StateInvariantFinalResolverFast


class AttributeAwareResolver(StateInvariantFinalResolverFast):
    """V6 resolver with visibility-aware clothing and head evidence."""

    VIEWS = ("full", "upper", "torso", "lower", "attributes")

    def __init__(self, cfg, registry=None):
        super().__init__(cfg, registry=registry)
        self.attr_color = float(cfg.get("attribute_color_min", 0.84))
        self.attr_pattern = float(cfg.get("attribute_pattern_min", 0.64))
        self.attr_detail = float(cfg.get("attribute_detail_min", 0.72))
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
        return {
            "ready": True,
            "full": bool(upvis and lowvis),
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
                result.setdefault(model, [])
                for view, values in views.items():
                    if view != "attributes":
                        result[model].extend(values)
        return result

    def pair(self, left, right, left_pool, right_pool):
        evidence = super().pair(left, right, left_pool, right_pool)
        self._pending = self._attrs(left, right)
        return evidence

    def _accept(self, evidence, same=False):
        good = super()._same_accept(evidence) if same else super()._cross_accept(evidence)
        if not good:
            return False
        attr = self._pending
        if not attr.get("ready", False):
            return True
        if attr["full"]:
            cloth = attr["cloth"] >= self.attr_color
            pattern = min(attr["upperpattern"], attr["lowerpattern"]) >= self.attr_pattern
            return bool(cloth and pattern)
        detail = attr["detail"] >= self.attr_detail
        return bool(detail and (attr["headvis"] or attr["eyevis"] or attr["upperpattern"] >= self.attr_pattern or attr["lowerpattern"] >= self.attr_pattern))

    def _same_accept(self, evidence):
        return self._accept(evidence, True)

    def _cross_accept(self, evidence):
        return self._accept(evidence, False)
