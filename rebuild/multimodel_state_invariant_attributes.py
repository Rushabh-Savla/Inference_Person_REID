from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from rebuild.multimodel_state_invariant_fast import StateInvariantFinalResolverFast


class AttributeAwareResolver(StateInvariantFinalResolverFast):
    """V6 resolver with conservative visibility-aware appearance validation."""

    VIEWS = ("full", "upper", "torso", "lower", "attributes")

    def __init__(self, cfg, registry=None):
        super().__init__(cfg, registry=registry)
        self._attr = {}
        self.attr_color = float(cfg.get("attribute_color_min", 0.84))
        self.attr_pattern = float(cfg.get("attribute_pattern_min", 0.64))
        self.attr_detail = float(cfg.get("attribute_detail_min", 0.72))

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
    def _attrpack(cls, group):
        values = list(group.state_bank.get("resnet", {}).get("attributes", []))
        if not values:
            return None
        arr = [np.asarray(value, np.float32).reshape(-1) for value in values if value is not None]
        arr = [value for value in arr if value.size == 112 and np.isfinite(value).all()]
        if not arr:
            return None
        return np.stack(arr)

    @classmethod
    def _attrs(cls, left, right):
        a = cls._attrpack(left)
        b = cls._attrpack(right)
        if a is None or b is None:
            return {"ready": False, "full": False, "upper": 0.5, "lower": 0.5, "upperpattern": 0.5, "lowerpattern": 0.5, "head": 0.5, "eye": 0.5, "detail": 0.5, "cloth": 0.5}

        au = a[:, 0:20]
        al = a[:, 20:40]
        ap = a[:, 40:54]
        aq = a[:, 54:68]
        ah = a[:, 68:102]
        ae = a[:, 102:108]
        av = a[:, 108:112]
        bu = b[:, 0:20]
        bl = b[:, 20:40]
        bp = b[:, 40:54]
        bq = b[:, 54:68]
        bh = b[:, 68:102]
        be = b[:, 102:108]
        bv = b[:, 108:112]

        upper = float(np.mean(av[:, 0]) > 0.15 and np.mean(bv[:, 0]) > 0.15)
        lower = float(np.mean(av[:, 1]) > 0.15 and np.mean(bv[:, 1]) > 0.15)
        head = float(np.mean(av[:, 2]) > 0.15 and np.mean(bv[:, 2]) > 0.15)
        eye = float(np.mean(av[:, 3]) > 0.15 and np.mean(bv[:, 3]) > 0.15)
        colorup = cls._best(au, bu)
        colorlow = cls._best(al, bl)
        patternup = cls._best(ap, bp)
        patternlow = cls._best(aq, bq)
        headscore = cls._best(ah, bh)
        eyescore = cls._best(ae, be)
        detail = max(patternup, patternlow, headscore, eyescore)
        full = bool(upper and lower)
        return {
            "ready": True,
            "full": full,
            "upper": colorup,
            "lower": colorlow,
            "upperpattern": patternup,
            "lowerpattern": patternlow,
            "head": headscore,
            "eye": eyescore,
            "detail": detail,
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
                    if view == "attributes":
                        continue
                    result[model].extend(values)
        return result

    def pair(self, left, right, left_pool, right_pool):
        evidence = super().pair(left, right, left_pool, right_pool)
        key = tuple(sorted((left.key, right.key)))
        self._attr[key] = self._attrs(left, right)
        return evidence

    def _gate(self):
        if not self._attr:
            return {"ready": False}
        return next(reversed(self._attr.values()))

    def _same_accept(self, evidence):
        if not super()._same_accept(evidence):
            return False
        attr = self._gate()
        if not attr.get("ready", False):
            return True
        if attr["full"]:
            return attr["cloth"] >= self.attr_color and min(attr["upperpattern"], attr["lowerpattern"]) >= self.attr_pattern
        return attr["detail"] >= self.attr_detail

    def _cross_accept(self, evidence):
        if not super()._cross_accept(evidence):
            return False
        attr = self._gate()
        if not attr.get("ready", False):
            return True
        if attr["full"]:
            return attr["cloth"] >= self.attr_color and min(attr["upperpattern"], attr["lowerpattern"]) >= self.attr_pattern
        return attr["detail"] >= self.attr_detail
