from __future__ import annotations

from typing import Dict

import numpy as np

from rebuild import batch_state_invariant as state_pipeline
from rebuild.batch_state_invariant_joint_attributes import BatchPipelineStateInvariantJointAttributes
from rebuild.qdrant_authoritative import QdrantAuthoritativeResolver


class BatchPipelineStateAuthoritative(BatchPipelineStateInvariantJointAttributes):
    """Final recorded-video path for accuracy-first MTMC person ReID.

    The existing detector/tracker and three-model feature extraction are kept.
    This wrapper changes only the identity boundary: overlap uses hysteresis,
    Qdrant provides identity retrieval, and unresolved observations are labeled
    UNKNOWN/PENDING rather than being converted into tracker-derived GIDs.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        guard = self.cfg.get("overlap_guard", {}) or {}
        self.overlap_enter = float(guard.get("iou_min", 0.70))
        self.overlap_intersection_enter = float(guard.get("intersection_min", 0.70))
        self.overlap_exit = float(self.cfg.get("authoritative_overlap_exit", 0.58))
        self.overlap_grace = max(1, int(guard.get("clear_grace_frames", 2)))
        self._overlap_pairs: Dict[tuple[str, str], Dict[str, float]] = {}
        self._authoritative_resolver = None

        # The parent state pipeline owns the persistent metadata registry.
        # Replace only the resolver class used by its Pass 3.
        state_pipeline.StateInvariantFinalResolver = QdrantAuthoritativeResolver

    @staticmethod
    def _metrics(left, right):
        ax1, ay1, ax2, ay2 = [float(v) for v in left]
        bx1, by1, bx2, by2 = [float(v) for v in right]
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter <= 0.0:
            return 0.0, 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if area_a <= 0.0 or area_b <= 0.0:
            return 0.0, 0.0
        union = area_a + area_b - inter
        iou = inter / union if union > 0.0 else 0.0
        iom = inter / min(area_a, area_b)
        return float(iou), float(iom)

    def _overlaps(self, items):
        blocked = set()
        partners = {}
        seen_pairs = set()

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                left = items[i]
                right = items[j]
                a = str(left["key"])
                b = str(right["key"])
                pair = tuple(sorted((a, b)))
                seen_pairs.add(pair)

                iou, iom = self._metrics(
                    left["bbox"],
                    right["bbox"],
                )
                signal = max(iou, iom)
                state = self._overlap_pairs.get(pair)

                if state is None:
                    active = bool(
                        iou >= self.overlap_enter
                        or iom >= self.overlap_intersection_enter
                    )
                    self._overlap_pairs[pair] = {
                        "active": float(active),
                        "clear": 0.0,
                    }
                elif bool(state["active"]):
                    if signal <= self.overlap_exit:
                        state["clear"] += 1.0
                        if state["clear"] >= self.overlap_grace:
                            state["active"] = 0.0
                            state["clear"] = 0.0
                    else:
                        state["clear"] = 0.0
                    active = bool(state["active"])
                else:
                    active = bool(
                        iou >= self.overlap_enter
                        or iom >= self.overlap_intersection_enter
                    )
                    if active:
                        state["active"] = 1.0
                        state["clear"] = 0.0

                if active:
                    blocked.add(a)
                    blocked.add(b)
                    partners.setdefault(a, []).append(b)
                    partners.setdefault(b, []).append(a)

        # Expire pairs which have disappeared. They are not used to block an
        # unrelated future pair because tracker keys are unique for a segment.
        for pair in list(self._overlap_pairs):
            if pair not in seen_pairs:
                self._overlap_pairs.pop(pair, None)

        return blocked, partners

    @staticmethod
    def label(mapping, key, overlap=False, recovery=False):
        value = mapping.get(key)
        if value is None:
            return "PENDING" if (overlap or recovery) else "UNKNOWN"
        text = str(value)
        if text == "PENDING":
            return "PENDING"
        if text == "UNKNOWN":
            return "UNKNOWN"
        if text.startswith("G") and text[1:].isdigit():
            return text
        return "UNKNOWN"


__all__ = ["BatchPipelineStateAuthoritative"]
