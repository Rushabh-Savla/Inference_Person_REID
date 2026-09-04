from __future__ import annotations

from typing import List

from rebuild.multimodel_state_invariant_attributes import AttributeAwareResolver


class AccurateAttributeAwareResolver(AttributeAwareResolver):
    """Strict feature-first resolver with chained same-camera repair.

    Overlap is never an identity decision. Same-camera repair is driven by the
    multimodel appearance evidence already produced by the V6 state banks.
    Every accepted edge is chronological, non-overlapping, and can participate
    in a longer A -> B -> C tracklet chain.
    """

    def same_camera_edges(self, groups: List[object]) -> List[dict]:
        ordered = sorted(groups, key=lambda group: (group.start, group.end, group.key))
        candidates = []
        for i, left in enumerate(ordered):
            for j in range(i + 1, len(ordered)):
                right = ordered[j]
                if right.start <= left.end:
                    continue
                gap = float(right.start - left.end)
                if gap > self.same_max_gap:
                    break
                evidence = self.pair(left, right, ordered, ordered)
                if not self._same_accept(evidence):
                    continue
                candidates.append(
                    {
                        "score": float(evidence.fused),
                        "left": left,
                        "right": right,
                        "evidence": evidence,
                        "gap": gap,
                    }
                )

        candidates.sort(
            key=lambda item: (
                item["score"],
                item["evidence"].agreement,
                item["evidence"].model_support,
                -item["gap"],
            ),
            reverse=True,
        )

        predecessor: set[str] = set()
        successor: set[str] = set()
        chosen = []

        # A tracklet may have one strong predecessor and one strong successor.
        # This fixes A->B->C fragmentation without allowing an identity to fan
        # out into multiple simultaneous alternatives.
        for item in candidates:
            left = item["left"]
            right = item["right"]
            if left.key in successor or right.key in predecessor:
                continue
            successor.add(left.key)
            predecessor.add(right.key)
            evidence = item["evidence"]
            chosen.append(
                {
                    "left": left.key,
                    "right": right.key,
                    "fused": float(evidence.fused),
                    "resnet": float(evidence.resnet),
                    "swin": float(evidence.swin),
                    "solider": float(evidence.solider),
                    "colour": float(evidence.colour),
                    "geometry": float(evidence.geometry),
                    "temporal": float(evidence.temporal),
                    "continuity": float(evidence.continuity),
                    "agreement": float(evidence.agreement),
                    "model_support": int(evidence.model_support),
                    "mutual_models": int(evidence.mutual_models),
                    "view_support": int(evidence.view_support),
                    "state_transition": bool(evidence.state_transition),
                }
            )

        return chosen
