from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from rebuild.identity_body_v6 import GlobalIdentityBodyV6
from rebuild.identity_v3 import Feature, Identity, Tracklet


class GlobalIdentityBodyV6Verified(GlobalIdentityBodyV6):
    """V6 matcher with a conservative cross-camera verification tie-breaker.

    The proven V6 appearance matcher remains authoritative. This subclass only
    adds evidence when a cross-camera candidate is already close in the V6
    ranking. It never lowers the V6 acceptance thresholds and never lets
    geometry or secondary evidence create an otherwise weak identity match.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cross_margin = float(cfg.get("cross_camera_tie_margin", 0.025))
        self.cross_bonus = float(cfg.get("cross_camera_consensus_bonus", 0.012))
        self.cross_bonus_strong = float(cfg.get("cross_camera_consensus_bonus_strong", 0.020))
        self.cross_consensus_threshold = float(cfg.get("cross_camera_consensus_threshold", 0.64))
        self.cross_consensus_strong = float(cfg.get("cross_camera_consensus_strong", 0.70))
        self.cross_geometry_bonus = float(cfg.get("cross_camera_geometry_bonus", 0.008))
        self.cross_geometry_min = float(cfg.get("cross_camera_geometry_min", 0.68))

    @staticmethod
    def _compatible(query: str, gallery: str) -> bool:
        if query == gallery:
            return True
        if query in {"upper", "lower"} and gallery in {"full", "light"}:
            return True
        if gallery in {"upper", "lower"} and query in {"full", "light"}:
            return True
        return {query, gallery} <= {"full", "light"}

    def consensus(self, query: List[Feature], gallery: List[Feature]) -> Tuple[int, int, float]:
        """Return distinct strong query observations, kind diversity and mean top evidence."""
        if not query or not gallery:
            return 0, 0, 0.0

        by_kind: Dict[str, List[Feature]] = {}
        for item in gallery:
            by_kind.setdefault(item.kind, []).append(item)

        bests: List[Tuple[str, float]] = []
        for item in query:
            candidates: List[float] = []
            for kind, values in by_kind.items():
                if not self._compatible(item.kind, kind):
                    continue
                candidates.extend(float(np.dot(item.vector, other.vector)) for other in values)
            if candidates:
                bests.append((item.kind, max(candidates)))

        if not bests:
            return 0, 0, 0.0

        strong_items = [(kind, value) for kind, value in bests if value >= self.cross_consensus_threshold]
        strong_count = len(strong_items)
        diversity = len({kind for kind, _ in strong_items})
        values = sorted((value for _, value in bests), reverse=True)
        top = values[: min(3, len(values))]
        mean_top = float(np.mean(top)) if top else 0.0
        return strong_count, diversity, mean_top

    @staticmethod
    def _shape_support(query: Tracklet, identity: Identity) -> float:
        if query.shape <= 0 or not identity.geometry:
            return 0.5
        left = float(query.shape)
        right = float(np.median(identity.geometry))
        ratio = min(left, right) / max(left, right)
        return float(max(0.0, min(1.0, ratio)))

    def rank(self, track: Tracklet, tracks: Dict[str, Tracklet]) -> List[dict]:
        rows = super().rank(track, tracks)
        if not rows:
            return rows

        # Only use the extra verifier when this is genuinely a cross-camera
        # candidate. Same-camera V6 behavior remains untouched.
        adjusted: List[dict] = []
        for row in rows:
            identity = self.identities[row["gid"]]
            cross = track.camera not in identity.cameras
            row = dict(row)
            row["cross_camera"] = bool(cross)
            row["cross_consensus"] = 0
            row["cross_kind_diversity"] = 0
            row["cross_consensus_mean"] = 0.0
            row["cross_geometry"] = 0.5
            if not cross:
                adjusted.append(row)
                continue

            count, diversity, mean_top = self.consensus(track.features, identity.values())
            geom = self._shape_support(track, identity)
            row["cross_consensus"] = int(count)
            row["cross_kind_diversity"] = int(diversity)
            row["cross_consensus_mean"] = float(mean_top)
            row["cross_geometry"] = float(geom)

            # Never change a strong V6 match. The additional evidence only
            # separates close candidates in the ambiguous region.
            close = row["body"] < self.strong
            if close:
                if count >= 3 and diversity >= 2 and mean_top >= self.cross_consensus_strong:
                    row["score"] += self.cross_bonus_strong
                elif count >= 2 and mean_top >= self.cross_consensus_threshold:
                    row["score"] += self.cross_bonus
                if geom >= self.cross_geometry_min and row["body"] >= self.threshold - 0.02:
                    row["score"] += self.cross_geometry_bonus
            adjusted.append(row)

        return sorted(adjusted, key=lambda x: x["score"], reverse=True)

    def record(self, track: Tracklet, gid: str, state: str, reason: str, row: dict | None, provisional: str, merged: bool = False) -> None:
        # Keep the original V6 decision schema. Diagnostics are emitted through
        # the standard debug JSON and do not affect compatibility with renderers.
        super().record(track, gid, state, reason, row, provisional, merged)
