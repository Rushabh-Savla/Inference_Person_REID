from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from rebuild.identity_body_v6 import GlobalIdentityBodyV6
from rebuild.identity_v3 import Feature, Identity, Tracklet


class GlobalIdentityBodyV6Verified(GlobalIdentityBodyV6):
    """Known-good V6 matcher with conservative cross-camera verification.

    The original V6 body-ReID matcher remains the primary signal. This layer
    adds three secondary signals only for cross-camera candidates:

    * multi-observation appearance consensus;
    * person-shape consistency;
    * time-aligned track evidence from the other cameras.

    Same-camera V6 matching is untouched. Extra evidence never creates an
    identity from a weak appearance score.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cross_margin = float(cfg.get("cross_camera_tie_margin", 0.025))
        self.cross_bonus = float(cfg.get("cross_camera_consensus_bonus", 0.015))
        self.cross_bonus_strong = float(cfg.get("cross_camera_consensus_bonus_strong", 0.028))
        self.cross_consensus_threshold = float(cfg.get("cross_camera_consensus_threshold", 0.64))
        self.cross_consensus_strong = float(cfg.get("cross_camera_consensus_strong", 0.70))
        self.cross_geometry_bonus = float(cfg.get("cross_camera_geometry_bonus", 0.004))
        self.cross_geometry_min = float(cfg.get("cross_camera_geometry_min", 0.70))
        self.cross_temporal_enabled = bool(cfg.get("cross_camera_temporal_enabled", True))
        self.cross_temporal_tolerance = float(cfg.get("cross_camera_temporal_tolerance_sec", 6.0))
        self.cross_temporal_bonus = float(cfg.get("cross_camera_temporal_bonus", 0.045))
        self.cross_temporal_strong_bonus = float(cfg.get("cross_camera_temporal_strong_bonus", 0.065))
        self.cross_temporal_threshold = float(cfg.get("cross_camera_temporal_threshold", 0.56))
        self.cross_temporal_strong = float(cfg.get("cross_camera_temporal_strong", 0.66))
        self.cross_temporal_conflict_threshold = float(cfg.get("cross_camera_temporal_conflict_threshold", 0.45))
        self.cross_temporal_conflict_penalty = float(cfg.get("cross_camera_temporal_conflict_penalty", 0.050))
        self.camera_offsets = {
            str(k): float(v)
            for k, v in (cfg.get("camera_time_offsets_sec", {}) or {}).items()
        }

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

    def _span(self, track: Tracklet) -> Tuple[float, float]:
        offset = self.camera_offsets.get(track.camera, 0.0)
        return float(track.start + offset), float(track.end + offset)

    def _aligned(self, query: Tracklet, prior: Tracklet) -> bool:
        if query.camera == prior.camera:
            return False
        q0, q1 = self._span(query)
        p0, p1 = self._span(prior)
        if q1 < p0:
            gap = p0 - q1
        elif p1 < q0:
            gap = q0 - p1
        else:
            gap = 0.0
        return gap <= self.cross_temporal_tolerance

    def _temporal_evidence(
        self,
        query: Tracklet,
        identity: Identity,
        tracks: Dict[str, Tracklet],
    ) -> Tuple[float, int, float]:
        """Best aligned-track score, supporting camera count and conflict score."""
        if not self.cross_temporal_enabled:
            return 0.0, 0, 0.0

        matches: Dict[str, float] = {}
        conflicts: List[float] = []
        for key in identity.tracks:
            prior = tracks.get(key)
            if prior is None or not self._aligned(query, prior):
                continue
            score, _, _ = self.body_score(query.features, prior.features)
            if score >= self.cross_temporal_threshold:
                matches[prior.camera] = max(matches.get(prior.camera, 0.0), float(score))
            elif score < self.cross_temporal_conflict_threshold:
                conflicts.append(float(score))

        best = max(matches.values()) if matches else 0.0
        conflict = max(conflicts) if conflicts else 0.0
        return float(best), len(matches), float(conflict)

    def rank(self, track: Tracklet, tracks: Dict[str, Tracklet]) -> List[dict]:
        rows = super().rank(track, tracks)
        if not rows:
            return rows

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
            row["cross_temporal_score"] = 0.0
            row["cross_temporal_cameras"] = 0
            row["cross_temporal_conflict"] = 0.0
            if not cross:
                adjusted.append(row)
                continue

            count, diversity, mean_top = self.consensus(track.features, identity.values())
            geom = self._shape_support(track, identity)
            temporal, temporal_cameras, temporal_conflict = self._temporal_evidence(track, identity, tracks)
            row["cross_consensus"] = int(count)
            row["cross_kind_diversity"] = int(diversity)
            row["cross_consensus_mean"] = float(mean_top)
            row["cross_geometry"] = float(geom)
            row["cross_temporal_score"] = float(temporal)
            row["cross_temporal_cameras"] = int(temporal_cameras)
            row["cross_temporal_conflict"] = float(temporal_conflict)

            # The original V6 decision remains the authority. Secondary
            # evidence is used only while the appearance score is not already
            # strong and never eliminates the original V6 acceptance rules.
            if row["body"] < self.strong:
                if count >= 3 and diversity >= 2 and mean_top >= self.cross_consensus_strong:
                    row["score"] += self.cross_bonus_strong
                elif count >= 2 and mean_top >= self.cross_consensus_threshold:
                    row["score"] += self.cross_bonus

                if geom >= self.cross_geometry_min and row["body"] >= self.threshold - 0.02:
                    row["score"] += self.cross_geometry_bonus

                # For the supplied synchronized recordings, this is the key
                # 222/224 protection: compare a query against the actual
                # simultaneously observed track behind each candidate GID.
                if temporal >= self.cross_temporal_strong and temporal_cameras >= 1:
                    row["score"] += self.cross_temporal_strong_bonus
                elif temporal >= self.cross_temporal_threshold:
                    row["score"] += self.cross_temporal_bonus

                # If a candidate GID has a time-aligned track that is strongly
                # inconsistent with the query, suppress that candidate. This
                # prevents one wrong camera assignment from propagating into
                # another camera.
                if temporal_conflict > 0.0 and temporal_conflict < self.cross_temporal_conflict_threshold:
                    row["score"] -= self.cross_temporal_conflict_penalty

            adjusted.append(row)

        return sorted(adjusted, key=lambda x: x["score"], reverse=True)

    def record(
        self,
        track: Tracklet,
        gid: str,
        state: str,
        reason: str,
        row: dict | None,
        provisional: str,
        merged: bool = False,
    ) -> None:
        super().record(track, gid, state, reason, row, provisional, merged)
