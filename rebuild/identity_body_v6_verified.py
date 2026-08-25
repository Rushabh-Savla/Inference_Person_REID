from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from rebuild.identity_body_v6 import GlobalIdentityBodyV6
from rebuild.identity_v3 import Feature, Identity, Tracklet


class GlobalIdentityBodyV6Verified(GlobalIdentityBodyV6):
    """V6 matcher with order-independent cross-camera identity verification.

    The protected NVIDIA V6 appearance matcher remains the primary signal.
    Cross-camera reasoning is isolated here and uses synchronized track
    pairing computed from the complete set of tracklets before identity
    assignment. This avoids the previous online-order dependency where a
    candidate camera could be processed before the corroborating camera.
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
        self.cross_temporal_threshold = float(cfg.get("cross_camera_temporal_threshold", 0.56))
        self.cross_temporal_strong = float(cfg.get("cross_camera_temporal_strong", 0.66))
        self.cross_temporal_bonus = float(cfg.get("cross_camera_temporal_bonus", 0.045))
        self.cross_temporal_strong_bonus = float(cfg.get("cross_camera_temporal_strong_bonus", 0.065))
        self.cross_temporal_conflict_threshold = float(cfg.get("cross_camera_temporal_conflict_threshold", 0.45))
        self.cross_temporal_conflict_penalty = float(cfg.get("cross_camera_temporal_conflict_penalty", 0.050))

        # New order-independent linkage layer.
        self.cross_link_min_body = float(cfg.get("cross_camera_link_min_body", 0.50))
        self.cross_link_min_score = float(cfg.get("cross_camera_link_min_score", 0.53))
        self.cross_link_margin = float(cfg.get("cross_camera_link_margin", 0.035))
        self.cross_link_bonus = float(cfg.get("cross_camera_link_bonus", 0.16))
        self.cross_link_strong_bonus = float(cfg.get("cross_camera_link_strong_bonus", 0.22))
        self.cross_link_conflict_penalty = float(cfg.get("cross_camera_link_conflict_penalty", 0.18))
        self.cross_link_temporal_weight = float(cfg.get("cross_camera_link_temporal_weight", 0.12))
        self.cross_link_geometry_weight = float(cfg.get("cross_camera_link_geometry_weight", 0.06))
        self.camera_offsets = {
            str(k): float(v)
            for k, v in (cfg.get("camera_time_offsets_sec", {}) or {}).items()
        }
        self.camera_priority = {
            str(k): int(v)
            for k, v in (cfg.get("camera_processing_priority", {}) or {}).items()
        }
        self.cross_links: Dict[str, dict] = {}

    @staticmethod
    def _compatible(query: str, gallery: str) -> bool:
        if query == gallery:
            return True
        if query in {"upper", "lower"} and gallery in {"full", "light"}:
            return True
        if gallery in {"upper", "lower"} and query in {"full", "light"}:
            return True
        return {query, gallery} <= {"full", "light"}

    def _span(self, track: Tracklet) -> Tuple[float, float]:
        offset = self.camera_offsets.get(track.camera, 0.0)
        return float(track.start + offset), float(track.end + offset)

    def _gap(self, left: Tracklet, right: Tracklet) -> float:
        a0, a1 = self._span(left)
        b0, b1 = self._span(right)
        if a1 < b0:
            return float(b0 - a1)
        if b1 < a0:
            return float(a0 - b1)
        return 0.0

    def _alignment(self, left: Tracklet, right: Tracklet) -> float:
        if not self.cross_temporal_enabled:
            return 0.0
        gap = self._gap(left, right)
        if gap > self.cross_temporal_tolerance:
            return 0.0
        return float(max(0.0, 1.0 - gap / max(self.cross_temporal_tolerance, 1e-6)))

    @staticmethod
    def _shape_pair(left: Tracklet, right: Tracklet) -> float:
        if left.shape <= 0 or right.shape <= 0:
            return 0.5
        ratio = min(float(left.shape), float(right.shape)) / max(float(left.shape), float(right.shape))
        return float(max(0.0, min(1.0, ratio)))

    def _pair_score(self, left: Tracklet, right: Tracklet) -> Tuple[float, float]:
        body, _, _ = self.body_score(left.features, right.features)
        alignment = self._alignment(left, right)
        shape = self._shape_pair(left, right)
        score = (
            0.82 * body
            + self.cross_link_temporal_weight * alignment
            + self.cross_link_geometry_weight * shape
        )
        return float(score), float(body)

    @staticmethod
    def _maximum_matching(left: List[Tracklet], right: List[Tracklet], matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Maximum-weight one-to-one assignment with a scipy fast path and greedy fallback."""
        if not left or not right:
            return []
        try:
            from scipy.optimize import linear_sum_assignment

            rows = len(left)
            cols = len(right)
            size = max(rows, cols)
            padded = np.zeros((size, size), dtype=np.float64)
            padded[:rows, :cols] = matrix
            rr, cc = linear_sum_assignment(-padded)
            return [(int(r), int(c)) for r, c in zip(rr, cc) if r < rows and c < cols and matrix[r, c] > 0.0]
        except Exception:
            edges = [
                (float(matrix[i, j]), i, j)
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])
                if matrix[i, j] > 0.0
            ]
            edges.sort(reverse=True)
            used_left, used_right = set(), set()
            out: List[Tuple[int, int]] = []
            for _, i, j in edges:
                if i in used_left or j in used_right:
                    continue
                used_left.add(i)
                used_right.add(j)
                out.append((i, j))
            return out

    def _build_cross_links(self, tracks: Dict[str, Tracklet]) -> Dict[str, dict]:
        """Build order-independent, margin-checked cross-camera track correspondences."""
        if not self.cross_temporal_enabled:
            return {}

        cameras = sorted({track.camera for track in tracks.values()})
        links: Dict[str, dict] = {}
        for i, camera_a in enumerate(cameras):
            for camera_b in cameras[i + 1:]:
                left = [x for x in tracks.values() if x.camera == camera_a and x.count() > 0]
                right = [x for x in tracks.values() if x.camera == camera_b and x.count() > 0]
                if not left or not right:
                    continue

                matrix = np.zeros((len(left), len(right)), dtype=np.float64)
                bodies = np.zeros_like(matrix)
                for li, one in enumerate(left):
                    for ri, two in enumerate(right):
                        if self._gap(one, two) > self.cross_temporal_tolerance:
                            continue
                        score, body = self._pair_score(one, two)
                        if body < self.cross_link_min_body or score < self.cross_link_min_score:
                            continue
                        matrix[li, ri] = score
                        bodies[li, ri] = body

                assignments = self._maximum_matching(left, right, matrix)
                for li, ri in assignments:
                    score = float(matrix[li, ri])
                    body = float(bodies[li, ri])
                    if score < self.cross_link_min_score:
                        continue

                    row_values = np.sort(matrix[li, :][matrix[li, :] > 0.0])[::-1]
                    col_values = np.sort(matrix[:, ri][matrix[:, ri] > 0.0])[::-1]
                    row_margin = float(row_values[0] - row_values[1]) if len(row_values) > 1 else float("inf")
                    col_margin = float(col_values[0] - col_values[1]) if len(col_values) > 1 else float("inf")
                    if row_margin < self.cross_link_margin or col_margin < self.cross_link_margin:
                        continue

                    a, b = left[li], right[ri]
                    payload = {
                        "other": b.key,
                        "score": score,
                        "body": body,
                        "row_margin": row_margin,
                        "col_margin": col_margin,
                        "strong": bool(body >= self.cross_temporal_strong),
                    }
                    reverse = {
                        "other": a.key,
                        "score": score,
                        "body": body,
                        "row_margin": col_margin,
                        "col_margin": row_margin,
                        "strong": bool(body >= self.cross_temporal_strong),
                    }
                    links[a.key] = payload
                    links[b.key] = reverse
        return links

    def order_key(self, track: Tracklet) -> Tuple[float, int, str, str]:
        """Use the calibrated recording clock so anchor-camera evidence is available first."""
        start = float(track.start) + self.camera_offsets.get(track.camera, 0.0)
        priority = self.camera_priority.get(track.camera, 1000)
        return start, priority, track.camera, track.key

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
        return strong_count, diversity, float(np.mean(top)) if top else 0.0

    @staticmethod
    def _shape_support(query: Tracklet, identity: Identity) -> float:
        if query.shape <= 0 or not identity.geometry:
            return 0.5
        left = float(query.shape)
        right = float(np.median(identity.geometry))
        ratio = min(left, right) / max(left, right)
        return float(max(0.0, min(1.0, ratio)))

    def _temporal_evidence(self, query: Tracklet, identity: Identity, tracks: Dict[str, Tracklet]) -> Tuple[float, int, float]:
        if not self.cross_temporal_enabled:
            return 0.0, 0, 0.0
        matches: Dict[str, float] = {}
        conflicts: List[float] = []
        for key in identity.tracks:
            prior = tracks.get(key)
            if prior is None or prior.camera == query.camera:
                continue
            if self._gap(query, prior) > self.cross_temporal_tolerance:
                continue
            score, _, _ = self.body_score(query.features, prior.features)
            if score >= self.cross_temporal_threshold:
                matches[prior.camera] = max(matches.get(prior.camera, 0.0), float(score))
            elif score < self.cross_temporal_conflict_threshold:
                conflicts.append(float(score))
        return (
            float(max(matches.values()) if matches else 0.0),
            len(matches),
            float(max(conflicts) if conflicts else 0.0),
        )

    def rank(self, track: Tracklet, tracks: Dict[str, Tracklet]) -> List[dict]:
        rows = super().rank(track, tracks)
        if not rows:
            return rows

        link = self.cross_links.get(track.key)
        link_target = tracks.get(link["other"]) if link else None
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
            row["cross_link_score"] = 0.0
            row["cross_link_body"] = 0.0
            row["cross_link_conflict"] = False

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

            target_in_identity = bool(link_target is not None and link_target.key in identity.tracks)
            row["cross_link_score"] = float(link["score"]) if target_in_identity and link else 0.0
            row["cross_link_body"] = float(link["body"]) if target_in_identity and link else 0.0
            row["cross_link_conflict"] = bool(link and not target_in_identity)

            if row["body"] < self.strong:
                if count >= 3 and diversity >= 2 and mean_top >= self.cross_consensus_strong:
                    row["score"] += self.cross_bonus_strong
                elif count >= 2 and mean_top >= self.cross_consensus_threshold:
                    row["score"] += self.cross_bonus
                if geom >= self.cross_geometry_min and row["body"] >= self.threshold - 0.02:
                    row["score"] += self.cross_geometry_bonus
                if temporal >= self.cross_temporal_strong and temporal_cameras >= 1:
                    row["score"] += self.cross_temporal_strong_bonus
                elif temporal >= self.cross_temporal_threshold:
                    row["score"] += self.cross_temporal_bonus
                if temporal_conflict > 0.0 and temporal_conflict < self.cross_temporal_conflict_threshold:
                    row["score"] -= self.cross_temporal_conflict_penalty

                # Order-independent one-to-one corroboration is the dominant
                # cross-camera safeguard. A confident paired observation gets
                # a substantial boost; the wrong identity gets a penalty.
                if target_in_identity and link is not None:
                    if link["strong"] and link["score"] >= self.cross_link_min_score:
                        row["score"] += self.cross_link_strong_bonus
                    else:
                        row["score"] += self.cross_link_bonus
                elif link is not None:
                    row["score"] -= self.cross_link_conflict_penalty

            adjusted.append(row)
        return sorted(adjusted, key=lambda x: x["score"], reverse=True)

    def run(self, tracks: Dict[str, Tracklet]):
        # The complete track set is known before association, so cross-camera
        # correspondences can be computed without depending on processing order.
        self.cross_links = self._build_cross_links(tracks)
        return super().run(tracks)

    def summary(self, tracks: Dict[str, Tracklet]) -> dict:
        result = super().summary(tracks)
        result["cross_link_count"] = len(self.cross_links) // 2
        result["cross_link_strong"] = sum(1 for item in self.cross_links.values() if item["strong"]) // 2
        return result

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
