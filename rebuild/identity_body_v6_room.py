from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from rebuild.identity_body_v6_crosslink import GlobalIdentityBodyV6CrossLink
from rebuild.identity_v3 import Tracklet


class GlobalIdentityBodyV6Room(GlobalIdentityBodyV6CrossLink):
    """V6 cross-camera verifier with optional fixed-room zone corroboration.

    The cameras in this experiment watch the same physical workspace. For the
    persistent seated subjects, image appearance can be ambiguous while the
    physical seat/room position is highly discriminative. This layer uses
    calibrated normalized seat anchors only when a track is spatially stable.
    Moving tracks receive no room-zone evidence.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.zone_weight = float(cfg.get("room_zone_pair_weight", 0.22))
        self.zone_conflict_penalty = float(cfg.get("room_zone_conflict_penalty", 0.18))
        self.zone_radius = float(cfg.get("room_zone_radius_norm", 0.10))
        self.zone_stability = float(cfg.get("room_zone_stability_ratio", 1.20))
        self.room_anchors = {
            str(camera): {
                str(zone): (float(point[0]), float(point[1]))
                for zone, point in values.items()
            }
            for camera, values in (cfg.get("room_zone_anchors", {}) or {}).items()
        }
        self.room_maps = {
            (str(a), str(b)): {str(k): str(v) for k, v in mapping.items()}
            for a, values in (cfg.get("room_zone_maps", {}) or {}).items()
            for b, mapping in values.items()
        }

    def _stable_point(self, track: Tracklet) -> Tuple[float, float] | None:
        if not track.observations:
            return None
        rows = track.observations
        centers = []
        scales = []
        for row in rows:
            x1, y1, x2, y2 = [float(v) for v in row.get("bbox", [0, 0, 0, 0])]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            centers.append(((x1 + x2) * 0.5, y2))
            scales.append(h)
        points = np.asarray(centers, dtype=np.float32)
        scale = float(np.median(scales))
        if scale <= 0.0:
            return None
        spread = float(np.median(np.linalg.norm(points - np.median(points, axis=0), axis=1)) / scale)
        if spread > self.zone_stability:
            return None
        width = max(1.0, float(track.observations[-1].get("frame_width", 2560)))
        height = max(1.0, float(track.observations[-1].get("frame_height", 1440)))
        point = np.median(points, axis=0)
        return float(point[0] / width), float(point[1] / height)

    def _zone(self, track: Tracklet) -> str | None:
        point = self._stable_point(track)
        anchors = self.room_anchors.get(track.camera)
        if point is None or not anchors:
            return None
        best = None
        distance = float("inf")
        for zone, anchor in anchors.items():
            d = float(np.hypot(point[0] - anchor[0], point[1] - anchor[1]))
            if d < distance:
                distance = d
                best = zone
        return best if best is not None and distance <= self.zone_radius else None

    def _zone_pair(self, left: Tracklet, right: Tracklet) -> float:
        mapping = self.room_maps.get((left.camera, right.camera))
        if mapping is None:
            inverse = self.room_maps.get((right.camera, left.camera))
            if inverse is None:
                return 0.0
            mapping = {value: key for key, value in inverse.items()}
        left_zone = self._zone(left)
        right_zone = self._zone(right)
        if left_zone is None or right_zone is None:
            return 0.0
        expected = mapping.get(left_zone)
        if expected is None:
            return 0.0
        return 1.0 if expected == right_zone else -1.0

    def _pair_score(self, left: Tracklet, right: Tracklet) -> Tuple[float, float]:
        score, body = super()._pair_score(left, right)
        zone = self._zone_pair(left, right)
        if zone > 0.0:
            score += self.zone_weight
        elif zone < 0.0:
            score -= self.zone_conflict_penalty
        return float(score), float(body)
