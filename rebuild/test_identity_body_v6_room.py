from __future__ import annotations

import numpy as np

from rebuild.identity_body_v6_room import GlobalIdentityBodyV6Room
from rebuild.identity_v3 import Feature, Tracklet


def unit(vector):
    value = np.asarray(vector, dtype=np.float32)
    return value / (np.linalg.norm(value) + 1e-12)


def make_track(camera, track_id, start, vector, point):
    x, y = point
    item = Tracklet(camera, track_id, 1, 20.0, start=start, end=start + 4.0)
    item.features = [Feature(unit(vector), "full", 0.95, camera, start)]
    item.observations = [
        {"timestamp": start, "bbox": [x - 20, y - 200, x + 20, y]},
        {"timestamp": start + 4.0, "bbox": [x - 18, y - 198, x + 22, y + 2]},
    ]
    item.shape = 2.5
    return item


def test_room_zone_can_correct_appearance_swap() -> None:
    cfg = {
        "match_threshold": 0.60,
        "match_margin": 0.03,
        "strong_threshold": 0.72,
        "support_required": 1,
        "accumulated_body": 0.56,
        "accumulated_support": 1,
        "partial_threshold": 0.58,
        "partial_support": 1,
        "same_camera_gap_sec": 15.0,
        "same_camera_distance": 5.0,
        "camera_time_offsets_sec": {"cam_222": 0.0, "cam_224": 3.0},
        "camera_processing_priority": {"cam_222": 0, "cam_224": 1},
        "cross_camera_temporal_enabled": True,
        "cross_camera_temporal_tolerance_sec": 6.0,
        "cross_camera_link_min_body": 0.50,
        "cross_camera_link_min_score": 0.50,
        "cross_camera_link_margin": 0.01,
        "cross_camera_link_bonus": 0.12,
        "cross_camera_link_strong_bonus": 0.16,
        "cross_camera_link_conflict_penalty": 0.12,
        "cross_camera_link_temporal_weight": 0.10,
        "cross_camera_link_geometry_weight": 0.02,
        "cross_camera_link_override_min": 0.50,
        "cross_camera_link_override_bonus": 0.20,
        "cross_camera_link_override_penalty": 0.20,
        "room_zone_pair_weight": 0.22,
        "room_zone_conflict_penalty": 0.18,
        "room_zone_radius_norm": 0.04,
        "room_zone_stability_ratio": 1.20,
        "room_zone_anchors": {
            "cam_222": {"white": [0.293, 0.229], "brown": [0.176, 0.556]},
            "cam_224": {"white": [0.438, 0.993], "brown": [0.176, 0.972]},
        },
        "room_zone_maps": {
            "cam_222": {"cam_224": {"white": "white", "brown": "brown"}}
        },
    }
    engine = GlobalIdentityBodyV6Room(cfg)

    # Deliberately make each 224 embedding look more like the opposite 222
    # identity. Appearance alone would swap them; room-zone correspondence
    # should recover the physical seat identity.
    white_222 = make_track("cam_222", 1, 20.0, [1.0, 0.0], (750, 330))
    brown_222 = make_track("cam_222", 2, 20.0, [0.0, 1.0], (450, 800))
    white_224 = make_track("cam_224", 3, 21.0, [0.60, 0.80], (1120, 1430))
    brown_224 = make_track("cam_224", 4, 21.0, [0.80, 0.60], (450, 1400))

    mapping, _ = engine.run({x.key: x for x in [white_222, brown_222, white_224, brown_224]})
    assert mapping[white_224.key] == mapping[white_222.key]
    assert mapping[brown_224.key] == mapping[brown_222.key]
    assert mapping[white_224.key] != mapping[brown_224.key]


if __name__ == "__main__":
    test_room_zone_can_correct_appearance_swap()
    print("ROOM GEOMETRY TEST: OK")
