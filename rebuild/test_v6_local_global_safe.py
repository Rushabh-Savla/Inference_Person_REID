from __future__ import annotations

import numpy as np

from rebuild.identity_v3 import Tracklet
from rebuild.v6_local_global_safe import SafeLocalGlobalResolver


def vector(index: int, dim: int = 256) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    value[index] = 1.0
    return value


def make_track(camera: str, track_id: int, index: int, start: float, end: float) -> Tracklet:
    track = Tracklet(camera, track_id, 1, 20.0)
    for stamp in (start, end):
        meta = {
            "camera": camera,
            "timestamp": stamp,
            "frame": int(stamp * 20),
            "track_id": track_id,
            "bbox": [0, 0, 100, 220],
            "detection_score": 0.95,
            "quality": 0.95,
            "kind": "full",
        }
        track.add(vector(index), "full", 0.95, meta, 0.985, 8)
        track.add(vector(index), "light", 0.94, {**meta, "kind": "light"}, 0.985, 8)
    track.start = start
    track.end = end
    track.shape = 2.2
    return track


def cfg() -> dict:
    return {
        "match_threshold": 0.60,
        "match_margin": 0.035,
        "strong_threshold": 0.72,
        "support_required": 2,
        "accumulated_body": 0.56,
        "accumulated_support": 3,
        "partial_threshold": 0.58,
        "partial_support": 2,
        "gallery": 16,
        "candidate_gallery": 8,
        "promote_quality": 0.68,
        "novelty": 0.985,
        "same_camera_gap_sec": 15.0,
        "same_camera_distance": 5.0,
        "same_camera_min_continuity": 0.35,
        "merge_body": 0.82,
        "merge_support": 3,
        "seed_count": 3,
        "cross_local_min_reid": 0.66,
        "cross_local_min_score": 0.67,
        "cross_local_margin": 0.04,
        "cross_local_strong_reid": 0.72,
        "cross_local_color_weight": 0.06,
        "cross_local_time_weight": 0.04,
        "cross_local_shape_weight": 0.02,
        "cross_local_time_tolerance_sec": 12.0,
    }


def test_overlapping_tracks_with_same_local_gid_are_split() -> None:
    tracks = {
        "cam_224:1:1": make_track("cam_224", 1, 0, 1.0, 8.0),
        "cam_224:2:1": make_track("cam_224", 2, 1, 2.0, 7.0),
        "cam_222:1:1": make_track("cam_222", 1, 0, 1.0, 8.0),
        "cam_222:2:1": make_track("cam_222", 2, 1, 2.0, 7.0),
    }
    local = {
        "cam_224:1:1": "G000001",
        "cam_224:2:1": "G000001",  # deliberately contaminated local V6 result
        "cam_222:1:1": "G000001",
        "cam_222:2:1": "G000002",
    }
    result, components, _ = SafeLocalGlobalResolver(cfg()).resolve(local, tracks, ["cam_222", "cam_224"])
    assert result["cam_224:1:1"] != result["cam_224:2:1"]
    for members in components.values():
        cameras = [key.split("::", 1)[0] for key in members]
        assert len(cameras) == len(set(cameras))


def test_clean_local_ids_remain_one_to_one() -> None:
    tracks = {
        "cam_222:1:1": make_track("cam_222", 1, 0, 1.0, 8.0),
        "cam_222:2:1": make_track("cam_222", 2, 1, 2.0, 7.0),
        "cam_224:1:1": make_track("cam_224", 1, 1, 1.0, 8.0),
        "cam_224:2:1": make_track("cam_224", 2, 0, 2.0, 7.0),
    }
    local = {
        "cam_222:1:1": "G000001",
        "cam_222:2:1": "G000002",
        "cam_224:1:1": "G000001",
        "cam_224:2:1": "G000002",
    }
    result, _, edges = SafeLocalGlobalResolver(cfg()).resolve(local, tracks, ["cam_222", "cam_224"])
    assert result["cam_222:1:1"] == result["cam_224:2:1"]
    assert result["cam_222:2:1"] == result["cam_224:1:1"]
    assert len(edges) == 2
