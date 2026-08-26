from __future__ import annotations

import numpy as np

from rebuild.identity_v3 import Feature, Tracklet
from rebuild.v6_local_global import LocalGlobalResolver


def vec(index: int, dim: int = 256) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[index] = 1.0
    return out


def track(camera: str, tid: int, vector: np.ndarray, stamp: float = 10.0) -> Tracklet:
    item = Tracklet(camera, tid, 1, 20.0)
    meta = {
        "camera": camera,
        "timestamp": stamp,
        "frame": int(stamp * 20),
        "track_id": tid,
        "bbox": [0, 0, 100, 220],
        "detection_score": 0.95,
        "quality": 0.95,
        "kind": "full",
    }
    item.add(vector, "full", 0.95, meta, 0.985, 8)
    item.add(vector, "light", 0.94, {**meta, "kind": "light"}, 0.985, 8)
    return item


def resolver() -> LocalGlobalResolver:
    cfg = {
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
        "camera_time_offsets_sec": {},
    }
    return LocalGlobalResolver(cfg)


def test_camera_permutation_is_resolved_without_global_gid_input() -> None:
    # 222 local G1=brown/G2=white. 224 local G1=white/G2=brown.
    tracks = {
        "cam_222:1:1": track("cam_222", 1, vec(0)),
        "cam_222:2:1": track("cam_222", 2, vec(1)),
        "cam_224:1:1": track("cam_224", 1, vec(1)),
        "cam_224:2:1": track("cam_224", 2, vec(0)),
    }
    local = {
        "cam_222:1:1": "G000001",
        "cam_222:2:1": "G000002",
        "cam_224:1:1": "G000001",
        "cam_224:2:1": "G000002",
    }
    result, components, edges = resolver().resolve(local, tracks, ["cam_222", "cam_224"])
    assert result["cam_222:1:1"] == result["cam_224:2:1"]
    assert result["cam_222:2:1"] == result["cam_224:1:1"]
    assert result["cam_222:1:1"] != result["cam_222:2:1"]
    assert len(edges) == 2
    assert len(components) == 2


def test_same_camera_identities_never_merge() -> None:
    tracks = {
        "cam_224:1:1": track("cam_224", 1, vec(0)),
        "cam_224:2:1": track("cam_224", 2, vec(1)),
        "cam_222:1:1": track("cam_222", 1, vec(0)),
        "cam_222:2:1": track("cam_222", 2, vec(1)),
    }
    local = {
        "cam_224:1:1": "G000001",
        "cam_224:2:1": "G000002",
        "cam_222:1:1": "G000001",
        "cam_222:2:1": "G000002",
    }
    result, components, _ = resolver().resolve(local, tracks, ["cam_222", "cam_224"])
    assert result["cam_224:1:1"] != result["cam_224:2:1"]
    for members in components.values():
        cameras = {key.split("::", 1)[0] for key in members}
        assert len(cameras) == len(members)


def test_weak_cross_camera_case_stays_separate() -> None:
    tracks = {
        "cam_222:1:1": track("cam_222", 1, vec(0)),
        "cam_224:1:1": track("cam_224", 1, vec(2)),
    }
    local = {
        "cam_222:1:1": "G000001",
        "cam_224:1:1": "G000001",
    }
    result, components, edges = resolver().resolve(local, tracks, ["cam_222", "cam_224"])
    assert result["cam_222:1:1"] != result["cam_224:1:1"]
    assert edges == []
    assert len(components) == 2


def test_camera_local_labels_do_not_control_global_labels() -> None:
    tracks = {
        "cam_213:8:1": track("cam_213", 8, vec(3), 1.0),
        "cam_224:3:1": track("cam_224", 3, vec(3), 1.1),
    }
    local = {
        "cam_213:8:1": "G000099",
        "cam_224:3:1": "G000001",
    }
    result, _, edges = resolver().resolve(local, tracks, ["cam_213", "cam_224"])
    assert result["cam_213:8:1"] == result["cam_224:3:1"]
    assert edges
    assert result["cam_213:8:1"].startswith("G")
