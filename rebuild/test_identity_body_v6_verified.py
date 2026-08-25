from __future__ import annotations

import numpy as np

from rebuild.identity_body_v6_verified import GlobalIdentityBodyV6Verified
from rebuild.identity_v3 import Feature, Tracklet


def unit(value: float, dim: int = 8) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[0] = value
    out[1] = 1.0 - value
    return out / (np.linalg.norm(out) + 1e-12)


def basis(index: int, dim: int = 8) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[index] = 1.0
    return out


def feature(vector, camera: str, stamp: float, kind: str = "full", quality: float = 0.95) -> Feature:
    return Feature(np.asarray(vector, dtype=np.float32), kind, quality, camera, stamp)


def track(camera: str, track_id: int, start: float, value: float, x: float = 100.0, kinds=("full",)) -> Tracklet:
    item = Tracklet(camera, track_id, 1, 20.0, start=start, end=start + 4.0)
    item.features = [Feature(unit(value), kind, 0.95, camera, start) for kind in kinds]
    item.observations = [
        {"timestamp": start, "bbox": [x, 100, x + 40, 200]},
        {"timestamp": start + 4.0, "bbox": [x + 2, 100, x + 42, 200]},
    ]
    item.shape = 2.5
    return item


def engine(**extra) -> GlobalIdentityBodyV6Verified:
    cfg = {
        "match_threshold": 0.60,
        "match_margin": 0.03,
        "strong_threshold": 0.70,
        "support_required": 2,
        "accumulated_body": 0.56,
        "accumulated_support": 3,
        "partial_threshold": 0.58,
        "partial_support": 2,
        "same_camera_gap_sec": 15.0,
        "same_camera_distance": 5.0,
        "cross_camera_tie_margin": 0.05,
        "cross_camera_consensus_threshold": 0.64,
        "cross_camera_consensus_strong": 0.70,
        "cross_camera_temporal_enabled": True,
        "cross_camera_temporal_tolerance_sec": 6.0,
        "cross_camera_temporal_bonus": 0.045,
        "cross_camera_temporal_strong_bonus": 0.065,
        "cross_camera_temporal_threshold": 0.56,
        "cross_camera_temporal_strong": 0.66,
        "cross_camera_temporal_conflict_threshold": 0.45,
        "cross_camera_temporal_conflict_penalty": 0.050,
        "camera_time_offsets_sec": {"cam_213": 8.0, "cam_222": 0.0, "cam_224": 3.0},
    }
    cfg.update(extra)
    return GlobalIdentityBodyV6Verified(cfg)


def test_strong_cross_camera_match_unchanged() -> None:
    e = engine()
    first = track("cam_213", 1, 0.0, 0.99, kinds=("full", "upper"))
    second = track("cam_224", 2, 12.0, 0.99, kinds=("full", "upper"))
    mapping, _ = e.run({x.key: x for x in [first, second]})
    assert mapping[first.key] == mapping[second.key]


def test_cross_camera_consensus_is_only_a_tiebreaker() -> None:
    e = engine()
    first = track("cam_213", 1, 0.0, 0.99, kinds=("full", "upper"))
    second = track("cam_224", 2, 12.0, 0.985, kinds=("full", "upper"))
    other = track("cam_224", 3, 12.0, 0.30, kinds=("full", "upper"))
    mapping, _ = e.run({x.key: x for x in [first, second, other]})
    assert mapping[first.key] == mapping[second.key]
    assert mapping[other.key] != mapping[first.key]


def test_same_camera_does_not_receive_cross_camera_bonus() -> None:
    e = engine()
    first = track("cam_213", 1, 0.0, 0.99, kinds=("full",))
    second = track("cam_213", 2, 12.0, 0.99, x=105.0, kinds=("full",))
    mapping, _ = e.run({x.key: x for x in [first, second]})
    assert mapping[first.key] == mapping[second.key]


def test_partial_cross_camera_remains_supported() -> None:
    e = engine()
    first = track("cam_213", 1, 0.0, 0.99, kinds=("full", "upper"))
    partial = track("cam_224", 2, 12.0, 0.99, kinds=("upper",))
    mapping, _ = e.run({x.key: x for x in [first, partial]})
    assert mapping[first.key] == mapping[partial.key]


def test_time_aligned_other_camera_track_can_disambiguate_224() -> None:
    e = engine()

    # cam_222 has two simultaneous, clearly different people. cam_224 starts
    # three seconds later in the recording, so query start=21 corresponds to
    # approximately absolute time 24 in this synthetic setup.
    a = Tracklet("cam_222", 1, 1, 20.0, start=20.0, end=24.0)
    a.features = [feature(basis(0), "cam_222", 20.0)]
    a.observations = [{"timestamp": 20.0, "bbox": [100, 100, 160, 220]}, {"timestamp": 24.0, "bbox": [105, 100, 165, 220]}]
    a.shape = 2.0

    b = Tracklet("cam_222", 2, 1, 20.0, start=20.0, end=24.0)
    b.features = [feature(basis(1), "cam_222", 20.0)]
    b.observations = [{"timestamp": 20.0, "bbox": [500, 100, 560, 220]}, {"timestamp": 24.0, "bbox": [505, 100, 565, 220]}]
    b.shape = 2.0

    query = Tracklet("cam_224", 3, 1, 20.0, start=21.0, end=25.0)
    query.features = [feature(basis(0), "cam_224", 21.0)]
    query.observations = [{"timestamp": 21.0, "bbox": [800, 100, 860, 220]}, {"timestamp": 25.0, "bbox": [805, 100, 865, 220]}]
    query.shape = 2.1

    mapping, decisions = e.run({x.key: x for x in [a, b, query]})
    assert mapping[a.key] == mapping[query.key]
    assert mapping[b.key] != mapping[query.key]
    decision = next(x for x in decisions if x.key == query.key)
    assert decision.gid == mapping[a.key]


if __name__ == "__main__":
    test_strong_cross_camera_match_unchanged()
    test_cross_camera_consensus_is_only_a_tiebreaker()
    test_same_camera_does_not_receive_cross_camera_bonus()
    test_partial_cross_camera_remains_supported()
    test_time_aligned_other_camera_track_can_disambiguate_224()
    print("V6 CROSS-CAMERA VERIFIED IDENTITY TEST: OK")
