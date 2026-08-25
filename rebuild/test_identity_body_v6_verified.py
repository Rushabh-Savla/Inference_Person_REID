from __future__ import annotations

import numpy as np

from rebuild.identity_body_v6_verified import GlobalIdentityBodyV6Verified
from rebuild.identity_v3 import Feature, Tracklet


def unit(value: float, dim: int = 8) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[0] = value
    out[1] = 1.0 - value
    return out / (np.linalg.norm(out) + 1e-12)


def feature(value: float, camera: str, stamp: float, kind: str = "full", quality: float = 0.95) -> Feature:
    return Feature(unit(value), kind, quality, camera, stamp)


def track(camera: str, track_id: int, start: float, value: float, x: float = 100.0, kinds=("full",)) -> Tracklet:
    item = Tracklet(camera, track_id, 1, 20.0, start=start, end=start + 4.0)
    item.features = [feature(value, camera, start, kind) for kind in kinds]
    item.observations = [
        {"timestamp": start, "bbox": [x, 100, x + 40, 200]},
        {"timestamp": start + 4.0, "bbox": [x + 2, 100, x + 42, 200]},
    ]
    item.shape = 2.5
    return item


def engine() -> GlobalIdentityBodyV6Verified:
    return GlobalIdentityBodyV6Verified({
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
    })


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
    ids = {x.key: x for x in [first, second]}
    e.identities = {}
    e.mapping = {}
    e.next_id = 1
    mapping, _ = e.run(ids)
    assert mapping[first.key] == mapping[second.key]


def test_partial_cross_camera_remains_supported() -> None:
    e = engine()
    first = track("cam_213", 1, 0.0, 0.99, kinds=("full", "upper"))
    partial = track("cam_224", 2, 12.0, 0.99, kinds=("upper",))
    mapping, _ = e.run({x.key: x for x in [first, partial]})
    assert mapping[first.key] == mapping[partial.key]
