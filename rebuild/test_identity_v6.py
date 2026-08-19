from __future__ import annotations

import numpy as np

from rebuild.identity_v3 import Feature, Tracklet
from rebuild.identity_v6 import GlobalIdentityV6


def feature(value: float, camera: str, stamp: float, kind: str = "full") -> Feature:
    vec = np.zeros(8, dtype=np.float32)
    vec[0] = value
    vec[1] = 1.0 - value
    vec /= np.linalg.norm(vec) + 1e-12
    return Feature(vec, kind, 0.95, camera, stamp)


def track(camera: str, track_id: int, start: float, end: float, value: float, x: float) -> Tracklet:
    item = Tracklet(camera, track_id, 1, 20.0, start=start, end=end)
    item.features = [feature(value, camera, start), feature(min(1.0, value + 0.001), camera, end)]
    item.observations = [
        {"timestamp": start, "bbox": [x, 100, x + 40, 200]},
        {"timestamp": end, "bbox": [x + 3, 100, x + 43, 200]},
    ]
    item.shape = 2.5
    return item


def test_same_camera_fragment():
    engine = GlobalIdentityV6({
        "body_strong": 0.70,
        "body_medium": 0.62,
        "same_camera_score": 0.50,
    })
    left = track("cam_213", 3, 0.0, 10.0, 0.999, 100)
    right = track("cam_213", 5, 10.5, 20.0, 0.998, 104)
    other = track("cam_213", 7, 11.0, 19.0, 0.20, 700)
    mapping, decisions = engine.run({x.key: x for x in [left, right, other]}, {})
    assert mapping[left.key] == mapping[right.key]
    assert mapping[other.key] != mapping[left.key]
    assert any("recent_lost_track" in item.reason for item in decisions if item.key == right.key)


def test_cross_camera_persistent_identity():
    engine = GlobalIdentityV6({"body_strong": 0.70, "body_medium": 0.62})
    first = track("cam_213", 3, 0.0, 5.0, 0.999, 100)
    second = track("cam_224", 9, 12.0, 18.0, 0.998, 400)
    different = track("cam_224", 10, 12.0, 18.0, 0.10, 900)
    mapping, _ = engine.run({x.key: x for x in [first, second, different]}, {})
    assert mapping[first.key] == mapping[second.key]
    assert mapping[different.key] != mapping[first.key]


def test_face_assisted_fragment():
    engine = GlobalIdentityV6({"body_strong": 0.70, "face_strong": 0.84})
    first = track("cam_213", 3, 0.0, 5.0, 0.60, 100)
    second = track("cam_213", 5, 6.0, 10.0, 0.60, 105)
    face = [Feature(np.array([1.0, 0, 0, 0], dtype=np.float32), "face", 0.95, "cam_213", 0.0)]
    faces = {
        first.key: face,
        second.key: [Feature(np.array([1.0, 0, 0, 0], dtype=np.float32), "face", 0.95, "cam_213", 6.0)],
    }
    mapping, _ = engine.run({x.key: x for x in [first, second]}, faces)
    assert mapping[first.key] == mapping[second.key]


if __name__ == "__main__":
    test_same_camera_fragment()
    test_cross_camera_persistent_identity()
    test_face_assisted_fragment()
    print("IDENTITY V6 TEST: OK")
