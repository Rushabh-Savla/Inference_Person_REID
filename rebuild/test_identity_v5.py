from __future__ import annotations

import numpy as np

from rebuild.identity_v3 import Feature, Tracklet
from rebuild.identity_v5 import GlobalIdentityV5


def feat(value: float, kind: str = "full", quality: float = 0.9, stamp: float = 0.0) -> Feature:
    vec = np.zeros(8, dtype=np.float32)
    vec[0] = value
    vec[1] = 1.0 - value
    vec /= np.linalg.norm(vec) + 1e-12
    return Feature(vec, kind, quality, "cam_a", stamp)


def track(key_id: int, values, camera="cam_a", start=0.0, end=1.0) -> Tracklet:
    t = Tracklet(camera, key_id, 1, 20.0, start=start, end=end)
    for idx, value in enumerate(values):
        t.features.append(feat(value, stamp=start + idx * 0.2))
    t.observations = [{"timestamp": t.start}, {"timestamp": t.end}]
    return t


def test_body_match():
    engine = GlobalIdentityV5({})
    first = track(1, [0.999, 0.998, 0.997])
    engine.assign(first, [], {first.key: first})
    second = track(2, [0.998, 0.997, 0.996], start=2.0, end=3.0)
    result = engine.assign(second, [], {first.key: first, second.key: second})
    assert result.gid == "G000001"
    assert result.reason in {"body_strong", "body_gallery", "body_accumulated"}


def test_face_rescue():
    engine = GlobalIdentityV5({"face_rescue": 0.80, "face_quality": 0.50, "face_strong_quality": 0.50})
    first = track(1, [0.80, 0.79, 0.78])
    faces = [Feature(np.array([1.0, 0, 0, 0], dtype=np.float32), "face", 0.9, "cam_a", 0.0)]
    engine.assign(first, faces, {first.key: first})
    second = track(2, [0.51, 0.50, 0.49], start=2.0, end=3.0)
    second_faces = [Feature(np.array([1.0, 0, 0, 0], dtype=np.float32), "face", 0.9, "cam_b", 2.0)]
    result = engine.assign(second, second_faces, {first.key: first, second.key: second})
    assert result.gid == "G000001"
    assert "face" in result.reason


def test_fragment_merge():
    engine = GlobalIdentityV5({"merge_body": 0.74, "merge_support": 2})
    left = track(1, [0.999, 0.998, 0.997], start=0.0, end=1.0)
    right = track(2, [0.997, 0.996, 0.995], start=5.0, end=6.0)
    third = track(3, [0.20, 0.25, 0.30], start=2.0, end=3.0)
    tracks = {x.key: x for x in [left, right, third]}
    engine.run(tracks, {})
    assert engine.mapping[left.key] == engine.mapping[right.key]
    assert engine.mapping[third.key] != engine.mapping[left.key]


if __name__ == "__main__":
    test_body_match()
    test_face_rescue()
    test_fragment_merge()
    print("IDENTITY V5 TEST: OK")
