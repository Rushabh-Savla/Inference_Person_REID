from __future__ import annotations

import numpy as np

from rebuild.identity_v3 import Feature, Tracklet
from rebuild.identity_v6 import GlobalIdentityV6


def vec(value: float, dim: int = 8) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[0] = value
    out[1] = 1.0 - value
    return out / (np.linalg.norm(out) + 1e-12)


def feature(value: float, camera: str, stamp: float, kind: str = "full", quality: float = 0.95) -> Feature:
    return Feature(vec(value), kind, quality, camera, stamp)


def track(camera: str, track_id: int, start: float, end: float, value: float, x: float, upper: float | None = None) -> Tracklet:
    item = Tracklet(camera, track_id, 1, 20.0, start=start, end=end)
    item.features = [feature(value, camera, start)]
    if upper is not None:
        item.features.append(feature(upper, camera, start, "upper"))
    item.observations = [
        {"timestamp": start, "bbox": [x, 100, x + 40, 200]},
        {"timestamp": end, "bbox": [x + 3, 100, x + 43, 200]},
    ]
    item.shape = 2.5
    return item


def run(items, faces=None):
    engine = GlobalIdentityV6({
        "body_strong": 0.70,
        "body_medium": 0.61,
        "same_camera_body": 0.48,
        "partial_strong": 0.66,
        "same_camera_gap_sec": 15.0,
    })
    mapping, decisions = engine.run({x.key: x for x in items}, faces or {})
    return engine, mapping, decisions


def test_same_camera_fragment():
    left = track("cam_213", 3, 0.0, 10.0, 0.995, 100, upper=0.997)
    right = track("cam_213", 5, 10.5, 20.0, 0.82, 104, upper=0.997)
    other = track("cam_213", 7, 11.0, 19.0, 0.20, 700, upper=0.21)
    engine, mapping, decisions = run([left, right, other])
    assert mapping[left.key] == mapping[right.key]
    assert mapping[other.key] != mapping[left.key]
    reasons = [x.reason for x in decisions if x.key == right.key]
    assert any("lost_track" in x or "body" in x or "identity_merge" in x for x in reasons)
    assert engine.summary({x.key: x for x in [left, right, other]})["fragmented_identity_count"] <= 1


def test_leave_return_same_camera():
    first = track("cam_213", 3, 0.0, 5.0, 0.995, 100, upper=0.997)
    returned = track("cam_213", 8, 10.0, 16.0, 0.88, 118, upper=0.997)
    engine, mapping, _ = run([first, returned])
    assert mapping[first.key] == mapping[returned.key]
    assert engine.same_camera_reassociated >= 1 or len(set(mapping.values())) == 1


def test_cross_camera_search():
    first = track("cam_213", 3, 0.0, 5.0, 0.995, 100, upper=0.995)
    second = track("cam_224", 9, 12.0, 18.0, 0.995, 400, upper=0.995)
    different = track("cam_224", 10, 12.0, 18.0, 0.20, 900, upper=0.21)
    engine, mapping, _ = run([first, second, different])
    assert mapping[first.key] == mapping[second.key]
    assert mapping[different.key] != mapping[first.key]
    assert engine.cross_camera_reidentified >= 1 or len(set(mapping.values())) < 3


def test_partial_to_full():
    full = track("cam_213", 3, 0.0, 5.0, 0.995, 100, upper=0.995)
    upper_only = track("cam_224", 9, 8.0, 12.0, 0.40, 400, upper=0.995)
    engine, mapping, _ = run([full, upper_only])
    assert mapping[full.key] == mapping[upper_only.key]
    assert engine.summary({x.key: x for x in [full, upper_only]})["global_ids"] == 1


def test_face_rescue():
    first = track("cam_213", 3, 0.0, 5.0, 0.60, 100, upper=0.60)
    second = track("cam_213", 5, 6.0, 10.0, 0.58, 105, upper=0.58)
    faces = {
        first.key: [Feature(np.array([1.0, 0, 0, 0], dtype=np.float32), "face", 0.95, "cam_213", 0.0)],
        second.key: [Feature(np.array([1.0, 0, 0, 0], dtype=np.float32), "face", 0.95, "cam_213", 6.0)],
    }
    engine, mapping, _ = run([first, second], faces)
    assert mapping[first.key] == mapping[second.key]
    assert engine.face_matches >= 1 or len(set(mapping.values())) == 1


if __name__ == "__main__":
    test_same_camera_fragment()
    test_leave_return_same_camera()
    test_cross_camera_search()
    test_partial_to_full()
    test_face_rescue()
    print("IDENTITY V6 TEST: OK")
