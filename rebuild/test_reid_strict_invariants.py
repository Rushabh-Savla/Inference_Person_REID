from __future__ import annotations

import numpy as np

from rebuild.assignment_guard import solve
from rebuild.batch_nvdcf import BatchNvDCF
from rebuild.multimodal_identity_strict import MultiModalStrict


def unit(index: int, size: int = 8) -> np.ndarray:
    value = np.zeros(size, np.float32)
    value[index % size] = 1.0
    return value


def attrs(top: int, bottom: int) -> np.ndarray:
    value = np.zeros(112, np.float32)
    value[top % 20] = 1.0
    value[20 + bottom % 20] = 1.0
    value[40 + top % 14] = 1.0
    value[54 + bottom % 14] = 1.0
    value[68 + top % 34] = 1.0
    value[102 + bottom % 6] = 1.0
    value[108:112] = 1.0
    return value / max(np.linalg.norm(value), 1e-12)


def face(index: int) -> dict:
    value = np.zeros(8, np.float32)
    value[index % 8] = 1.0
    return {"vector": value, "quality": 0.95, "visibility": 0.95, "valid": True}


def stub() -> MultiModalStrict:
    item = object.__new__(MultiModalStrict)
    item.models = ("resnet", "swin", "solider")
    item.icfg = {
        "model_min": 0.46, "top_min": 0.42, "bottom_min": 0.42,
        "face_min": 0.60, "face_quality_min": 0.50,
        "existing_min": 0.62, "margin": 0.025, "deep_min": 0.48,
        "recovery_min": 0.62, "recovery_margin": 0.025,
        "memory_quality_min": 0.45, "dummy_floor": 0.35,
    }
    item.fcfg = {"min_visibility": 0.68}
    item.stats = {
        "feature": 0, "recovery": 0, "recovery_match": 0,
        "recovery_feature_verified": 0, "new": 0, "pending": 0,
        "cross": 0, "duplicate": 0,
    }
    item.pro = {}
    item.new = lambda _obs: (_ for _ in ()).throw(
        AssertionError("new GID must not be created in recovery")
    )
    item.save = lambda _gid, _obs: None
    return item


def profile(index: int) -> dict:
    value = unit(index)
    return {
        "resnet": [value], "swin": [value], "solider": [value],
        "attributes": [attrs(index, index + 3)], "pose": [value],
        "face": [face(index)], "camera": {"cam_1"},
    }


def observation(index: int, track_id: int, hints) -> dict:
    value = unit(index)
    return {
        "camera": "cam_1", "time": 10.0,
        "row": {"camera": "cam_1", "track_id": track_id, "frame": 10},
        "resnet": value, "swin": value, "solider": value,
        "attributes": attrs(index, index + 3), "pose": value,
        "face": face(index), "quality": 0.95,
        "recovery_hints": list(hints),
    }


def test_post_overlap_recovery_is_feature_driven_not_tracker_driven():
    item = stub()
    item.pro = {1: profile(1), 2: profile(2)}
    first = observation(2, 401, hints=[1])
    second = observation(1, 402, hints=[2])
    sets = [
        {1: item.score(first, 1, {}), 2: item.score(first, 2, {})},
        {1: item.score(second, 1, {}), 2: item.score(second, 2, {})},
    ]
    result = item.assign([first, second], sets, commit=True, recovery=True)
    assert result == ["G000002", "G000001"]
    assert result[0] != result[1]
    assert item.stats["recovery_feature_verified"] == 2


def test_reliable_face_rejects_body_strong_but_face_wrong_identity():
    item = stub()
    item.pro = {1: profile(1), 2: profile(2)}
    current = observation(1, 77, hints=[])
    current["resnet"] = unit(2)
    current["swin"] = unit(2)
    current["solider"] = unit(2)
    assert item.score(current, 1, {}) is not None
    assert item.score(current, 2, {}) is None


def test_assignment_guard_is_strictly_one_to_one():
    rows = [
        {10: {"score": 0.95}, 11: {"score": 0.80}},
        {10: {"score": 0.94}, 11: {"score": 0.79}},
    ]
    chosen = solve(
        rows,
        lambda value, second:
            value["score"] >= 0.70 and value["score"] - second >= 0.02,
    )
    assert len(chosen) == 2
    assert len(set(chosen.values())) == 2


def test_recovery_hints_are_spatial_hypotheses_only():
    rows = [
        {"track_id": 401, "bbox": [100, 100, 170, 280]},
        {"track_id": 402, "bbox": [500, 100, 570, 280]},
    ]
    anchors = [
        {"bbox": [102, 102, 172, 282], "gid": "G000001"},
        {"bbox": [498, 102, 568, 282], "gid": "G000002"},
    ]
    hints = BatchNvDCF._recovery_hints(rows, anchors)
    assert hints[401][0] == 1
    assert hints[402][0] == 2


def test_tracker_id_change_does_not_change_identity_score():
    item = stub()
    item.pro = {1: profile(1)}
    a = observation(1, 7, hints=[])
    b = observation(1, 99, hints=[])
    assert item.score(a, 1, {})["score"] == item.score(b, 1, {})["score"]
