from types import SimpleNamespace

import numpy as np

from rebuild.multimodal_identity_strict import MultiModalStrict


def stub():
    item = object.__new__(MultiModalStrict)
    item.icfg = {
        "model_min": 0.46,
        "top_min": 0.42,
        "bottom_min": 0.42,
        "face_min": 0.68,
        "face_score_min": 0.68,
        "face_margin": 0.02,
        "existing_min": 0.61,
        "margin": 0.025,
        "deep_min": 0.48,
        "recovery_min": 0.58,
        "recovery_margin": 0.018,
        "dummy_floor": 0.35,
        "new_max": 0.48,
        "new_deep_max": 0.54,
    }
    item.pro = {
        1: {"resnet": [], "swin": [], "solider": [], "attributes": [], "face": [], "pose": [], "camera": set()},
        2: {"resnet": [], "swin": [], "solider": [], "attributes": [], "face": [], "pose": [], "camera": set()},
    }
    item.stats = {"feature": 0, "recovery": 0, "recovery_match": 0, "face": 0, "face_match": 0, "new": 0, "pending": 0, "cross": 0, "duplicate": 0}
    item.save = lambda gid, obs: None
    item.new = lambda obs: (_ for _ in ()).throw(AssertionError("new identity allocated"))
    return item


def row(score, face=0.0, faceused=False, top=0.80, bottom=0.80, deep=0.80):
    return {
        "score": score,
        "deep": deep,
        "resnet": deep,
        "swin": deep,
        "solider": deep,
        "top": top,
        "bottom": bottom,
        "upper_pattern": 0.80,
        "lower_pattern": 0.80,
        "pose": 0.80,
        "face": face,
        "faceused": faceused,
    }


def obs():
    value = np.zeros(4, np.float32)
    value[0] = 1.0
    return {
        "camera": "cam_219",
        "time": 1.0,
        "row": {"camera": "cam_219", "track_id": 7},
        "resnet": value,
        "swin": value,
        "solider": value,
        "attributes": np.ones(112, np.float32),
        "pose": None,
        "face": None,
    }


def test_assignment_is_one_to_one_even_with_same_tracker_id():
    item = stub()
    values = [obs(), obs()]
    sets = [
        {1: row(0.92), 2: row(0.74)},
        {1: row(0.91), 2: row(0.73)},
    ]
    result = item.assign(values, sets, commit=True, recovery=False)
    assert len(result) == 2
    assert result[0] != result[1]
    assert result[0].startswith("G") and result[1].startswith("G")


def test_bottom_clothing_is_mandatory():
    item = stub()
    bad = row(0.90, top=0.90, bottom=0.20, deep=0.90)
    assert item.accept(bad, 0.40) is False


def test_top_clothing_is_mandatory():
    item = stub()
    bad = row(0.90, top=0.20, bottom=0.90, deep=0.90)
    assert item.accept(bad, 0.40) is False


def test_reliable_face_has_priority_and_requires_face_match():
    item = stub()
    good = row(0.82, face=0.90, faceused=True, top=0.80, bottom=0.80, deep=0.50)
    bad = row(0.90, face=0.55, faceused=True, top=0.90, bottom=0.90, deep=0.95)
    assert item.accept(good, 0.70) is True
    assert item.accept(bad, 0.20) is False


def test_recovery_never_creates_a_new_gid_from_weak_match():
    item = stub()
    values = [obs()]
    sets = [{1: row(0.50), 2: row(0.49)}]
    result = item.assign(values, sets, commit=True, recovery=True)
    assert result == ["PENDING"]
    assert item.stats["new"] == 0
