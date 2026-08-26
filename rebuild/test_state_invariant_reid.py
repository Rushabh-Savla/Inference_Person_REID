from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rebuild.multimodel_state_invariant_final import StateInvariantFinalResolver


def basis(index: int, dim: int = 8) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    value[index] = 1.0
    return value


def bank(person: np.ndarray):
    return {
        "resnet": {"full": [person], "upper": [person], "torso": [person], "lower": [person]},
        "swin": {"full": [person], "upper": [person], "torso": [person], "lower": [person]},
        "solider": {"full": [person], "upper": [person], "torso": [person], "lower": [person]},
    }


def make(camera, key, start, end, state_bank, colour, aspect=1.6, observations=None):
    return SimpleNamespace(
        key=key,
        camera=camera,
        start=float(start),
        end=float(end),
        state_bank=state_bank,
        colour_signature=np.asarray(colour, np.float32),
        shape=float(aspect),
        state_type="upright" if aspect >= 1.35 else "compact",
        observations=observations or [],
    )


def obs(t, x, y=100.0, h=180.0, w=70.0):
    return {
        "timestamp": float(t),
        "bbox": [float(x), float(y), float(x + w), float(y + h)],
    }


def test_standing_to_sitting_uses_upper_torso_consensus():
    resolver = StateInvariantFinalResolver({})
    person = basis(0)
    left = make("cam_222", "a", 10, 20, bank(person), [1, 0, 0, 0], aspect=1.8)
    right_bank = {
        "resnet": {"upper": [person], "torso": [person]},
        "swin": {"upper": [person], "torso": [person]},
        "solider": {"upper": [person], "torso": [person]},
    }
    right = make("cam_224", "b", 10, 20, right_bank, [1, 0, 0, 0], aspect=0.9)
    pair = resolver.pair(left, right, [left], [right])
    assert pair.state_transition is True
    assert pair.model_support == 3
    assert pair.mutual_models == 3
    assert pair.fused >= 0.54


def test_different_people_are_not_merged_by_shared_posture():
    resolver = StateInvariantFinalResolver({})
    left = make("cam_222", "a", 0, 10, bank(basis(0)), [1, 0, 0, 0], aspect=0.9)
    right = make("cam_224", "b", 0, 10, bank(basis(1)), [0, 1, 0, 0], aspect=0.9)
    pair = resolver.pair(left, right, [left], [right])
    assert pair.model_support == 0


def test_same_camera_overlapping_local_tracks_are_split_into_separate_groups():
    tracks = {
        "cam_224:1:1": make("cam_224", "a", 0, 10, bank(basis(0)), [1, 0, 0, 0], observations=[obs(0, 100)]),
        "cam_224:2:1": make("cam_224", "b", 5, 15, bank(basis(0)), [1, 0, 0, 0], observations=[obs(5, 110)]),
    }
    groups = StateInvariantFinalResolver.build_groups({"cam_224:1:1": "G1", "cam_224:2:1": "G1"}, tracks)
    assert len(groups) == 2
    assert all(len(group.members) == 1 for group in groups)


def test_fragmented_same_camera_person_is_repaired_before_cross_camera():
    person = basis(0)
    tracks = {
        "cam_213:1:1": make("cam_213", "a", 0, 5, bank(person), [1, 0, 0, 0], observations=[obs(0, 100, h=190)]),
        "cam_213:2:1": make("cam_213", "b", 6, 10, bank(person), [1, 0, 0, 0], observations=[obs(6, 108, h=188)]),
        "cam_224:9:1": make("cam_224", "c", 12, 20, bank(person), [1, 0, 0, 0], observations=[obs(12, 100, h=188)]),
    }
    mapping, _components, _edges = StateInvariantFinalResolver({}).resolve(
        {"a": "G1", "b": "G2", "c": "G1"}, tracks, ["cam_213", "cam_224"]
    )
    assert mapping["a"] == mapping["b"]
    assert mapping["a"] == mapping["c"]


def test_same_camera_people_with_overlap_are_never_stitched():
    person = basis(0)
    tracks = {
        "a": make("cam_213", "a", 0, 8, bank(person), [1, 0, 0, 0], observations=[obs(0, 100)]),
        "b": make("cam_213", "b", 4, 12, bank(person), [1, 0, 0, 0], observations=[obs(4, 105)]),
    }
    resolver = StateInvariantFinalResolver({})
    groups = resolver.build_groups({"a": "G1", "b": "G2"}, tracks)
    edges = resolver.same_camera_edges(groups)
    assert edges == []


def test_cross_camera_timestamp_offset_is_not_required_for_strong_match():
    resolver = StateInvariantFinalResolver({})
    person = basis(0)
    left = make("cam_213", "a", 0, 5, bank(person), [1, 0, 0, 0])
    right = make("cam_224", "b", 40, 45, bank(person), [1, 0, 0, 0])
    pair = resolver.pair(left, right, [left], [right])
    edges = resolver.cross_edges([left], [right])
    assert pair.model_support == 3
    assert edges


def test_one_model_agreement_is_never_enough():
    resolver = StateInvariantFinalResolver({})
    a = basis(0)
    b = basis(1)
    left = make("cam_222", "a", 0, 10, {
        "resnet": {"full": [a]}, "swin": {"full": [a]}, "solider": {"full": [a]},
    }, [1, 0, 0, 0])
    right = make("cam_224", "b", 0, 10, {
        "resnet": {"full": [b]}, "swin": {"full": [a]}, "solider": {"full": [b]},
    }, [1, 0, 0, 0])
    pair = resolver.pair(left, right, [left], [right])
    assert pair.model_support == 1
    assert pair.mutual_models <= 1


def test_same_camera_is_hard_component_boundary():
    groups = [
        type("Group", (), {"key": "a", "camera": "cam_224", "members": ["a"], "start": 0.0})(),
        type("Group", (), {"key": "b", "camera": "cam_224", "members": ["b"], "start": 11.0})(),
    ]
    components = StateInvariantFinalResolver._components(groups, [{"left": "a", "right": "b", "fused": 0.99}])
    assert len(components) == 2
