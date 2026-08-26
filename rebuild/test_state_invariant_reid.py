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


def make(camera, key, start, end, state_bank, colour, aspect=1.6):
    return SimpleNamespace(
        camera=camera,
        start=float(start),
        end=float(end),
        state_bank=state_bank,
        colour_signature=np.asarray(colour, np.float32),
        shape=float(aspect),
    )


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
    assert pair.fused >= 0.62


def test_different_people_are_not_merged_by_shared_posture():
    resolver = StateInvariantFinalResolver({})
    left = make("cam_222", "a", 0, 10, bank(basis(0)), [1, 0, 0, 0], aspect=0.9)
    right = make("cam_224", "b", 0, 10, bank(basis(1)), [0, 1, 0, 0], aspect=0.9)
    pair = resolver.pair(left, right, [left], [right])
    assert pair.model_support == 0


def test_same_camera_overlapping_local_tracks_are_split_into_separate_groups():
    tracks = {
        "cam_224:1:1": make("cam_224", "a", 0, 10, bank(basis(0)), [1, 0, 0, 0]),
        "cam_224:2:1": make("cam_224", "b", 5, 15, bank(basis(0)), [1, 0, 0, 0]),
    }
    groups = StateInvariantFinalResolver.build_groups(
        {"cam_224:1:1": "G1", "cam_224:2:1": "G1"}, tracks,
    )
    assert len(groups) == 2
    assert all(len(group.members) == 1 for group in groups)


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
    a = make("cam_224", "a", 0, 10, bank(basis(0)), [1, 0, 0, 0])
    b = make("cam_224", "b", 11, 20, bank(basis(0)), [1, 0, 0, 0])
    groups = [
        type("Group", (), {"key": "a", "camera": "cam_224", "members": ["a"], "start": a.start})(),
        type("Group", (), {"key": "b", "camera": "cam_224", "members": ["b"], "start": b.start})(),
    ]
    components = StateInvariantFinalResolver._components(groups, [{"left": "a", "right": "b", "fused": 0.99}])
    assert len(components) == 2
