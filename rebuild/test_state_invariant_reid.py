from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rebuild.multimodel_state_invariant import StateInvariantResolver


def basis(index: int, dim: int = 8) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    value[index] = 1.0
    return value


def feature_bank(person: np.ndarray, other: np.ndarray | None = None):
    other = person if other is None else other
    return {
        "resnet": {
            "full": [person],
            "upper": [person],
            "torso": [person],
            "lower": [person],
        },
        "swin": {
            "full": [person],
            "upper": [person],
            "torso": [person],
            "lower": [person],
        },
        "solider": {
            "full": [other],
            "upper": [other],
            "torso": [other],
            "lower": [other],
        },
    }


def track(camera, key, start, end, bank, colour):
    return SimpleNamespace(
        camera=camera,
        start=float(start),
        end=float(end),
        state_bank=bank,
        colour_signature=np.asarray(colour, np.float32),
    )


def test_standing_to_sitting_matches_through_upper_and_torso_views():
    cfg = {}
    resolver = StateInvariantResolver(cfg)
    person = basis(0)
    left = track(
        "cam_222", "cam_222:G1:lane0", 10, 20,
        feature_bank(person), [1, 0, 0, 0],
    )
    right = track(
        "cam_224", "cam_224:G2:lane0", 10, 20,
        {
            "resnet": {"upper": [person], "torso": [person]},
            "swin": {"upper": [person], "torso": [person]},
            "solider": {"upper": [person], "torso": [person]},
        },
        [1, 0, 0, 0],
    )
    pair = resolver._pair(left, right)
    assert pair.state_transition is True
    assert pair.model_support == 3
    assert pair.fused >= cfg.get("state_cross_partial_fused_min", 0.62)


def test_different_people_do_not_match_just_because_both_are_seated():
    resolver = StateInvariantResolver({})
    a = basis(0)
    b = basis(1)
    left = track("cam_222", "a", 0, 10, feature_bank(a), [1, 0, 0, 0])
    right = track("cam_224", "b", 0, 10, feature_bank(b), [0, 1, 0, 0])
    pair = resolver._pair(left, right)
    assert pair.model_support < 2


def test_same_camera_component_cannot_contain_two_overlapping_tracks():
    a = basis(0)
    tracks = {
        "cam_224:1:1": track("cam_224", "a", 0, 10, feature_bank(a), [1, 0, 0, 0]),
        "cam_224:2:1": track("cam_224", "b", 5, 15, feature_bank(a), [1, 0, 0, 0]),
    }
    groups = StateInvariantResolver.build_groups(
        {"cam_224:1:1": "G1", "cam_224:2:1": "G1"}, tracks,
    )
    assert len(groups) == 2


def test_ambiguous_one_model_evidence_does_not_force_cross_camera_merge():
    resolver = StateInvariantResolver({})
    a = basis(0)
    b = basis(1)
    left = track(
        "cam_222", "a", 0, 10,
        {"resnet": {"full": [a]}, "swin": {"full": [a]}, "solider": {"full": [a]},},
        [1, 0, 0, 0],
    )
    right = track(
        "cam_224", "b", 0, 10,
        {"resnet": {"full": [b]}, "swin": {"full": [a]}, "solider": {"full": [b]},},
        [1, 0, 0, 0],
    )
    pair = resolver._pair(left, right)
    assert pair.model_support == 1
