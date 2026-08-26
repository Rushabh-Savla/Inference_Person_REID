from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rebuild.multimodel_state_invariant_final import LocalGroup, StateInvariantFinalResolver


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
    observations = observations or []
    first = last = None
    height = 1.0
    if observations:
        ordered = sorted(observations, key=lambda row: float(row["timestamp"]))
        def centre(row):
            x1, y1, x2, y2 = row["bbox"]
            return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        first = centre(ordered[0])
        last = centre(ordered[-1])
        height = max(1.0, float(ordered[-1]["bbox"][3]) - float(ordered[-1]["bbox"][1]))
    return LocalGroup(
        key=key, camera=camera, local_gid="GTEST", members=[key], start=float(start), end=float(end),
        aspect=float(aspect), state_type="upright" if aspect >= 1.35 else "compact", state_bank=state_bank,
        colour_signature=np.asarray(colour, np.float32), start_center=first, end_center=last, end_height=height,
    )


def obs(t, x, y=100.0, h=180.0, w=70.0):
    return {"timestamp": float(t), "bbox": [float(x), float(y), float(x + w), float(y + h)]}


def test_standing_to_sitting_uses_upper_torso_consensus():
    resolver = StateInvariantFinalResolver({})
    person = basis(0)
    left = make("cam_222", "a", 10, 20, bank(person), [1, 0, 0, 0], aspect=1.8)
    right_bank = {"resnet": {"upper": [person], "torso": [person]}, "swin": {"upper": [person], "torso": [person]}, "solider": {"upper": [person], "torso": [person]}}
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
    assert pair.fused < 0.54


def test_same_camera_overlapping_tracks_never_stitch():
    resolver = StateInvariantFinalResolver({})
    a = make("cam_213", "a", 0, 10, bank(basis(0)), [1, 0, 0, 0], observations=[obs(1, 100)])
    b = make("cam_213", "b", 5, 15, bank(basis(0)), [1, 0, 0, 0], observations=[obs(6, 120)])
    assert resolver.same_camera_edges([a, b]) == []


def test_same_camera_reset_fragments_stitch():
    resolver = StateInvariantFinalResolver({})
    a = make("cam_213", "a", 0, 4, bank(basis(0)), [1, 0, 0, 0], observations=[obs(3.5, 100), obs(4.0, 105)])
    b = make("cam_213", "b", 5, 9, bank(basis(0)), [1, 0, 0, 0], observations=[obs(5.0, 108), obs(8.5, 112)])
    edges = resolver.same_camera_edges([a, b])
    assert len(edges) == 1
    stitched = resolver._stitch([a, b], edges)
    assert len(stitched) == 1
    assert sorted(stitched[0].members) == ["a", "b"]


def test_same_camera_reset_is_repaired_before_cross_camera():
    person = basis(0)
    tracks = {
        "cam_213:1:1": make("cam_213", "a", 0, 5, bank(person), [1, 0, 0, 0], observations=[obs(0, 100, h=190)]),
        "cam_213:2:1": make("cam_213", "b", 6, 10, bank(person), [1, 0, 0, 0], observations=[obs(6, 108, h=188)]),
        "cam_224:9:1": make("cam_224", "c", 12, 20, bank(person), [1, 0, 0, 0], observations=[obs(12, 100, h=188)]),
    }
    mapping, _components, _edges = StateInvariantFinalResolver({}).resolve({"cam_213:1:1": "G1", "cam_213:2:1": "G2", "cam_224:9:1": "G1"}, tracks, ["cam_213", "cam_224"])
    assert mapping["cam_213:1:1"] == mapping["cam_213:2:1"]
    assert mapping["cam_213:1:1"] == mapping["cam_224:9:1"]


def test_cross_camera_timestamp_offset_is_not_required_for_strong_match():
    resolver = StateInvariantFinalResolver({})
    person = bank(basis(0))
    left = make("cam_213", "a", 0, 5, person, [1, 0, 0, 0])
    right = make("cam_224", "b", 40, 45, person, [1, 0, 0, 0])
    pair = resolver.pair(left, right, [left], [right])
    edges = resolver.cross_edges([left], [right])
    assert pair.model_support == 3
    assert edges


def test_cross_camera_reciprocal_consensus():
    resolver = StateInvariantFinalResolver({})
    person = bank(basis(0))
    left = [make("cam_222", "a", 0, 5, person, [1, 0, 0, 0])]
    right = [make("cam_224", "b", 0, 5, person, [1, 0, 0, 0]), make("cam_224", "c", 0, 5, bank(basis(1)), [0, 1, 0, 0])]
    edges = resolver.cross_edges(left, right)
    assert len(edges) == 1
    assert edges[0]["left"] == "a"
    assert edges[0]["right"] == "b"


def test_one_model_agreement_is_never_enough():
    resolver = StateInvariantFinalResolver({})
    a, b = basis(0), basis(1)
    left = make("cam_222", "a", 0, 10, {"resnet": {"full": [a]}, "swin": {"full": [a]}, "solider": {"full": [a]}}, [1, 0, 0, 0])
    right = make("cam_224", "b", 0, 10, {"resnet": {"full": [b]}, "swin": {"full": [a]}, "solider": {"full": [b]}}, [1, 0, 0, 0])
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


def test_full_upper_torso_lower_are_retained():
    resolver = StateInvariantFinalResolver({})
    item = make("cam_224", "a", 0, 1, bank(basis(0)), [1, 0, 0, 0])
    groups = resolver.build_groups({"a": "G1"}, {"a": item})
    assert len(groups) == 1
    for model in resolver.MODELS:
        assert all(view in groups[0].state_bank[model] for view in resolver.VIEWS)
