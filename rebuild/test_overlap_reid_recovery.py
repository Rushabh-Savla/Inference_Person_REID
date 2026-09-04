from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rebuild.face_v4 import FaceExtractorV4
from rebuild.multimodel_state_invariant_attributes import AttributeAwareResolver
from rebuild.multimodel_state_invariant_final import LocalGroup
from rebuild.overlap_recovery import OverlapEpisode, bbox_overlap_metrics, is_severe_overlap, recovery_sources


def unit(index: int, dim: int = 8) -> np.ndarray:
    value = np.zeros(dim, np.float32)
    value[index] = 1.0
    return value


def attrs(upper: int, lower: int, up_pattern: int | None = None, low_pattern: int | None = None):
    value = np.zeros(112, np.float32)
    value[upper % 20] = 1.0
    value[20 + (lower % 20)] = 1.0
    value[40 + ((up_pattern if up_pattern is not None else upper) % 14)] = 1.0
    value[54 + ((low_pattern if low_pattern is not None else lower) % 14)] = 1.0
    value[68] = 1.0
    value[102] = 1.0
    value[108:112] = 1.0
    return value / np.linalg.norm(value)


def bank(value: np.ndarray):
    return {model: {view: [value] for view in ("full", "upper", "torso", "lower")} for model in ("resnet", "swin", "solider")}


def group(key, camera, start, end, value, colour=None, aspect=1.6, attr=None, face=None, recovery=False, sources=None):
    result = LocalGroup(
        key=key, camera=camera, local_gid="GTEST", members=[key], start=float(start), end=float(end),
        aspect=float(aspect), state_type="upright" if aspect >= 1.35 else "compact", state_bank=bank(value),
        colour_signature=np.asarray(colour if colour is not None else [1, 0, 0, 0], np.float32),
        start_center=(100.0, 100.0), end_center=(100.0, 100.0), end_height=180.0,
    )
    setattr(result, "attribute_bank", [attr if attr is not None else attrs(0, 0)])
    setattr(result, "face_bank", list(face or []))
    setattr(result, "overlap_recovery", bool(recovery))
    setattr(result, "recovery_sources", list(sources or []))
    return result


def test_severe_overlap_uses_containment_not_ordinary_crossing():
    severe = bbox_overlap_metrics((0, 0, 100, 300), (10, 30, 90, 270))
    normal = bbox_overlap_metrics((0, 0, 100, 100), (70, 0, 170, 100))
    assert is_severe_overlap(severe, 0.80, 0.85)
    assert not is_severe_overlap(normal, 0.80, 0.85)


def test_tracker_id_change_is_recovered_from_pre_overlap_geometry():
    episode = OverlapEpisode("cam_213", 100, 110)
    episode.touch("cam_213:17:1", (100, 100, 170, 280), 110, anchor=True)
    episode.clear(112, 1)
    sources = recovery_sources((106, 102, 176, 282), 114, 20.0, [episode], 4.5, 1.75)
    assert sources == ["cam_213:17:1"]


def test_tracker_id_itself_is_not_required_for_recovery():
    episode = OverlapEpisode("cam_213", 100, 110)
    episode.touch("cam_213:17:1", (100, 100, 170, 280), 110, anchor=True)
    episode.clear(112, 1)
    result = recovery_sources((106, 102, 176, 282), 114, 20.0, [episode], 4.5, 1.75)
    assert "cam_213:44:1" not in result
    assert result


def test_top_match_with_conflicting_lower_clothing_is_rejected():
    resolver = AttributeAwareResolver({})
    person = unit(0)
    left = group("a", "cam_213", 0, 10, person, attr=attrs(0, 0))
    right = group("b", "cam_213", 12, 22, person, attr=attrs(0, 7))
    evidence = resolver.pair(left, right, [left], [right])
    assert resolver._accept(evidence, left, right, True) is False


def test_matching_lower_clothing_supports_same_identity():
    resolver = AttributeAwareResolver({})
    person = unit(0)
    left = group("a", "cam_213", 0, 10, person, attr=attrs(0, 2, 1, 3))
    right = group("b", "cam_213", 12, 22, person, attr=attrs(0, 2, 1, 3))
    evidence = resolver.pair(left, right, [left], [right])
    assert resolver._accept(evidence, left, right, True) is True


def test_post_overlap_source_can_stitch_new_tracker_key():
    resolver = AttributeAwareResolver({})
    person = unit(0)
    pre = group("cam_213:17:1", "cam_213", 0, 10, person, attr=attrs(0, 2))
    post = group("cam_213:44:1", "cam_213", 11, 20, person, attr=attrs(0, 2), recovery=True, sources=[pre.key])
    edges = resolver.same_camera_edges([pre, post])
    assert len(edges) == 1
    assert edges[0]["right"] == post.key
    assert edges[0]["recovery_source_match"]


def test_ambiguous_post_overlap_group_stays_pending():
    resolver = AttributeAwareResolver({})
    item = group("cam_213:44:1", "cam_213", 11, 20, unit(0), attr=attrs(0, 2), recovery=True, sources=["cam_213:17:1"])
    mapping = resolver._assign([[item]])
    assert mapping[item.key] == "PENDING"


def test_face_vector_is_high_value_when_valid():
    resolver = AttributeAwareResolver({"face_match_weight": 0.40})
    person = unit(0)
    face = {"vector": unit(0, 16), "quality": 0.90, "visibility": 0.80, "valid": True}
    left = group("a", "cam_213", 0, 10, person, attr=attrs(0, 2), face=[face])
    right = group("b", "cam_224", 12, 22, person, attr=attrs(0, 2), face=[face])
    evidence = resolver.pair(left, right, [left], [right])
    meta = resolver._meta[tuple(sorted((left.key, right.key)))]
    assert meta["face"]["valid"]
    assert evidence.fused > 0.60


def test_face_is_ignored_when_visibility_is_below_gate():
    resolver = AttributeAwareResolver({})
    person = unit(0)
    face = {"vector": unit(0, 16), "quality": 0.90, "visibility": 0.40, "valid": False}
    left = group("a", "cam_213", 0, 10, person, attr=attrs(0, 2), face=[face])
    right = group("b", "cam_224", 12, 22, person, attr=attrs(0, 2), face=[face])
    evidence = resolver.pair(left, right, [left], [right])
    meta = resolver._meta[tuple(sorted((left.key, right.key)))]
    assert not meta["face"]["valid"]
    assert evidence.fused < 0.99


def test_face_visibility_proxy_distinguishes_full_and_partial_landmarks():
    full = SimpleNamespace(landmark_2d_106=np.asarray([
        [40 + 38 * np.cos(theta), 40 + 38 * np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ], np.float32))
    partial = SimpleNamespace(landmark_2d_106=np.asarray([
        [40 + 8 * np.cos(theta), 40 + 8 * np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 20, endpoint=False)
    ], np.float32))
    full_score = FaceExtractorV4._visibility_fraction(full, (0, 0, 80, 80))
    partial_score = FaceExtractorV4._visibility_fraction(partial, (0, 0, 80, 80))
    assert full_score > partial_score
    assert full_score >= 0.60
    assert partial_score < 0.60


def test_simultaneous_same_camera_overlap_is_not_created_as_a_stitch():
    resolver = AttributeAwareResolver({})
    person = unit(0)
    first = group("a", "cam_213", 0, 10, person, attr=attrs(0, 2))
    second = group("b", "cam_213", 5, 15, person, attr=attrs(0, 2))
    assert resolver.same_camera_edges([first, second]) == []
