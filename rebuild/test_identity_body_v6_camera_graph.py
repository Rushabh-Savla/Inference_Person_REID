from __future__ import annotations

import numpy as np

from rebuild.identity_body_v6_camera_graph import GlobalIdentityBodyV6CameraGraph
from rebuild.identity_v3 import Feature, Tracklet


def basis(index: int, dim: int = 8) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    value[index] = 1.0
    return value


def make_track(camera: str, track_id: int, value: np.ndarray, colour: np.ndarray, start: float) -> Tracklet:
    track = Tracklet(camera, track_id, 1, 20.0, start=start, end=start + 5.0)
    track.features = [Feature(value.copy(), "full", 0.95, camera, start)]
    track.observations = [
        {"timestamp": start, "bbox": [100, 100, 180, 300]},
        {"timestamp": start + 5.0, "bbox": [105, 102, 185, 302]},
    ]
    track.shape = 2.5
    track.colour_signature = colour.astype(np.float32)
    return track


def engine() -> GlobalIdentityBodyV6CameraGraph:
    return GlobalIdentityBodyV6CameraGraph(
        {
            "match_threshold": 0.60,
            "match_margin": 0.035,
            "strong_threshold": 0.72,
            "support_required": 2,
            "accumulated_body": 0.56,
            "accumulated_support": 3,
            "partial_threshold": 0.58,
            "partial_support": 2,
            "gallery": 32,
            "candidate_gallery": 12,
            "promote_quality": 0.68,
            "merge_body": 0.82,
            "merge_support": 3,
            "cross_graph_reid_weight": 0.78,
            "cross_graph_color_weight": 0.12,
            "cross_graph_time_weight": 0.06,
            "cross_graph_shape_weight": 0.04,
            "cross_graph_min_reid": 0.50,
            "cross_graph_min_score": 0.58,
            "cross_graph_margin": 0.035,
            "cross_graph_color_gate": 0.62,
            "cross_graph_strong_reid": 0.68,
            "cross_graph_time_tolerance_sec": 8.0,
            "camera_time_offsets_sec": {"cam_213": 8.0, "cam_222": 0.0, "cam_224": 3.0},
        }
    )


def test_camera_graph_fixes_cross_camera_label_swap() -> None:
    # Base mapping represents the real failure pattern observed in the videos:
    # brown is G1 in 222 but G2 in 224; white is G2 in 222 but G1 in 224.
    brown = basis(0)
    white = basis(1)
    brown_colour = np.asarray([1.0, 0.05, 0.05, 0.05], dtype=np.float32)
    white_colour = np.asarray([0.05, 0.05, 0.05, 1.0], dtype=np.float32)

    tracks = {
        "cam_222:1:1": make_track("cam_222", 1, brown, brown_colour, 20.0),
        "cam_222:2:1": make_track("cam_222", 2, white, white_colour, 20.0),
        "cam_224:1:1": make_track("cam_224", 1, white, white_colour, 20.0),
        "cam_224:2:1": make_track("cam_224", 2, brown, brown_colour, 20.0),
    }

    engine = engine()
    mapping = {
        "cam_222:1:1": "G000001",
        "cam_222:2:1": "G000002",
        "cam_224:1:1": "G000001",  # WRONG: should become G000002.
        "cam_224:2:1": "G000002",  # WRONG: should become G000001.
    }

    corrected = engine._reconcile(mapping, tracks)

    assert corrected["cam_222:1:1"] == "G000001"
    assert corrected["cam_222:2:1"] == "G000002"
    assert corrected["cam_224:1:1"] == "G000002"
    assert corrected["cam_224:2:1"] == "G000001"


def test_components_never_merge_two_people_from_one_camera() -> None:
    engine = engine()
    edges = [
        {"camera_a": "cam_222", "camera_b": "cam_224", "gid_a": "G000001", "gid_b": "G000001", "score": 0.90},
        {"camera_a": "cam_224", "camera_b": "cam_213", "gid_a": "G000001", "gid_b": "G000001", "score": 0.89},
        {"camera_a": "cam_222", "camera_b": "cam_213", "gid_a": "G000002", "gid_b": "G000001", "score": 0.88},
    ]
    nodes = [
        ("cam_222", "G000001"),
        ("cam_222", "G000002"),
        ("cam_224", "G000001"),
        ("cam_213", "G000001"),
    ]
    components = engine._components(edges, nodes)
    for component in components:
        cameras = [camera for camera, _ in component]
        assert len(cameras) == len(set(cameras))


if __name__ == "__main__":
    test_camera_graph_fixes_cross_camera_label_swap()
    test_components_never_merge_two_people_from_one_camera()
    print("V6 CAMERA GRAPH TEST: OK")
