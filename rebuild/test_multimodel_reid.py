from __future__ import annotations

import numpy as np

from rebuild.identity_v3 import Tracklet
from rebuild.multimodel_reid import MultiModelLocalGlobalResolver


def vec(index: int, dim: int = 16, value: float = 1.0) -> np.ndarray:
    out = np.zeros(dim, np.float32)
    out[index] = value
    return out


def mix(a: int, b: int, wa: float = 0.6, wb: float = 0.8, dim: int = 16) -> np.ndarray:
    out = np.zeros(dim, np.float32)
    out[a] = wa
    out[b] = wb
    return out


def make_track(camera: str, tid: int, start: float, end: float, idx: int, colour: np.ndarray) -> Tracklet:
    item = Tracklet(camera, tid, 1, 20.0)
    for stamp in (start, end):
        meta = {
            "camera": camera,
            "timestamp": stamp,
            "frame": int(stamp * 20),
            "track_id": tid,
            "bbox": [10, 10, 110, 230],
            "detection_score": 0.95,
            "quality": 0.95,
            "kind": "full",
        }
        item.add(vec(idx), "full", 0.95, meta, 0.985, 8)
        item.add(vec(idx), "light", 0.94, {**meta, "kind": "light"}, 0.985, 8)
    item.start = start
    item.end = end
    item.shape = 2.2
    item.model_bank = {
        "swin": [vec(idx), vec(idx, value=0.98)],
        "solider": [vec(idx), vec(idx, value=0.97)],
    }
    item.colour_signature = colour
    return item


def cfg() -> dict:
    return {
        "match_threshold": 0.60,
        "match_margin": 0.035,
        "strong_threshold": 0.72,
        "support_required": 2,
        "accumulated_body": 0.56,
        "accumulated_support": 3,
        "partial_threshold": 0.58,
        "partial_support": 2,
        "gallery": 16,
        "candidate_gallery": 8,
        "promote_quality": 0.68,
        "novelty": 0.985,
        "same_camera_gap_sec": 15.0,
        "same_camera_distance": 5.0,
        "same_camera_min_continuity": 0.35,
        "merge_body": 0.82,
        "merge_support": 3,
        "seed_count": 3,
        "final_cross_resnet_min": 0.55,
        "final_cross_swin_min": 0.55,
        "final_cross_solider_min": 0.50,
        "final_cross_fused_min": 0.69,
        "final_cross_strong": 0.72,
        "final_cross_margin": 0.045,
        "final_cross_conflict": 0.32,
        "final_cross_time_tolerance_sec": 12.0,
        "camera_time_offsets_sec": {},
        "final_w_resnet": 0.28,
        "final_w_swin": 0.34,
        "final_w_solider": 0.30,
        "final_w_colour": 0.05,
        "final_w_shape": 0.03,
        "final_pair_weights": {
            "cam_222-cam_224": {
                "resnet": 0.25,
                "swin": 0.32,
                "solider": 0.38,
                "colour": 0.03,
                "shape": 0.02,
            }
        },
    }


def test_three_model_permutation() -> None:
    red = np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)
    blue = np.asarray([0.0, 1.0, 0.0, 0.0], np.float32)
    tracks = {
        "cam_222:1:1": make_track("cam_222", 1, 10.0, 20.0, 0, red),
        "cam_222:2:1": make_track("cam_222", 2, 10.0, 20.0, 1, blue),
        "cam_224:1:1": make_track("cam_224", 1, 10.1, 20.1, 1, blue),
        "cam_224:2:1": make_track("cam_224", 2, 10.1, 20.1, 0, red),
    }
    local = {
        "cam_222:1:1": "G000001",
        "cam_222:2:1": "G000002",
        "cam_224:1:1": "G000001",
        "cam_224:2:1": "G000002",
    }
    result, components, edges = MultiModelLocalGlobalResolver(cfg()).resolve(local, tracks, ["cam_222", "cam_224"])
    assert result["cam_222:1:1"] == result["cam_224:2:1"]
    assert result["cam_222:2:1"] == result["cam_224:1:1"]
    assert len(components) == 2
    assert len(edges) == 2


def test_swin_and_solider_rescue_a_resnet_mistake() -> None:
    colour = np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)
    left = make_track("cam_222", 1, 5.0, 15.0, 0, colour)
    wrong = make_track("cam_224", 1, 5.1, 15.1, 0, colour)
    correct = make_track("cam_224", 2, 5.1, 15.1, 1, colour)
    # ResNet alone prefers wrong; Swin + SOLIDER agree on correct.
    wrong.features = [type(left.features[0])(mix(0, 1, 0.93, 0.37), "full", 0.95, "cam_224", 5.1, left.features[0].meta)]
    wrong.model_bank["swin"] = [mix(0, 1, 0.35, 0.94)]
    wrong.model_bank["solider"] = [mix(0, 1, 0.30, 0.95)]
    correct.features = [type(left.features[0])(mix(0, 1, 0.60, 0.80), "full", 0.95, "cam_224", 5.1, left.features[0].meta)]
    correct.model_bank["swin"] = [mix(0, 1, 0.92, 0.39)]
    correct.model_bank["solider"] = [mix(0, 1, 0.94, 0.34)]
    local = {left.key: "G000001", wrong.key: "G000001", correct.key: "G000002"}
    result, _, edges = MultiModelLocalGlobalResolver(cfg()).resolve(
        local, {left.key: left, wrong.key: wrong, correct.key: correct}, ["cam_222", "cam_224"]
    )
    assert result[left.key] == result[correct.key]
    assert result[left.key] != result[wrong.key]
    assert edges


def test_model_disagreement_is_not_forced() -> None:
    red = np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)
    blue = np.asarray([0.0, 1.0, 0.0, 0.0], np.float32)
    left = make_track("cam_222", 1, 1.0, 2.0, 0, red)
    right = make_track("cam_224", 1, 1.1, 2.1, 1, blue)
    right.model_bank["swin"] = [vec(0), vec(0, value=0.98)]
    local = {left.key: "G000001", right.key: "G000001"}
    result, _, edges = MultiModelLocalGlobalResolver(cfg()).resolve(local, {left.key: left, right.key: right}, ["cam_222", "cam_224"])
    assert result[left.key] != result[right.key]
    assert edges == []


def test_same_camera_overlap_is_split_before_global_matching() -> None:
    red = np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)
    a = make_track("cam_224", 1, 1.0, 8.0, 0, red)
    b = make_track("cam_224", 2, 2.0, 7.0, 0, red)
    c = make_track("cam_222", 1, 1.0, 8.0, 0, red)
    local = {a.key: "G000001", b.key: "G000001", c.key: "G000001"}
    result, components, _ = MultiModelLocalGlobalResolver(cfg()).resolve(local, {a.key: a, b.key: b, c.key: c}, ["cam_222", "cam_224"])
    assert result[a.key] != result[b.key]
    for members in components.values():
        cams = [x.split("::", 1)[0] for x in members]
        assert len(cams) == len(set(cams))
