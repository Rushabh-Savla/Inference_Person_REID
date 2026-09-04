from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rebuild.identity_v3 import Tracklet
from rebuild.multimodel_reid import MultiModelLocalGlobalResolver
from live.persistent_multimodel import PersistentMultimodelRegistry


def vec(index: int, dim: int = 16, value: float = 1.0) -> np.ndarray:
    out = np.zeros(dim, np.float32)
    out[index] = value
    return out


def make_track(camera: str, tid: int, start: float, end: float, index: int) -> Tracklet:
    track = Tracklet(camera, tid, 1, 20.0)
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
        track.add(vec(index), "full", 0.95, meta, 0.985, 8)
        track.add(vec(index), "light", 0.94, {**meta, "kind": "light"}, 0.985, 8)
    track.start = start
    track.end = end
    track.shape = 2.2
    track.model_bank = {
        "swin": [vec(index), vec(index, value=0.98)],
        "solider": [vec(index), vec(index, value=0.97)],
    }
    track.colour_signature = np.eye(4, dtype=np.float32)[index % 4]
    return track


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
        "final_cross_fused_min": 0.70,
        "final_cross_strong": 0.72,
        "final_cross_margin": 0.050,
        "final_cross_time_tolerance_sec": 10.0,
        "final_cross_max_gap_without_overlap_sec": 3.0,
        "final_gallery_match_min": 0.73,
        "final_gallery_margin": 0.050,
        "camera_time_offsets_sec": {},
        "final_w_resnet": 0.27,
        "final_w_swin": 0.34,
        "final_w_solider": 0.30,
        "final_w_colour": 0.04,
        "final_w_shape": 0.02,
        "final_w_temporal": 0.03,
        "final_pair_weights": {
            "cam_222-cam_224": {
                "resnet": 0.24,
                "swin": 0.32,
                "solider": 0.38,
                "colour": 0.03,
                "shape": 0.01,
                "temporal": 0.02,
            }
        },
    }


def test_registry_survives_engine_recreation(tmp_path: Path) -> None:
    path = tmp_path / "identity.sqlite3"
    first = PersistentMultimodelRegistry(path, model_id="test-v1")
    gid = first.allocate_gid()
    assert gid == 1
    track = make_track("cam_222", 1, 1.0, 3.0, 0)
    first.save_component(
        gid,
        model_banks={"resnet": [x.vector for x in track.features], **track.model_bank},
        cameras=["cam_222"],
        last_ts=3.0,
        obs=len(track.features),
    )
    first.close()

    second = PersistentMultimodelRegistry(path, model_id="test-v1")
    assert second.gids() == [1]
    assert second.next_gid == 2
    gallery = second.load_gallery()
    assert 1 in gallery
    assert set(gallery[1]) == {"resnet", "swin", "solider"}
    second.close()


def test_new_component_continues_monotonic_namespace(tmp_path: Path) -> None:
    path = tmp_path / "identity.sqlite3"
    registry = PersistentMultimodelRegistry(path, model_id="test-v1")
    assert registry.allocate_gid() == 1
    assert registry.allocate_gid() == 2
    registry.close()
    reopened = PersistentMultimodelRegistry(path, model_id="test-v1")
    assert reopened.allocate_gid() == 3
    reopened.close()


def test_multimodel_resolver_preserves_persistent_gid(tmp_path: Path) -> None:
    path = tmp_path / "identity.sqlite3"
    registry = PersistentMultimodelRegistry(path, model_id="test-v1")
    seed = make_track("cam_222", 1, 5.0, 15.0, 0)
    gid = registry.allocate_gid()
    registry.save_component(
        gid,
        model_banks={"resnet": [x.vector for x in seed.features], **seed.model_bank},
        cameras=["cam_222"],
        last_ts=15.0,
        obs=len(seed.features),
    )

    resolver = MultiModelLocalGlobalResolver(cfg(), registry=registry)
    same = make_track("cam_224", 1, 5.1, 15.1, 0)
    result, _, edges = resolver.resolve(
        {seed.key: "G000001", same.key: "G000009"},
        {seed.key: seed, same.key: same},
        ["cam_222", "cam_224"],
    )
    assert result[seed.key] == "G000001"
    assert result[same.key] == "G000001"
    assert edges
    registry.close()
