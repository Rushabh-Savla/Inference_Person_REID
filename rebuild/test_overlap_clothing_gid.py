from __future__ import annotations

import inspect

import numpy as np

from rebuild.multimodal_identity import MultiModal
from rebuild.multimodal_identity_strict import MultiModalStrict
from src.live.persistent_multimodel import PersistentMultimodelRegistry


def vector(size: int, index: int) -> np.ndarray:
    value = np.zeros(size, np.float32)
    value[index] = 1.0
    return value


def attributes(top: int, bottom: int) -> np.ndarray:
    value = np.zeros(112, np.float32)
    value[0:20] = vector(20, top)
    value[20:40] = vector(20, bottom)
    value[40:54] = vector(14, top % 14)
    value[54:68] = vector(14, bottom % 14)
    value[68:102] = 1.0
    value[102:108] = 1.0
    value[108:112] = 1.0
    return value / np.linalg.norm(value)


def test_clothing_cannot_mix_top_from_one_gallery_view_and_bottom_from_another():
    query = attributes(0, 0)
    split_a = attributes(0, 1)
    split_b = attributes(1, 0)
    result = MultiModalStrict.attr(query, [split_a, split_b])

    assert result is not None
    assert result["joint"] < 0.52
    assert not (
        result["top"] >= 0.52
        and result["bottom"] >= 0.52
        and result["joint"] >= 0.52
    )


def test_matching_top_and_bottom_from_same_gallery_view_passes():
    query = attributes(2, 3)
    result = MultiModalStrict.attr(query, [attributes(2, 3)])

    assert result is not None
    assert result["top"] > 0.90
    assert result["bottom"] > 0.90
    assert result["joint"] > 0.90


def test_gid_allocation_starts_at_one_and_advances_from_persisted_max(tmp_path):
    path = tmp_path / "identity.sqlite3"
    reg = PersistentMultimodelRegistry(path, model_id="test", bank_size=4)

    assert reg.allocate_gid() == 1
    reg.save_component(
        1,
        model_banks={"resnet": [vector(4, 0)]},
        cameras={"cam_a"},
        last_ts=1.0,
        obs=1,
    )
    assert reg.allocate_gid() == 2

    reg._db.execute("UPDATE meta SET value='1' WHERE key='next_gid'")
    reg._db.commit()
    assert reg.allocate_gid() == 2

    reg.save_component(
        9,
        model_banks={"resnet": [vector(4, 1)]},
        cameras={"cam_a"},
        last_ts=2.0,
        obs=1,
    )
    assert reg.allocate_gid() == 10
    reg.close()


def test_active_multimodal_forwards_recovery_hints():
    parameter = inspect.signature(MultiModal.observe).parameters
    assert "recovery_hints" in parameter
