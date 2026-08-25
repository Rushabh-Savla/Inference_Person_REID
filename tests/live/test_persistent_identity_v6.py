from __future__ import annotations

from pathlib import Path

import numpy as np

from live.persistent_identity import PersistentIdentityEngine


def vec(index: int, dim: int = 8) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    value[index] = 1.0
    return value


def assign_person(engine: PersistentIdentityEngine, cam: str, track_id: int, value: np.ndarray, start: float) -> int:
    rid = None
    for i in range(3):
        rid = engine.assign(cam, track_id, value, {"accepted": True}, start + i, True)
    assert rid is not None and rid > 0
    return int(rid)


def test_empty_scene_does_not_recycle_gid(tmp_path: Path):
    db = tmp_path / "identity.sqlite3"
    engine = PersistentIdentityEngine(
        min_evidence_obs=3,
        same_camera_threshold=0.90,
        cross_camera_threshold=0.63,
        accept_margin=0.03,
        bank_size=20,
        active_ttl_sec=1.0,
        max_active_identities=200,
        state_path=str(db),
    )

    first = assign_person(engine, "cam_213", 1, vec(0), 100.0)
    assert first == 1

    engine.sweep(1000.0)
    assert not engine.store.has(first)
    assert engine.registry.gids() == [1]

    second = assign_person(engine, "cam_213", 2, vec(1), 1100.0)
    assert second == 2
    engine.close()


def test_process_restart_recovers_existing_identity(tmp_path: Path):
    db = tmp_path / "identity.sqlite3"
    first_engine = PersistentIdentityEngine(state_path=str(db), min_evidence_obs=3)
    first = assign_person(first_engine, "cam_213", 1, vec(0), 100.0)
    first_engine.close()

    second_engine = PersistentIdentityEngine(state_path=str(db), min_evidence_obs=3)
    assert second_engine.registry.gids() == [first]
    recovered = assign_person(second_engine, "cam_213", 9, vec(0), 500.0)
    assert recovered == first
    second = assign_person(second_engine, "cam_224", 3, vec(1), 600.0)
    assert second != first
    assert second == 2
    second_engine.close()


def test_tracker_id_reset_reuses_same_gid(tmp_path: Path):
    db = tmp_path / "identity.sqlite3"
    engine = PersistentIdentityEngine(state_path=str(db), min_evidence_obs=3)
    first = assign_person(engine, "cam_224", 7, vec(2), 100.0)
    # New tracker id for the same person after a long enough gap.
    recovered = assign_person(engine, "cam_224", 2, vec(2), 1000.0)
    assert recovered == first
    engine.close()


def test_multiple_people_stay_distinct(tmp_path: Path):
    db = tmp_path / "identity.sqlite3"
    engine = PersistentIdentityEngine(state_path=str(db), min_evidence_obs=3)
    a = assign_person(engine, "cam_222", 1, vec(0), 100.0)
    b = assign_person(engine, "cam_222", 2, vec(1), 100.0)
    assert a != b
    engine.close()


def test_allocator_is_monotonic_after_restart(tmp_path: Path):
    db = tmp_path / "identity.sqlite3"
    engine = PersistentIdentityEngine(state_path=str(db), min_evidence_obs=3)
    assert assign_person(engine, "cam_213", 1, vec(0), 100.0) == 1
    assert assign_person(engine, "cam_213", 2, vec(1), 100.0) == 2
    engine.close()

    restarted = PersistentIdentityEngine(state_path=str(db), min_evidence_obs=3)
    assert assign_person(restarted, "cam_224", 8, vec(3), 300.0) == 3
    restarted.close()


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        test_empty_scene_does_not_recycle_gid(path)
        test_process_restart_recovers_existing_identity(path)
        test_tracker_id_reset_reuses_same_gid(path)
        test_multiple_people_stay_distinct(path)
        test_allocator_is_monotonic_after_restart(path)
        print("PERSISTENT V6 IDENTITY TEST: OK")
