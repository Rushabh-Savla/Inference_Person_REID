from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def config():
    return yaml.safe_load(text("rebuild/config_state_invariant.yaml"))


def test_nvdcf_is_the_only_tracker_backend():
    value = config()["detector"]
    assert value["backend"] == "nvdcf"
    assert "config" in value
    assert "nvdcf" in str(value["config"]).lower()
    whole = text("rebuild/config_state_invariant.yaml").lower()
    assert "botsort" not in whole
    assert "bytetrack" not in whole


def test_nvdcf_uses_real_deepstream_tracker():
    value = text("rebuild/nvdcf_tracker.py")
    assert "nvtracker" in value
    assert "libnvds_nvmultiobjecttracker.so" in value
    assert "pyds" in value
    assert 'tracker.set_property("ll-lib-file"' in value
    assert 'tracker.set_property("ll-config-file"' in value
    assert "object_id" in value


def test_global_assignment_is_feature_first_and_one_to_one():
    value = text("rebuild/multimodal_identity.py")
    batch = text("rebuild/batch_nvdcf.py")
    assert "linear_sum_assignment" in value
    assert "self.identity.observe" in batch
    assert "recovery=True" in batch
    assert "feature-only" in batch.lower()
    assert "tracker_id_global_fallback" not in batch
    assert 'feature_map.get(int(item["track_id"]), "PENDING")' in batch
    assert 'if active_overlap:' in batch
    assert 'gids = {' in batch


def test_new_gid_admission_requires_multiframe_evidence():
    value = text("rebuild/multimodal_identity.py")
    cfg = config()["identity"]
    assert "_stage_new" in value
    assert "self.pending" in value
    assert "_new(seed)" in value
    assert int(cfg["new_confirm_frames"]) >= 3
    assert float(cfg["new_pending_match_min"]) >= 0.80
    assert float(cfg["new_pending_model_min"]) >= 0.50
    assert float(cfg["new_pending_clothing_min"]) >= 0.55
    assert int(cfg["new_pending_required_models"]) == 3


def test_required_feature_stack_is_present():
    value = text("rebuild/multimodal_identity.py") + "\n" + text("rebuild/multimodal_identity_strict.py")
    for name in (
        "NVIDIAReIDExtractor",
        "NVIDIASwinReIDExtractor",
        "SOLIDERReIDExtractor",
        "FaceExtractorV4",
        "pack",
        "QdrantGallery",
        "YOLO",
        "top",
        "bottom",
        "pose",
        "face",
    ):
        assert name in value

    # Face is the highest-weight single identity cue when reliable.
    assert "0.62 * face" in value
    # Top and bottom clothing are independent mandatory score terms.
    assert "0.09 * top" in value
    assert "0.09 * bottom" in value
    assert "0.22 * top" not in value or "0.22 * bottom" not in value


def test_active_resolver_is_strict_wrapper():
    value = text("rebuild/multimodal_identity.py")
    assert "MultiModalStrict" in value
    assert "feature-only" in value.lower()
    assert "track_id" in value


def test_face_and_pose_are_mandatory_runtime_components():
    value = config()
    assert value["face"]["enabled"] is True
    assert value["face"]["required"] is True
    assert float(value["face"]["min_visibility"]) >= 0.65
    assert value["pose"]["enabled"] is True


def test_qdrant_has_every_identity_space():
    value = text("src/live/qdrant_gallery.py")
    for name in (
        '"resnet": 256',
        '"swin": 1024',
        '"solider": 1024',
        '"attributes": 112',
        '"face": 512',
        '"pose": 51',
    ):
        assert name in value


def test_batch_has_hard_same_frame_collision_gate():
    value = text("rebuild/batch_nvdcf.py")
    assert "same-frame duplicate GID" in value
    assert "same-frame duplicate GID survived validation" in value
    assert "PENDING" in value


def test_no_tracker_id_to_gid_fallback_exists():
    value = text("rebuild/batch_nvdcf.py")
    assert "self.identity.trackmap.get" not in value
    assert "trackmap.get" not in value
    assert "f\"G{int(prior)" not in value