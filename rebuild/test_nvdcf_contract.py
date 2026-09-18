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
    value = text("rebuild/multimodal_identity.py") + "\n" + text("rebuild/multimodal_identity_strict.py")
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
    value = text("rebuild/multimodal_identity.py") + "\n" + text("rebuild/multimodal_identity_strict.py")
    cfg = config()["identity"]
    assert "stage_new" in value
    assert "self.pending" in value
    assert "self.new(seed)" in value
    assert int(cfg["new_confirm_frames"]) >= 3
    assert float(cfg["new_pending_match_min"]) >= 0.80
    assert float(cfg["new_pending_model_min"]) >= 0.50
    assert float(cfg["new_pending_clothing_min"]) >= 0.55
    assert int(cfg["new_pending_required_models"]) == 3
    assert float(cfg["new_pending_margin"]) >= 0.08


def test_required_feature_stack_is_present():
    value = text("rebuild/multimodal_identity.py") + "\n" + text("rebuild/multimodal_identity_strict.py")
    for name in (
        "NVIDIAReIDExtractor",
        "NVIDIASwinReIDExtractor",
        "SOLIDERReIDExtractor",
        "pack",
        "QdrantGallery",
        "YOLO",
        "top",
        "bottom",
        "pose",
    ):
        assert name in value

    # Top and bottom clothing are independent mandatory score terms.
    assert '"top": float(top)' in value
    assert '"bottom": float(bot)' in value
    assert 'attrs["top"]' in value
    assert 'attrs["bottom"]' in value


def test_active_resolver_is_strict_wrapper():
    value = text("rebuild/multimodal_identity.py")
    assert "MultiModalStrict" in value
    assert "feature-only" in value.lower()
    assert "track_id" in value


def test_unknown_observation_alignment_is_preserved():
    value = text("rebuild/multimodal_identity_strict.py")
    assert "for item, value in zip(obs, sets):" in value
    assert "scores = [x for x in sets if x]" not in value


def test_pending_promotion_requires_all_three_models():
    value = text("rebuild/multimodal_identity_strict.py")
    assert "self.pending_required_models = 3" in value
    assert 'score["support"] < self.pending_required_models' in value
    assert "self.pending_margin" in value


def test_nvidia_reid_and_pose_are_mandatory_runtime_components():
    value = config()
    assert value["reid"]["model"] == "nvidia_reidentificationnet"
    assert value["cross_camera_models"]["swin_weights"]
    assert value["cross_camera_models"]["solider_weights"]
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


def test_overlap_uses_clean_anchors_only_inside_overlap():
    value = text("rebuild/batch_nvdcf.py")
    assert "overlap_anchors" in value
    assert "last_gids" in value
    assert "never used after the overlap ends" in value
    assert "overlap_anchors = {}" in value
    assert "candidate = str(" in value
    assert "candidate not in anchor_used" in value


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


def test_face_runtime_is_mandatory():
    cfg = config()
    assert cfg["face"]["enabled"] is True
    assert cfg["face"]["model"] == "buffalo_l"
    assert float(cfg["face"]["min_visibility"]) >= 0.65
    assert float(cfg["face"]["min_quality"]) >= 0.50
    value = text("rebuild/multimodal_identity_strict.py")
    assert "FaceExtractorV4" in value
    assert "self.face.extract" in value
    assert "faceval" in value

def test_deepstream_runtime_is_discovered():
    value = text("rebuild/deepstream_runtime.py") + "\n" + text("rebuild/nvdcf_tracker.py")
    assert "DeepStreamRuntime.find" in value
    assert "NVDCF_DEEPSTREAM_ROOT" in value
    assert "libnvds_nvmultiobjecttracker.so" in value
