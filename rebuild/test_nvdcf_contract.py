from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_nvdcf_is_the_only_tracker_backend():
    value = text("rebuild/config_state_invariant.yaml")
    assert "backend: nvdcf" in value
    assert "tracker:" not in value
    assert "botsort" not in value.lower()
    assert "bytetrack" not in value.lower()


def test_nvdcf_uses_deepstream_tracker():
    value = text("rebuild/nvdcf_tracker.py")
    assert "nvtracker" in value
    assert "libnvds_nvmultiobjecttracker.so" in value
    assert "config_tracker_NvDCF_accuracy.yml" in value
    assert "pyds" in value
    assert "object_id" in value


def test_global_assignment_is_feature_first_and_one_to_one():
    value = text("rebuild/multimodal_identity.py")
    assert "linear_sum_assignment" in value
    assert "top_clothing_color" in text("rebuild/batch_nvdcf.py")
    assert "bottom_clothing_color" in text("rebuild/batch_nvdcf.py")
    assert "commit=not bool(active_overlap)" in text(
        "rebuild/batch_nvdcf.py"
    )
    assert "Post-overlap identity MUST come from features" in text(
        "rebuild/batch_nvdcf.py"
    )


def test_required_feature_stack_is_present():
    value = text("rebuild/multimodal_identity.py")
    for name in (
        "NVIDIAReIDExtractor",
        "NVIDIASwinReIDExtractor",
        "SOLIDERReIDExtractor",
        "FaceExtractorV4",
        "pack",
        "QdrantGallery",
        "YOLO",
    ):
        assert name in value

    assert "0.55 * face" in value
    assert "0.10 * top" in value
    assert "0.10 * bottom" in value
    assert "0.05 * pose" in value


def test_face_and_pose_are_mandatory_runtime_components():
    config = text("rebuild/config_state_invariant.yaml")
    assert "face:" in config and "enabled: true" in config
    assert "required: true" in config
    assert "min_visibility: 0.68" in config
    assert "pose:" in config and "enabled: true" in config


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
    assert "same-frame duplicate global ID" in value
    assert "same-frame duplicate GID survived validation" in value
