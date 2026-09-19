from rebuild.overlap_guard import carry, merge


def test_two_detections_survive_one_nvdcf_box():
    tracked = [
        {"camera": "cam", "frame": 1, "timestamp": 0.1, "track_id": 7,
         "bbox": [0, 0, 100, 200], "detection_score": 0.9, "tracker_confidence": 0.9}
    ]
    detections = [
        {"camera": "cam", "frame": 2, "timestamp": 0.2,
         "bbox": [0, 0, 50, 200], "detection_score": 0.9},
        {"camera": "cam", "frame": 2, "timestamp": 0.2,
         "bbox": [50, 0, 100, 200], "detection_score": 0.9},
    ]
    out = merge(tracked, detections, 2)
    assert len(out) == 2
    assert sum(bool(x.get("shadow")) for x in out) == 1
    assert len({int(x["track_id"]) for x in out}) == 2


def test_spatial_overlap_carry_is_one_to_one_and_not_tracker_based():
    rows = [
        {"track_id": 91, "bbox": [0, 0, 50, 100]},
        {"track_id": 92, "bbox": [50, 0, 100, 100]},
    ]
    anchors = [
        {"gid": "G000001", "bbox": [50, 0, 100, 100]},
        {"gid": "G000002", "bbox": [0, 0, 50, 100]},
    ]
    out = carry(rows, anchors, set(), 0.15)
    assert out == {0: "G000002", 1: "G000001"}
    assert len(set(out.values())) == 2


def test_carry_never_reuses_gid():
    rows = [
        {"track_id": 1, "bbox": [0, 0, 60, 100]},
        {"track_id": 2, "bbox": [0, 0, 60, 100]},
    ]
    anchors = [
        {"gid": "G000001", "bbox": [0, 0, 60, 100]},
    ]
    out = carry(rows, anchors, set(), 0.15)
    assert len(set(out.values())) <= 1
