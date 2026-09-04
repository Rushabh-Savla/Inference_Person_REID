import numpy as np

from rebuild.identity_v2 import GlobalIdentityEngine, Tracklet, diverse_select, illumination_variant, geometry_similarity, unit


def feature(seed, dim=16):
    rng = np.random.default_rng(seed)
    return unit(rng.normal(size=dim).astype(np.float32))


def test_diverse_gallery():
    base = feature(1)
    near = [unit(base + 0.001 * feature(i)) for i in range(2, 12)]
    far = [feature(i) for i in range(20, 30)]
    matrix = np.stack(near + far)
    selected = diverse_select(matrix, np.ones(len(matrix), dtype=np.float32), 6)
    assert selected.shape == (6, 16)


def test_geometry():
    assert geometry_similarity(2.0, 2.0) > 0.99
    assert geometry_similarity(2.0, 3.0) < geometry_similarity(2.0, 2.2)


def test_light_variant():
    image = np.zeros((128, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 60
    image[:, :, 1] = 110
    image[:, :, 2] = 180
    variant = illumination_variant(image)
    assert variant.shape == image.shape
    assert not np.array_equal(variant, image)


def test_same_person_match():
    left = Tracklet("cam_213", 1, 1, 20.0)
    right = Tracklet("cam_219", 3, 1, 20.0)
    for i in range(8):
        left.add(feature(i), 0.9, {"timestamp": i, "bbox": [0, 0, 40, 100]})
        right.add(feature(i), 0.9, {"timestamp": i, "bbox": [0, 0, 38, 96]})
    engine = GlobalIdentityEngine(0.60, 0.04, 0.72, 8, 3)
    assert engine.score_tracks(left, right) > 0.95


if __name__ == "__main__":
    test_diverse_gallery()
    test_geometry()
    test_light_variant()
    test_same_person_match()
    print("IDENTITY V2 TEST: OK")
