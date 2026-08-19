import numpy as np

from rebuild.identity_v3 import Feature, Tracklet
from rebuild.identity_v4 import GlobalIdentityV4


def feature(vector, kind="full", quality=0.9, camera="cam_a"):
    value = np.asarray(vector, dtype=np.float32)
    value /= np.linalg.norm(value) + 1e-12
    return Feature(value, kind, quality, camera, 0.0, {})


def test_body_reid():
    cfg = {"match_threshold": 0.61, "match_margin": 0.01, "strong_threshold": 0.74,
           "support": 1, "gallery": 8, "new_count": 1}
    engine = GlobalIdentityV4(cfg)
    base = np.zeros(8, dtype=np.float32)
    base[0] = 1.0
    t1 = Tracklet("cam_a", 1, 1, 20.0)
    t1.features = [feature(base, camera="cam_a"), feature(base, camera="cam_a")]
    t1.start, t1.end = 0.0, 1.0
    mapping, _ = engine.run({t1.key: t1}, {})
    assert mapping[t1.key] == "G000001"

    t2 = Tracklet("cam_a", 2, 1, 20.0)
    t2.features = [feature(base, camera="cam_a"), feature(base, camera="cam_a")]
    t2.start, t2.end = 10.0, 11.0
    decision = engine.assign(t2, [], {t1.key: t1, t2.key: t2})
    assert decision.gid == "G000001"


def test_face_reid():
    cfg = {"match_threshold": 0.99, "match_margin": 0.01, "strong_threshold": 0.99,
           "support": 1, "gallery": 8, "new_count": 3,
           "face_threshold": 0.65, "face_strong_threshold": 0.75,
           "face_margin": 0.01, "face_quality": 0.40,
           "face_strong_quality": 0.60}
    engine = GlobalIdentityV4(cfg)
    body = np.zeros(8, dtype=np.float32)
    body[0] = 1.0
    face = np.zeros(16, dtype=np.float32)
    face[0] = 1.0
    t1 = Tracklet("cam_a", 1, 1, 20.0)
    t1.features = [feature(body, camera="cam_a")] * 3
    mapping, _ = engine.run({t1.key: t1}, {t1.key: [feature(face, "face", 0.95, "cam_a")]})
    assert mapping[t1.key] == "G000001"

    t2 = Tracklet("cam_b", 2, 1, 20.0)
    t2.features = []
    decision = engine.assign(
        t2,
        [feature(face, "face", 0.95, "cam_b")],
        {t1.key: t1, t2.key: t2},
    )
    assert decision.gid == "G000001"


def main():
    test_body_reid()
    test_face_reid()
    print("IDENTITY V4 TEST: OK")


if __name__ == "__main__":
    main()
