import numpy as np

from rebuild.identity_v3 import GlobalIdentityV3, Identity, Tracklet, unit


def feat(seed, dim=32):
    rng = np.random.default_rng(seed)
    return unit(rng.normal(size=dim).astype(np.float32))


def identity_from(track):
    identity = Identity("TEST")
    identity.add(track, True, 12, 0.70)
    return identity


def same():
    a = Tracklet("cam_213", 1, 1, 20.0)
    b = Tracklet("cam_224", 7, 1, 20.0)
    for i in range(6):
        v = feat(i)
        meta = {"timestamp": float(i), "bbox": [0, 0, 40, 100]}
        assert a.add(v, "full", 0.9, meta, 0.985, 24)
        b.add(v, "full", 0.9, meta, 0.985, 24)
    engine = GlobalIdentityV3(0.60, 0.03, 0.74, 2, 12, 0.985, 0.70, 3)
    score, support = engine.score(b, identity_from(a))
    assert score > 0.85 and support >= 2


def part():
    a = Tracklet("cam_213", 1, 1, 20.0)
    b = Tracklet("cam_224", 2, 1, 20.0)
    for i in range(3):
        v = feat(i)
        meta = {"timestamp": float(i), "bbox": [0, 0, 40, 100]}
        a.add(v, "upper", 0.9, meta, 0.985, 24)
        b.add(v, "upper", 0.9, meta, 0.985, 24)
    engine = GlobalIdentityV3()
    score, support = engine.score(b, identity_from(a))
    assert score > 0.80 and support >= 2


def gallery_assignment():
    first = Tracklet("cam_213", 1, 1, 20.0)
    second = Tracklet("cam_224", 2, 1, 20.0)
    for i in range(4):
        v = feat(i)
        meta1 = {"timestamp": float(i), "bbox": [0, 0, 40, 100]}
        meta2 = {"timestamp": float(i + 10), "bbox": [0, 0, 42, 102]}
        first.add(v, "full", 0.9, meta1, 0.985, 24)
        second.add(v, "full", 0.9, meta2, 0.985, 24)

    engine = GlobalIdentityV3(0.60, 0.03, 0.74, 2, 12, 0.985, 0.70, 3)
    tracks = {first.key: first, second.key: second}
    mapping, decisions = engine.run(tracks)
    assert mapping[first.key] == mapping[second.key]
    assert any(item.reason == "reidentified" for item in decisions)


if __name__ == "__main__":
    same()
    part()
    gallery_assignment()
    print("IDENTITY V3 TEST: OK")
