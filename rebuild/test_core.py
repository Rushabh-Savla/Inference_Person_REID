from __future__ import annotations

import numpy as np

from core import GlobalIdentityEngine, Observation, Tracklet


def vector(index: int, dim: int = 8) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    value[index] = 1.0
    return value


def make_track(camera, track_id, segment, start, vectors):
    track = Tracklet(camera=camera, track_id=track_id, segment=segment, fps=20.0)
    for offset, emb in enumerate(vectors):
        timestamp = start + offset * 0.5
        frame = int(timestamp * 20)
        track.observations.append(Observation(camera, frame, timestamp, track_id, (0, 0, 100, 300), 0.9, 1.0))
        track.add_embedding(emb, 1.0)
    return track


def main():
    same_camera_return_a = make_track("cam_a", 1, 1, 0.0, [vector(0)] * 4)
    same_camera_return_b = make_track("cam_a", 7, 1, 50.0, [vector(0)] * 4)
    cross_camera_same = make_track("cam_b", 5, 1, 10.0, [vector(0)] * 4)
    different = make_track("cam_c", 8, 1, 10.0, [vector(1)] * 4)
    overlap = make_track("cam_a", 9, 1, 1.0, [vector(0)] * 4)

    engine = GlobalIdentityEngine(threshold=0.60, margin=0.03, strong=0.72, bank_size=8)
    mapping, matches = engine.reconcile({
        same_camera_return_a.key: same_camera_return_a,
        same_camera_return_b.key: same_camera_return_b,
        cross_camera_same.key: cross_camera_same,
        different.key: different,
        overlap.key: overlap,
    })

    assert mapping[same_camera_return_a.key] == mapping[same_camera_return_b.key]
    assert mapping[same_camera_return_a.key] == mapping[cross_camera_same.key]
    assert mapping[same_camera_return_a.key] != mapping[different.key]
    assert mapping[same_camera_return_a.key] != mapping[overlap.key]
    assert any(m.left == same_camera_return_a.key or m.right == same_camera_return_a.key for m in matches)

    print("GLOBAL IDENTITY CORE TEST: OK")


if __name__ == "__main__":
    main()
