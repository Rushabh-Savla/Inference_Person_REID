from __future__ import annotations

import numpy as np

from core import Observation, OfflineReconciler, Tracklet


def vector(index: int, dim: int = 8) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    value[index] = 1.0
    return value


def make_track(camera, track_id, start, end, vectors):
    track = Tracklet(camera=camera, track_id=track_id, fps=20.0)
    for offset, emb in enumerate(vectors):
        frame = int((start + offset * 0.5) * 20)
        track.observations.append(
            Observation(
                camera=camera,
                frame=frame,
                timestamp=start + offset * 0.5,
                track_id=track_id,
                bbox=(0, 0, 100, 300),
                detection_score=0.9,
                quality=1.0,
            )
        )
        track.add_embedding(emb, 1.0)
    return track


def main():
    same_a = make_track("cam_a", 1, 0.0, 1.5, [vector(0)] * 4)
    same_b = make_track("cam_b", 5, 2.0, 3.5, [vector(0)] * 4)
    different = make_track("cam_c", 8, 2.0, 3.5, [vector(1)] * 4)
    overlap = make_track("cam_a", 9, 0.5, 2.0, [vector(0)] * 4)

    engine = OfflineReconciler(
        same_threshold=0.80,
        cross_threshold=0.80,
        min_margin=0.03,
        bank_size=8,
        max_same_camera_gap_sec=20.0,
    )

    mapping = engine.reconcile(
        {
            "cam_a:1:1": same_a,
            "cam_b:5:1": same_b,
            "cam_c:8:1": different,
            "cam_a:9:1": overlap,
        }
    )

    assert mapping["cam_a:1:1"] == mapping["cam_b:5:1"]
    assert mapping["cam_a:1:1"] != mapping["cam_c:8:1"]
    assert mapping["cam_a:1:1"] != mapping["cam_a:9:1"]

    print("CLEAN CORE TEST: OK")


if __name__ == "__main__":
    main()
