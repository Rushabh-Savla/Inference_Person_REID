from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rebuild.batch_v6 import BatchPipelineV6  # noqa: E402
from rebuild.identity_body_v6_camera_graph import GlobalIdentityBodyV6CameraGraph  # noqa: E402
from rebuild.identity_v3 import Tracklet  # noqa: E402


class BatchPipelineV6CameraGraph(BatchPipelineV6):
    """Original NVIDIA V6 pipeline with camera-graph reconciliation."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.engine = GlobalIdentityBodyV6CameraGraph(self.cfg["identity_v6"])

    @staticmethod
    def colour_signature(image: np.ndarray) -> np.ndarray | None:
        """Low-dimensional torso colour descriptor used only for cross-camera reranking.

        The central torso region avoids most background and face pixels. Hue is
        saturation-weighted while low-saturation pixels contribute a value
        histogram, making white/neutral clothing distinguishable without using
        identity-specific face information.
        """
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        if h < 40 or w < 20:
            return None
        x1, x2 = int(0.18 * w), int(0.82 * w)
        y1, y2 = int(0.12 * h), int(0.72 * h)
        torso = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1].astype(np.float32) / 255.0
        val = hsv[..., 2].astype(np.float32) / 255.0
        hue = hsv[..., 0].astype(np.float32) / 180.0

        hue_hist, _ = np.histogram(hue, bins=8, range=(0.0, 1.0), weights=sat + 0.05)
        hue_hist = hue_hist.astype(np.float32)

        neutral = sat < 0.28
        value_hist, _ = np.histogram(val[neutral], bins=4, range=(0.0, 1.0))
        value_hist = value_hist.astype(np.float32)

        descriptor = np.concatenate([hue_hist, value_hist], axis=0)
        descriptor /= np.linalg.norm(descriptor) + 1e-12
        return descriptor.astype(np.float32)

    def add_body(self, key, camera, track_id, segment, bbox, stamp, score, image, feats):
        track = self.tracks.get(key)
        if track is None:
            track = Tracklet(camera, track_id, segment, self.meta[camera]["fps"])
            self.tracks[key] = track

        signature = self.colour_signature(image)
        if signature is not None:
            old = getattr(track, "colour_signature", None)
            quality = max(0.01, float(score))
            if old is None:
                track.colour_signature = signature
            else:
                mixed = 0.75 * np.asarray(old, dtype=np.float32) + 0.25 * signature
                mixed /= np.linalg.norm(mixed) + 1e-12
                track.colour_signature = mixed.astype(np.float32)
            track.colour_quality = max(float(getattr(track, "colour_quality", 0.0)), quality)

        base = {
            "camera": camera,
            "frame": int(stamp * self.meta[camera]["fps"]),
            "timestamp": stamp,
            "track_id": track_id,
            "bbox": list(bbox),
            "detection_score": float(score),
            "quality": float(score),
        }
        for kind, vector in feats.items():
            value = base.copy()
            value["kind"] = kind
            added = track.add(vector, kind, float(score), value, self.novelty, self.bank)
            if added:
                # Keep the observation contract explicit even if a future
                # Tracklet implementation normalizes or reconstructs metadata.
                if track.observations:
                    track.observations[-1].setdefault("bbox", list(bbox))
                self.store_crop(key, kind, image)

        # Be defensive about legacy/malformed observations. The cross-camera
        # graph only needs geometry when a bbox is actually available; it should
        # never crash the entire inference run because one observation lacks it.
        boxes = [
            obs["bbox"]
            for obs in track.observations
            if isinstance(obs, dict) and "bbox" in obs and len(obs["bbox"]) == 4
        ]
        if boxes:
            boxes = np.asarray(boxes, dtype=np.float32)
            hh = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
            ww = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
            track.shape = float(np.median(hh / ww))
        if track.observations:
            track.start = min(x.get("timestamp", stamp) for x in track.observations)
            track.end = max(x.get("timestamp", stamp) for x in track.observations)
