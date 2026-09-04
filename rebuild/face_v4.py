from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class FaceObservation:
    vector: np.ndarray
    quality: float
    detection: float
    width: float
    height: float
    area: float
    roll: float
    visibility: float = 0.0
    valid: bool = False


class FaceExtractorV4:
    """InsightFace buffalo_l wrapper used only when geometric face visibility is sufficient."""

    def __init__(
        self,
        model="buffalo_l",
        det_size=(640, 640),
        min_detection=0.55,
        min_size=32,
        min_quality=0.50,
        min_visibility=0.60,
        device="auto",
    ):
        try:
            import torch
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise RuntimeError(
                "Face ReID requires InsightFace. Install it separately without replacing the CUDA ONNX Runtime."
            ) from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
        if str(device).lower() == "cpu":
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
        elif str(device).lower() == "auto" and not torch.cuda.is_available():
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        self.app = FaceAnalysis(name=model, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=tuple(det_size))
        self.min_detection = float(min_detection)
        self.min_size = int(min_size)
        self.min_quality = float(min_quality)
        self.min_visibility = float(min_visibility)
        self.device = "cuda" if ctx_id == 0 else "cpu"

    @staticmethod
    def _roll(face):
        kps = getattr(face, "kps", None)
        if kps is None or len(kps) < 2:
            return 0.0
        left = np.asarray(kps[0], dtype=np.float32)
        right = np.asarray(kps[1], dtype=np.float32)
        return float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))

    @staticmethod
    def _sharp(image):
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return float(np.clip(value / 180.0, 0.0, 1.0))

    @staticmethod
    def _light(image):
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        contrast = float(gray.std())
        exposure = 1.0 - min(1.0, abs(mean - 128.0) / 128.0)
        return float(np.clip(0.65 * exposure + 0.35 * min(1.0, contrast / 55.0), 0.0, 1.0))

    @staticmethod
    def _points(face):
        for name in ("landmark_3d_68", "landmark_2d_106", "kps"):
            value = getattr(face, name, None)
            if value is not None:
                points = np.asarray(value, dtype=np.float32).reshape(-1, 2)
                if len(points) >= 5 and np.isfinite(points).all():
                    return points
        return None

    @classmethod
    def _visibility_fraction(cls, face, bbox):
        points = cls._points(face)
        if points is None:
            return 0.0
        x1, y1, x2, y2 = (float(v) for v in bbox)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        inside = ((points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2)).astype(np.float32)
        inside_ratio = float(np.mean(inside))
        shifted = points - np.asarray([x1, y1], dtype=np.float32)
        scale = np.asarray([bw, bh], dtype=np.float32)
        normalized = shifted / scale
        visible = normalized[(normalized[:, 0] >= 0.0) & (normalized[:, 0] <= 1.0) & (normalized[:, 1] >= 0.0) & (normalized[:, 1] <= 1.0)]
        if len(visible) < 3:
            return inside_ratio
        hull = cv2.convexHull(visible.astype(np.float32).reshape(-1, 1, 2))
        hull_area = float(cv2.contourArea(hull))
        coverage = float(np.clip(hull_area, 0.0, 1.0))
        return float(np.clip(0.75 * coverage + 0.25 * inside_ratio, 0.0, 1.0))

    def extract(self, image):
        if image is None or image.size == 0:
            return None
        faces = self.app.get(image)
        if not faces:
            return None
        h, w = image.shape[:2]
        best = None
        for face in faces:
            x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float32).tolist()
            fw = max(0.0, min(float(w), x2) - max(0.0, x1))
            fh = max(0.0, min(float(h), y2) - max(0.0, y1))
            if fw < self.min_size or fh < self.min_size:
                continue
            det = float(getattr(face, "det_score", 0.0))
            if det < self.min_detection:
                continue
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(w, int(x2)), min(h, int(y2))
            crop = image[iy1:iy2, ix1:ix2]
            area = float((fw * fh) / max(1.0, w * h))
            size_score = float(np.clip(np.sqrt(area) / 0.38, 0.0, 1.0))
            sharp = self._sharp(crop)
            light = self._light(crop)
            roll = self._roll(face)
            pose = float(np.clip(1.0 - abs(roll) / 35.0, 0.0, 1.0))
            visibility = self._visibility_fraction(face, (x1, y1, x2, y2))
            quality = float(np.clip(0.25 * det + 0.15 * size_score + 0.15 * sharp + 0.10 * light + 0.10 * pose + 0.25 * visibility, 0.0, 1.0))
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            vector = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.size == 0 or not np.isfinite(norm) or norm <= 0.0:
                continue
            vector /= norm
            valid = bool(visibility >= self.min_visibility and quality >= self.min_quality)
            obs = FaceObservation(vector, quality, det, fw, fh, area, roll, visibility, valid)
            if best is None or (obs.valid, obs.quality, obs.visibility) > (best.valid, best.quality, best.visibility):
                best = obs
        return best if best is not None and best.valid else None

    def describe(self):
        return f"InsightFace {self.device}, buffalo_l, SCRFD + ArcFace, visibility>={self.min_visibility:.2f}"
