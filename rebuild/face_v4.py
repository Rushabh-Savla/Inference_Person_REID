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


class FaceExtractorV4:
    """InsightFace buffalo_l wrapper used only for face evidence."""

    def __init__(self, model="buffalo_l", det_size=(640, 640), min_detection=0.55,
                 min_size=32, min_quality=0.42, device="auto"):
        try:
            import torch
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise RuntimeError(
                "V4 face recognition requires insightface. Run `pip install -r requirements.txt`."
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
            quality = float(np.clip(0.40 * det + 0.25 * size_score + 0.20 * sharp + 0.10 * light + 0.05 * pose, 0.0, 1.0))
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            vector = np.asarray(embedding, dtype=np.float32)
            vector /= np.linalg.norm(vector) + 1e-12
            obs = FaceObservation(vector, quality, det, fw, fh, area, roll)
            if best is None or obs.quality > best.quality:
                best = obs
        return best if best is not None and best.quality >= self.min_quality else None

    def describe(self):
        return f"InsightFace {self.device}, buffalo_l, SCRFD + ArcFace"
