
from __future__ import annotations

from dataclasses import dataclass

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
    visibility: float
    valid: bool


class FaceExtractorV4:
    """InsightFace SCRFD + ArcFace with a strict visible-face quality gate."""

    def __init__(
        self,
        model="buffalo_l",
        det_size=(640, 640),
        min_detection=0.55,
        min_size=32,
        min_quality=0.50,
        min_visibility=0.68,
        device="auto",
    ):
        try:
            import torch
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise RuntimeError(
                "Face ReID requires InsightFace installed without replacing "
                "onnxruntime-gpu."
            ) from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx = 0
        if str(device).lower() == "cpu":
            providers = ["CPUExecutionProvider"]
            ctx = -1
        elif str(device).lower() == "auto" and not torch.cuda.is_available():
            providers = ["CPUExecutionProvider"]
            ctx = -1

        self.app = FaceAnalysis(name=model, providers=providers)
        self.app.prepare(ctx_id=ctx, det_size=tuple(det_size))
        self.min_detection = float(min_detection)
        self.min_size = int(min_size)
        self.min_quality = float(min_quality)
        self.min_visibility = float(min_visibility)
        self.device = "cuda" if ctx == 0 else "cpu"

    @staticmethod
    def _roll(face):
        kps = getattr(face, "kps", None)
        if kps is None or len(kps) < 2:
            return 0.0
        left = np.asarray(kps[0], np.float32)
        right = np.asarray(kps[1], np.float32)
        return float(
            np.degrees(
                np.arctan2(
                    right[1] - left[1],
                    right[0] - left[0],
                )
            )
        )

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
        return float(
            np.clip(
                0.65 * exposure
                + 0.35 * min(1.0, contrast / 55.0),
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _points(face):
        for name in (
            "landmark_3d_68",
            "landmark_2d_106",
            "kps",
        ):
            value = getattr(face, name, None)
            if value is None:
                continue
            points = np.asarray(
                value,
                np.float32,
            ).reshape(-1, 2)
            if len(points) >= 5 and np.isfinite(points).all():
                return points
        return None

    @classmethod
    def _visibility(cls, face, bbox, parent):
        points = cls._points(face)
        if points is None:
            return 0.0

        fx1, fy1, fx2, fy2 = [
            float(x) for x in bbox
        ]
        px1, py1, px2, py2 = [
            float(x) for x in parent
        ]
        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)

        inside = (
            (points[:, 0] >= px1)
            & (points[:, 0] <= px2)
            & (points[:, 1] >= py1)
            & (points[:, 1] <= py2)
        )
        pointscore = float(np.mean(inside))

        fa = max(
            1.0,
            (fx2 - fx1) * (fy2 - fy1),
        )
        ratio = fa / max(1.0, pw * ph)
        sizescore = float(
            np.clip(
                np.sqrt(ratio) / 0.18,
                0.0,
                1.0,
            )
        )
        cy = 0.5 * (fy1 + fy2)
        head = py1 + 0.42 * ph
        positionscore = float(
            np.clip(
                1.0
                - max(0.0, cy - head)
                / max(1.0, 0.30 * ph),
                0.0,
                1.0,
            )
        )
        return float(
            np.clip(
                0.60 * pointscore
                + 0.25 * sizescore
                + 0.15 * positionscore,
                0.0,
                1.0,
            )
        )

    def extract(self, image, parent=None):
        if image is None or image.size == 0:
            return None

        faces = self.app.get(image)
        if not faces:
            return None

        h, w = image.shape[:2]
        parent = parent or (0, 0, w, h)
        best = None

        for face in faces:
            x1, y1, x2, y2 = np.asarray(
                face.bbox,
                np.float32,
            ).tolist()
            fw = max(
                0.0,
                min(float(w), x2) - max(0.0, x1),
            )
            fh = max(
                0.0,
                min(float(h), y2) - max(0.0, y1),
            )
            if fw < self.min_size or fh < self.min_size:
                continue

            detection = float(
                getattr(face, "det_score", 0.0)
            )
            if detection < self.min_detection:
                continue

            ix1 = max(0, int(x1))
            iy1 = max(0, int(y1))
            ix2 = min(w, int(x2))
            iy2 = min(h, int(y2))
            cut = image[iy1:iy2, ix1:ix2]

            area = float(
                (fw * fh) / max(1.0, w * h)
            )
            sizescore = float(
                np.clip(
                    np.sqrt(area) / 0.38,
                    0.0,
                    1.0,
                )
            )
            sharp = self._sharp(cut)
            light = self._light(cut)
            roll = self._roll(face)
            posescore = float(
                np.clip(
                    1.0 - abs(roll) / 35.0,
                    0.0,
                    1.0,
                )
            )
            visibility = self._visibility(
                face,
                (x1, y1, x2, y2),
                parent,
            )
            score = float(
                np.clip(
                    0.25 * detection
                    + 0.15 * sizescore
                    + 0.15 * sharp
                    + 0.10 * light
                    + 0.10 * posescore
                    + 0.25 * visibility,
                    0.0,
                    1.0,
                )
            )

            embedding = getattr(
                face,
                "normed_embedding",
                None,
            )
            if embedding is None:
                embedding = getattr(
                    face,
                    "embedding",
                    None,
                )
            if embedding is None:
                continue

            vector = np.asarray(
                embedding,
                np.float32,
            ).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if (
                vector.size != 512
                or not np.isfinite(norm)
                or norm <= 0
            ):
                continue
            vector /= norm

            valid = bool(
                visibility >= self.min_visibility
                and score >= self.min_quality
            )
            observation = FaceObservation(
                vector,
                score,
                detection,
                fw,
                fh,
                area,
                roll,
                visibility,
                valid,
            )
            if best is None or (
                observation.valid,
                observation.quality,
                observation.visibility,
            ) > (
                best.valid,
                best.quality,
                best.visibility,
            ):
                best = observation

        return best if best is not None and best.valid else None

    def describe(self):
        return (
            f"InsightFace {self.device}, buffalo_l, SCRFD + ArcFace, "
            f"visibility>={self.min_visibility:.2f}"
        )
