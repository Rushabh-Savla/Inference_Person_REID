from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort


class NVIDIAReIDExtractor:
    """NVIDIA ReIdentificationNet deployable ONNX inference.

    NVIDIA documents ReIdentificationNet as a ResNet-50 person embedding model
    with RGB input at 256x128. The TAO preprocessing uses ImageNet mean/std;
    this implementation keeps that recipe explicit and L2-normalizes the
    returned embedding before it enters the identity layer.
    """

    name = "NVIDIA ReIdentificationNet"
    input_size = (256, 128)

    def __init__(self, weights: str, device: str = "cuda", max_batch: int = 32):
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"NVIDIA ReIdentificationNet weights not found: {self.weights}\n"
                "Download resnet50_market1501_aicity156.onnx from NVIDIA NGC."
            )
        if str(device).lower() != "cuda":
            raise ValueError("NVIDIA ReIdentificationNet is mandatory for this experiment and must run with device=cuda")
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. This experiment must not silently fall back to CPU."
            )
        self.session = ort.InferenceSession(
            str(self.weights),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        providers = self.session.get_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(f"NVIDIA ReIdentificationNet did not load CUDAExecutionProvider: {providers}")
        self.max_batch = max(1, int(max_batch))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.embedding_dim = int(self.session.get_outputs()[0].shape[-1]) if isinstance(self.session.get_outputs()[0].shape[-1], int) else self._probe_dim()
        print(
            f"[reid] {self.describe()} | providers={providers} | input={self.input_name} output={self.output_name}"
        )

    def _probe_dim(self) -> int:
        dummy = np.zeros((1, 3, self.input_size[0], self.input_size[1]), dtype=np.float32)
        out = np.asarray(self.session.run([self.output_name], {self.input_name: dummy})[0])
        out = out.reshape(out.shape[0], -1)
        return int(out.shape[-1])

    def describe(self) -> str:
        return f"{self.name} (ResNet50, {self.embedding_dim}-d, 256x128, RGB/ImageNet normalization)"

    @staticmethod
    def _prep(crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            raise ValueError("Empty person crop")
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.226, 0.226, 0.226], dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std
        return np.transpose(x, (2, 0, 1)).astype(np.float32)

    @staticmethod
    def _unit(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True) + 1e-12
        return values / norms

    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        outputs = []
        for start in range(0, len(crops), self.max_batch):
            batch = np.stack([self._prep(x) for x in crops[start:start + self.max_batch]], axis=0)
            values = np.asarray(self.session.run([self.output_name], {self.input_name: batch})[0])
            values = values.reshape(values.shape[0], -1).astype(np.float32)
            outputs.append(self._unit(values))
        return np.concatenate(outputs, axis=0)

    def extract(self, crop: np.ndarray) -> np.ndarray:
        return self.extract_batch([crop])[0]
