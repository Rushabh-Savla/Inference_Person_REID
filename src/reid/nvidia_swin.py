from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort


class NVIDIASwinReIDExtractor:
    """NVIDIA ReIdentificationNet Transformer Swin-Base 1024-D adapter."""

    name = "NVIDIA ReIdentificationNet Transformer"
    expected_dim = 1024

    def __init__(self, weights: str, device: str = "cuda", max_batch: int = 16):
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"NVIDIA Swin Base weights not found: {self.weights}")
        if str(device).lower() != "cuda":
            raise ValueError("NVIDIA Swin Base requires CUDA")
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is unavailable")
        self.session = ort.InferenceSession(
            str(self.weights),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        loaded = self.session.get_providers()
        if "CUDAExecutionProvider" not in loaded:
            raise RuntimeError(f"NVIDIA Swin did not load CUDA: {loaded}")
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        if len(inp.shape) != 4 or inp.shape[1] != 3:
            raise RuntimeError(f"Invalid Swin input shape: {inp.shape}")
        if not isinstance(inp.shape[2], int) or not isinstance(inp.shape[3], int):
            raise RuntimeError(f"Swin input H/W must be static: {inp.shape}")
        self.input_name = inp.name
        self.output_name = out.name
        self.input_size = (int(inp.shape[2]), int(inp.shape[3]))
        self.embedding_dim = int(out.shape[-1]) if isinstance(out.shape[-1], int) else self._probe_dim()
        if self.embedding_dim != self.expected_dim:
            raise RuntimeError(f"Expected Swin 1024-D output, got {self.embedding_dim}")
        self.max_batch = max(1, int(max_batch))

    def _probe_dim(self) -> int:
        h, w = self.input_size
        value = np.asarray(self.session.run([self.output_name], {self.input_name: np.zeros((1, 3, h, w), np.float32)})[0])
        return int(value.reshape(value.shape[0], -1).shape[-1])

    def describe(self) -> str:
        h, w = self.input_size
        return f"{self.name} Swin-Base {self.embedding_dim}-D {h}x{w} CUDA"

    @staticmethod
    def _unit(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, np.float32)
        norm = np.linalg.norm(value, axis=1, keepdims=True)
        if np.any(~np.isfinite(norm)) or np.any(norm <= 0):
            raise RuntimeError("Swin produced invalid embeddings")
        return value / (norm + 1e-12)

    def _prep(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            raise ValueError("Empty person crop")
        h, w = self.input_size
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        mean = np.asarray([0.5, 0.5, 0.5], np.float32).reshape(1, 1, 3)
        x = (x - mean) / mean
        return np.transpose(x, (2, 0, 1)).astype(np.float32)

    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, self.embedding_dim), np.float32)
        output = []
        for start in range(0, len(crops), self.max_batch):
            batch = np.stack([self._prep(x) for x in crops[start:start + self.max_batch]])
            value = np.asarray(self.session.run([self.output_name], {self.input_name: batch})[0])
            value = value.reshape(value.shape[0], -1).astype(np.float32)
            if value.shape[1] != self.embedding_dim:
                raise RuntimeError(f"Swin output changed shape: {value.shape}")
            output.append(self._unit(value))
        return np.concatenate(output, axis=0)

    def extract(self, crop: np.ndarray) -> np.ndarray:
        return self.extract_batch([crop])[0]
