from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from rebuild.identity_v2 import crop, unit
from reid.extractor import ReIDExtractor


class NvidiaReID:
    """NVIDIA ReIdentificationNet deployable_v1.2 ONNX adapter.

    NVIDIA's DeepStream documentation specifies RGB 256x128 input, scaling
    0.0173520736 and offsets 123.675/116.28/103.53. The raw output is not L2
    normalized, so we normalize it here before cosine matching.
    """

    def __init__(self, path: str, device: str = "cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, providers=providers)
        self.input = self.session.get_inputs()[0].name
        self.shape = self.session.get_inputs()[0].shape
        out = self.session.get_outputs()[0]
        self.output = out.name
        self.dim = int(out.shape[-1]) if isinstance(out.shape[-1], int) else -1

    def preprocess(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        rgb -= np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
        rgb *= np.float32(0.01735207357279195)
        return np.transpose(rgb, (2, 0, 1))

    def extract(self, images):
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)
        batch = np.stack([self.preprocess(x) for x in images]).astype(np.float32)
        raw = self.session.run([self.output], {self.input: batch})[0]
        return unit(raw).astype(np.float32)


def collect(video_paths, max_per_track=8, max_tracks=250):
    samples = defaultdict(list)
    for video in video_paths:
        camera = Path(video).stem
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # The benchmark expects cached detections from the same detector used by
        # the pipeline so model comparison is not contaminated by detector drift.
        cache = Path("rebuild_outputs/cache") / f"{camera}.detections.jsonl"
        if not cache.exists():
            raise SystemExit(f"Missing {cache}. Run the v2 batch once before benchmarking.")
        rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
        picked = defaultdict(list)
        for row in rows:
            key = row["tracklet_key"]
            if len(picked[key]) >= max_per_track:
                continue
            if len(picked) >= max_tracks and key not in picked:
                continue
            picked[key].append(row)
        for key, items in picked.items():
            for row in items:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]) - 1)
                ok, frame = cap.read()
                if not ok:
                    continue
                person = crop(frame, tuple(row["bbox"]))
                if person is not None and person.size:
                    samples[key].append((camera, person))
        cap.release()
    return samples


def separation(vectors, groups):
    positives = []
    negatives = []
    keys = list(groups)
    for key in keys:
        x = vectors[key]
        if len(x) > 1:
            sim = x @ x.T
            vals = sim[np.triu_indices(len(x), 1)]
            positives.extend(vals.tolist())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if a.split(":")[0] == b.split(":")[0]:
                continue
            sim = vectors[a] @ vectors[b].T
            negatives.extend(sim.reshape(-1).tolist())
    return positives, negatives


def stats(values):
    if not values:
        return {"count": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0, "max": 0.0}
    a = np.asarray(values, dtype=np.float32)
    return {"count": int(len(a)), "min": float(a.min()), "mean": float(a.mean()), "median": float(np.median(a)), "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvidia", required=True, help="NVIDIA resnet50_market1501_aicity156.onnx")
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--max-per-track", type=int, default=8)
    args = parser.parse_args()

    samples = collect(args.videos, max_per_track=args.max_per_track)
    keys = sorted(samples)
    images = [image for key in keys for _, image in samples[key]]
    offsets = {}
    pos = 0
    for key in keys:
        offsets[key] = (pos, pos + len(samples[key]))
        pos += len(samples[key])

    print(f"samples: {len(images)} tracklets: {len(keys)}")

    fast = ReIDExtractor.from_config(config_path="rebuild/config.yaml")
    t0 = time.perf_counter()
    fast_out = fast.extract_batch(images)
    fast_sec = time.perf_counter() - t0

    nvidia = NvidiaReID(args.nvidia)
    t0 = time.perf_counter()
    nvidia_out = nvidia.extract(images)
    nvidia_sec = time.perf_counter() - t0

    fast_vectors = {key: fast_out[s:e] for key, (s, e) in offsets.items()}
    nvidia_vectors = {key: nvidia_out[s:e] for key, (s, e) in offsets.items()}

    fp, fn = separation(fast_vectors, {key: True for key in keys})
    np_, nn = separation(nvidia_vectors, {key: True for key in keys})

    print("\n===== SAME-TRACK POSITIVE SIMILARITY =====")
    print("FastReID:", json.dumps(stats(fp), sort_keys=True))
    print("NVIDIA:", json.dumps(stats(np_), sort_keys=True))

    print("\n===== CROSS-CAMERA NEGATIVE SIMILARITY =====")
    print("FastReID:", json.dumps(stats(fn), sort_keys=True))
    print("NVIDIA:", json.dumps(stats(nn), sort_keys=True))

    print("\n===== INFERENCE =====")
    print(f"FastReID: {len(images) / max(fast_sec, 1e-9):.2f} crops/s")
    print(f"NVIDIA:   {len(images) / max(nvidia_sec, 1e-9):.2f} crops/s")
    print(f"NVIDIA embedding dim: {nvidia.dim}")
    print("\nNOTE: same-track positives are guaranteed same physical track, but they do NOT measure cross-camera identity accuracy. True cross-camera precision/recall requires a manually verified identity-pair file. This script deliberately does not invent labels.")


if __name__ == "__main__":
    main()
