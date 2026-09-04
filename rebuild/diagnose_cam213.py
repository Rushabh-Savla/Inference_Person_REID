from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def stats(values: Iterable[float]) -> dict:
    a = np.asarray(list(values), dtype=np.float32)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max()),
    }


def image_metrics(path: Path) -> dict | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return {
        "width": w, "height": h, "area": w * h,
        "aspect": h / max(1.0, float(w)),
        "brightness": float(gray.mean()), "contrast": float(gray.std()),
        "saturation": float(hsv[..., 1].mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def crop_report(crop_root: Path) -> None:
    per_cam: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Counter[str] = Counter()
    if not crop_root.exists():
        raise SystemExit(f"Missing crop cache: {crop_root}")
    for folder in sorted(p for p in crop_root.iterdir() if p.is_dir()):
        parts = folder.name.split("_")
        if len(parts) < 4:
            continue
        camera = "_".join(parts[:2])
        for path in sorted(folder.glob("*.jpg")):
            metric = image_metrics(path)
            if metric is None:
                continue
            kind = path.stem.split("_", 1)[-1]
            counts[f"{camera}:{kind}"] += 1
            for key, value in metric.items():
                per_cam[camera][f"{kind}:{key}"].append(float(value))
    print("===== CROP / IMAGE REPORT =====")
    for camera in sorted(per_cam):
        print(f"\n[{camera}]")
        kinds = sorted({k.split(":", 1)[0] for k in per_cam[camera]})
        for kind in kinds:
            print(f"  {kind}: files={counts[f'{camera}:{kind}']}")
            for metric in ("width", "height", "area", "aspect", "brightness", "contrast", "saturation", "sharpness"):
                print(f"    {metric}: {stats(per_cam[camera].get(f'{kind}:{metric}', []))}")


def load_tracks(root: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    meta = json.loads((root / "tracklets_v6.json").read_text(encoding="utf-8"))
    arrays = np.load(root / "tracklets_v6.npz")
    return meta, {key: np.asarray(arrays[key], dtype=np.float32) for key in arrays.files}


def load_mapping(output: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    path = output / "identity_debug_v6.jsonl"
    if not path.exists():
        return mapping
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key, gid = row.get("key"), row.get("gid")
        if key and gid and gid not in {"PENDING", "UNKNOWN"}:
            mapping[str(key)] = str(gid)
    return mapping


def pair_distribution(keys_a: list[str], keys_b: list[str], arrays: dict[str, np.ndarray], positive_only: bool = False, mapping: dict[str, str] | None = None):
    values = []
    for left in keys_a:
        for right in keys_b:
            if left == right:
                continue
            if positive_only and (mapping is None or mapping.get(left) != mapping.get(right)):
                continue
            sims = np.asarray(arrays[left] @ arrays[right].T, dtype=np.float32).reshape(-1)
            if sims.size:
                values.extend(sims.tolist())
    return stats(values)


def ranking_report(by_cam: dict[str, list[str]], arrays: dict[str, np.ndarray], mapping: dict[str, str]) -> None:
    gid_to_features: dict[str, list[np.ndarray]] = defaultdict(list)
    for key, gid in mapping.items():
        if key in arrays:
            gid_to_features[gid].extend(list(arrays[key]))
    print("\n===== CAMERA 213 RANKING DIAGNOSTIC =====")
    for key in by_cam.get("cam_213", []):
        query = arrays[key]
        scored = []
        for gid, values in gid_to_features.items():
            gallery = np.asarray(values, dtype=np.float32)
            score = float(np.max(query @ gallery.T)) if gallery.size else -1.0
            scored.append((score, gid))
        scored.sort(reverse=True)
        true_gid = mapping.get(key)
        rank = next((i + 1 for i, (_, gid) in enumerate(scored) if gid == true_gid), None)
        print(f"{key} true={true_gid} rank={rank} top5={[(g, round(s, 4)) for s, g in scored[:5]]}")


def embedding_report(root: Path, output: Path) -> None:
    meta, arrays = load_tracks(root)
    mapping = load_mapping(output)
    by_cam: dict[str, list[str]] = defaultdict(list)
    for row in meta:
        key = row["key"]
        if key in arrays:
            by_cam[row["camera"]].append(key)

    print("\n===== EMBEDDING REPORT =====")
    for camera, keys in sorted(by_cam.items()):
        vectors = np.concatenate([arrays[k] for k in keys], axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        print(camera, f"tracks={len(keys)} features={len(vectors)} norms={stats(norms)}")
        for kind in ("full", "light", "upper", "lower"):
            q = [float(f.get("quality", 0.0)) for row in meta if row["camera"] == camera and row["key"] in arrays for f in row.get("features", []) if f.get("kind") == kind]
            if q:
                print(f"  {kind} quality: {stats(q)}")

    cams = sorted(by_cam)
    print("\n===== ALL PAIRWISE COSINE =====")
    for i, left in enumerate(cams):
        for right in cams[i:]:
            print(f"{left} <-> {right}: {pair_distribution(by_cam[left], by_cam[right], arrays)}")

    if mapping:
        print("\n===== CURRENT-GID POSITIVE CROSS-CAMERA PAIRS =====")
        for left, right in (("cam_213", "cam_219"), ("cam_213", "cam_224"), ("cam_219", "cam_224")):
            print(f"{left} -> {right}: {pair_distribution(by_cam[left], by_cam[right], arrays, True, mapping)}")
        ranking_report(by_cam, arrays, mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Camera 213 crops and NVIDIA ReID embeddings from the protected V6 run.")
    parser.add_argument("--output", default="rebuild_outputs")
    args = parser.parse_args()
    output = Path(args.output)
    root = output / "cache_v6"
    if not root.exists():
        raise SystemExit(f"Missing {root}; run the protected V6 baseline first.")
    crop_report(root / "crops")
    embedding_report(root, output)


if __name__ == "__main__":
    main()
