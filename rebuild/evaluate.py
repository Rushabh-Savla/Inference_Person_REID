from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict

import numpy as np

from core import Tracklet, OfflineReconciler, Observation


def auc(positive, negative):
    if not positive or not negative:
        return float("nan")
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    wins = 0.0
    for value in positive:
        wins += float(np.sum(value > negative))
        wins += 0.5 * float(np.sum(value == negative))
    return wins / float(len(positive) * len(negative))


def load_tracklets(root: Path):
    rows = json.loads((root / "cache" / "tracklets.json").read_text(encoding="utf-8"))
    packed = np.load(root / "cache" / "tracklet_embeddings.npz")
    out: Dict[str, Tracklet] = {}
    for row in rows:
        key = row["key"]
        track = Tracklet(
            camera=row["camera"],
            track_id=int(row["track_id"]),
            fps=float(row["fps"]),
        )
        start = float(row["start"])
        end = float(row["end"])
        track.observations = [
            Observation(
                camera=track.camera,
                frame=0,
                timestamp=start,
                track_id=track.track_id,
                bbox=(0, 0, 1, 1),
                detection_score=1.0,
                quality=1.0,
            ),
            Observation(
                camera=track.camera,
                frame=0,
                timestamp=end,
                track_id=track.track_id,
                bbox=(0, 0, 1, 1),
                detection_score=1.0,
                quality=1.0,
            ),
        ]
        matrix = np.asarray(packed[key], dtype=np.float32)
        track.embeddings = [row for row in matrix]
        track.embedding_quality = [float(value) for value in row["quality"]]
        out[key] = track
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="rebuild_outputs")
    parser.add_argument("--labels", default=None)
    args = parser.parse_args()

    root = Path(args.run)
    tracklets = load_tracklets(root)
    mapping = json.loads((root / "track_to_global.json").read_text(encoding="utf-8"))

    camera_ids = {}
    for key, gid in mapping.items():
        camera = key.split(":", 1)[0]
        camera_ids.setdefault(gid, set()).add(camera)

    overlapping = []
    for left, right in combinations(mapping, 2):
        if mapping[left] != mapping[right]:
            continue
        if tracklets[left].camera != tracklets[right].camera:
            continue
        if not (tracklets[left].end < tracklets[right].start or tracklets[right].end < tracklets[left].start):
            overlapping.append((left, right, mapping[left]))

    print("===== CLEAN REID EVALUATION =====")
    print(f"tracklets: {len(mapping)}")
    print(f"global IDs: {len(set(mapping.values()))}")
    print(f"multi-camera IDs: {sum(len(value) > 1 for value in camera_ids.values())}")
    print(f"same-camera overlap violations: {len(overlapping)}")
    for left, right, gid in overlapping[:20]:
        print(f"  VIOLATION {gid}: {left} <-> {right}")

    if not args.labels:
        print("No labels supplied; accuracy is intentionally not invented.")
        return

    positives = []
    negatives = []
    missing = 0
    for line in Path(args.labels).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("verdict") not in {"same", "different"}:
            continue
        left = row["a"]
        right = row["b"]
        if left not in tracklets or right not in tracklets:
            missing += 1
            continue
        score = OfflineReconciler.score(tracklets[left], tracklets[right], 8)
        if row["verdict"] == "same":
            positives.append(score)
        else:
            negatives.append(score)

    print(f"labelled same pairs: {len(positives)}")
    print(f"labelled different pairs: {len(negatives)}")
    print(f"missing labelled tracklets: {missing}")
    print(f"ROC AUC: {auc(positives, negatives):.4f}")
    if positives:
        print(f"same score mean: {np.mean(positives):.4f}")
        print(f"same score min: {np.min(positives):.4f}")
    if negatives:
        print(f"different score mean: {np.mean(negatives):.4f}")
        print(f"different score max: {np.max(negatives):.4f}")


if __name__ == "__main__":
    main()
