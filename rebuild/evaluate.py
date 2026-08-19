from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict

import numpy as np

from core import GlobalIdentityEngine, Observation, Tracklet


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


def load_tracklets(root: Path) -> Dict[str, Tracklet]:
    rows = json.loads((root / "cache" / "tracklets.json").read_text(encoding="utf-8"))
    packed = np.load(root / "cache" / "tracklet_embeddings.npz")
    out: Dict[str, Tracklet] = {}
    for row in rows:
        track = Tracklet(
            camera=row["camera"],
            track_id=int(row["track_id"]),
            segment=int(row["segment"]),
            fps=float(row["fps"]),
        )
        start = float(row["start"])
        end = float(row["end"])
        track.observations = [
            Observation(track.camera, 0, start, track.track_id, (0, 0, 1, 1), 1.0, 1.0),
            Observation(track.camera, 0, end, track.track_id, (0, 0, 1, 1), 1.0, 1.0),
        ]
        matrix = np.asarray(packed[row["key"]], dtype=np.float32)
        track.embeddings = [v for v in matrix]
        track.embedding_quality = [float(v) for v in row["quality"]]
        out[row["key"]] = track
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="rebuild_outputs")
    parser.add_argument("--labels", default=None, help="JSONL with {a,b,verdict:same|different}")
    args = parser.parse_args()

    root = Path(args.run)
    tracklets = load_tracklets(root)
    mapping = json.loads((root / "track_to_global.json").read_text(encoding="utf-8"))
    ids = {}
    for key, gid in mapping.items():
        ids.setdefault(gid, set()).add(tracklets[key].camera)

    overlap_violations = []
    for left, right in combinations(mapping, 2):
        if mapping[left] != mapping[right]:
            continue
        a, b = tracklets[left], tracklets[right]
        if a.camera == b.camera and not (a.end < b.start or b.end < a.start):
            overlap_violations.append((left, right, mapping[left]))

    print("===== CLEAN REID EVALUATION =====")
    print(f"tracklets: {len(mapping)}")
    print(f"global IDs: {len(set(mapping.values()))}")
    print(f"multi-camera IDs: {sum(len(v) > 1 for v in ids.values())}")
    print(f"same-camera overlap violations: {len(overlap_violations)}")
    for row in overlap_violations[:20]:
        print("  VIOLATION", row)

    if not args.labels:
        print("No ground-truth labels supplied; identity accuracy is not claimed.")
        return

    engine = GlobalIdentityEngine(0.60, 0.04, 0.72, 8)
    positives, negatives = [], []
    missing = 0
    for line in Path(args.labels).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("verdict") not in {"same", "different"}:
            continue
        a, b = row["a"], row["b"]
        if a not in tracklets or b not in tracklets:
            missing += 1
            continue
        score = engine.score(tracklets[a], tracklets[b])
        (positives if row["verdict"] == "same" else negatives).append(score)

    print(f"labelled same pairs: {len(positives)}")
    print(f"labelled different pairs: {len(negatives)}")
    print(f"missing labelled tracklets: {missing}")
    print(f"ROC AUC: {auc(positives, negatives):.4f}")
    if positives:
        print(f"same: mean={np.mean(positives):.4f} min={np.min(positives):.4f} p10={np.percentile(positives, 10):.4f} median={np.median(positives):.4f}")
    if negatives:
        print(f"different: mean={np.mean(negatives):.4f} max={np.max(negatives):.4f} p90={np.percentile(negatives, 90):.4f} median={np.median(negatives):.4f}")


if __name__ == "__main__":
    main()
