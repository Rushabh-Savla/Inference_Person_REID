from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(left, right) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in left]
    bx1, by1, bx2, by2 = [float(x) for x in right]
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    ab = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(1.0, aa + ab - inter))


def merge(tracked, detections, frame: int, minimum: float = 0.20):
    tracked = list(tracked or [])
    detections = list(detections or [])
    if not detections:
        return tracked

    if not tracked:
        matched = set()
        result = []
    else:
        matrix = np.asarray(
            [[iou(det["bbox"], item["bbox"]) for item in tracked] for det in detections],
            dtype=np.float32,
        )
        # If multiple detector boxes substantially overlap one NvDCF box,
        # that tracker observation is an occlusion-collapse result. Do not feed
        # the mixed crop into identity resolution; preserve each detector box as
        # an independent shadow observation instead.
        ambiguous_cols = {
            int(col)
            for col in range(matrix.shape[1])
            if int(np.sum(matrix[:, col] >= float(minimum))) >= 2
        }
        keep_cols = [
            col for col in range(matrix.shape[1])
            if col not in ambiguous_cols
        ]
        result = [tracked[col] for col in keep_cols]

        if keep_cols:
            reduced = matrix[:, keep_cols]
            rr, cc = linear_sum_assignment(-reduced)
            matched = {
                int(r)
                for r, col in zip(rr.tolist(), cc.tolist())
                if float(reduced[r, col]) >= float(minimum)
            }
        else:
            matched = set()

    for index, item in enumerate(detections):
        if index in matched:
            continue
        result.append(
            {
                "camera": str(item["camera"]),
                "frame": int(frame),
                "timestamp": float(item["timestamp"]),
                "track_id": -1000000 - int(frame) * 100 - int(index),
                "bbox": item["bbox"],
                "detection_score": float(item.get("detection_score", 0.0)),
                "tracker_confidence": 0.0,
                "shadow": True,
            }
        )
    return result


def carry(rows, anchors, gids_used: set[str], minimum: float = 0.15):
    rows = list(rows or [])
    anchors = list(anchors or [])
    used = set(gids_used)
    if not rows or not anchors:
        return {}

    matrix = np.asarray(
        [[iou(row["bbox"], anchor["bbox"]) for anchor in anchors] for row in rows],
        dtype=np.float32,
    )
    rr, cc = linear_sum_assignment(-matrix)
    result = {}
    for r, col in zip(rr.tolist(), cc.tolist()):
        gid = str(anchors[col]["gid"])
        value = float(matrix[r, col])
        if value >= float(minimum) and gid.startswith("G") and gid not in used:
            result[int(r)] = gid
            used.add(gid)
    return result
