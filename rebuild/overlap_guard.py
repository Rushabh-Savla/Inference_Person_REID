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
        collapse_rows = set()
        result = []
    else:
        matrix = np.asarray(
            [[iou(det["bbox"], item["bbox"]) for item in tracked] for det in detections],
            dtype=np.float32,
        )
        ambiguous_cols = {
            int(col)
            for col in range(matrix.shape[1])
            if int(np.sum(matrix[:, col] >= float(minimum))) >= 2
        }
        collapse_rows = {
            int(row)
            for row in range(matrix.shape[0])
            for col in ambiguous_cols
            if float(matrix[row, col]) >= float(minimum)
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
        reason = "collapse" if index in collapse_rows else "untracked"
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
                "shadow_reason": reason,
            }
        )
    return result


def carry(rows, anchors, gids_used: set[str], minimum: float = 0.15):
    """One-to-one spatial continuity for overlap/shadow rows.

    IoU alone is too brittle after a collapse because detector boxes can shift
    substantially while the person is still adjacent to the protected anchor.
    Use predicted/last boxes plus normalized center/height agreement, but keep
    the assignment strictly one-to-one.
    """
    rows = list(rows or [])
    anchors = list(anchors or [])
    used = set(gids_used)
    if not rows or not anchors:
        return {}

    matrix = np.zeros((len(rows), len(anchors)), dtype=np.float32)
    for r, row in enumerate(rows):
        rx1, ry1, rx2, ry2 = [float(x) for x in row["bbox"]]
        rcx = 0.5 * (rx1 + rx2)
        rcy = 0.5 * (ry1 + ry2)
        rh = max(1.0, ry2 - ry1)
        for a, anchor in enumerate(anchors):
            box = anchor.get("pred_bbox") or anchor.get("bbox")
            if not box:
                continue
            ax1, ay1, ax2, ay2 = [float(x) for x in box]
            acx = 0.5 * (ax1 + ax2)
            acy = 0.5 * (ay1 + ay2)
            ah = max(1.0, ay2 - ay1)
            overlap = iou(row["bbox"], box)
            distance = float(
                np.hypot(rcx - acx, rcy - acy)
                / max(rh, ah)
            )
            scale = min(rh, ah) / max(rh, ah)
            proximity = max(0.0, 1.0 - distance / 3.5)
            score = (
                0.45 * float(overlap)
                + 0.35 * float(proximity)
                + 0.20 * float(scale)
            )
            matrix[r, a] = score

    rr, cc = linear_sum_assignment(-matrix)
    result = {}
    for r, col in zip(rr.tolist(), cc.tolist()):
        gid = str(anchors[col].get("gid", ""))
        value = float(matrix[r, col])
        # A spatial carry is only a temporary overlap hypothesis. Require a
        # meaningful association and never permit the same identity twice.
        if (
            value >= float(minimum)
            and gid.startswith("G")
            and gid not in used
        ):
            result[int(r)] = gid
            used.add(gid)
    return result
