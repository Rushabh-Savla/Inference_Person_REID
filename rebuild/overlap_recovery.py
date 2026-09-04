from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass
class OverlapParticipant:
    key: str
    bbox: Tuple[float, float, float, float]
    frame: int
    anchor: bool = False
    history: List[Tuple[int, Tuple[float, float, float, float]]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])

    def update(self, bbox: Sequence[float], frame: int) -> None:
        self.bbox = tuple(float(v) for v in bbox)
        self.frame = int(frame)
        self.history.append((self.frame, self.bbox))
        self.history = self.history[-12:]

    def predicted_center(self) -> Tuple[float, float]:
        if len(self.history) < 2:
            return self.center
        _, previous = self.history[-2]
        _, current = self.history[-1]
        px = 0.5 * (previous[0] + previous[2])
        py = 0.5 * (previous[1] + previous[3])
        cx = 0.5 * (current[0] + current[2])
        cy = 0.5 * (current[1] + current[3])
        return 2.0 * cx - px, 2.0 * cy - py


@dataclass
class OverlapEpisode:
    camera: str
    start_frame: int
    last_blocked_frame: int
    clear_frames: int = 0
    participants: Dict[str, OverlapParticipant] = field(default_factory=dict)
    exit_frame: int | None = None
    closed: bool = False

    def touch(self, key: str, bbox: Sequence[float], frame: int, anchor: bool = False) -> None:
        if key not in self.participants:
            self.participants[key] = OverlapParticipant(
                key=key,
                bbox=tuple(float(v) for v in bbox),
                frame=int(frame),
                anchor=bool(anchor),
                history=[(int(frame), tuple(float(v) for v in bbox))],
            )
        else:
            participant = self.participants[key]
            participant.anchor = participant.anchor or bool(anchor)
            participant.update(bbox, frame)

    def block(self, frame: int) -> None:
        self.last_blocked_frame = int(frame)
        self.clear_frames = 0

    def clear(self, frame: int, grace: int) -> bool:
        self.clear_frames += 1
        if self.clear_frames < max(1, int(grace)):
            return False
        self.exit_frame = int(frame)
        self.closed = True
        return True


def bbox_overlap_metrics(left: Sequence[float], right: Sequence[float]) -> Mapping[str, float]:
    ax1, ay1, ax2, ay2 = (float(v) for v in left)
    bx1, by1, bx2, by2 = (float(v) for v in right)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    min_area = min(area_a, area_b)
    max_area = max(area_a, area_b)
    return {
        "intersection": inter,
        "iou": inter / union if union > 0.0 else 0.0,
        "intersection_min_area": inter / min_area if min_area > 0.0 else 0.0,
        "intersection_max_area": inter / max_area if max_area > 0.0 else 0.0,
        "area_ratio": min_area / max_area if max_area > 0.0 else 0.0,
    }


def is_severe_overlap(metrics: Mapping[str, float], iou_min: float, intersection_min: float) -> bool:
    return bool(
        float(metrics.get("iou", 0.0)) >= float(iou_min)
        or float(metrics.get("intersection_min_area", 0.0)) >= float(intersection_min)
    )


def participant_anchor(track) -> bool:
    if track is None:
        return False
    bank = getattr(track, "state_bank", {}) or {}
    for model in ("resnet", "swin", "solider"):
        if not bank.get(model, {}).get("full"):
            return False
    return True


def recovery_sources(
    box: Sequence[float],
    frame: int,
    fps: float,
    episodes: Iterable[OverlapEpisode],
    spatial_scale: float,
    max_gap_sec: float,
) -> List[str]:
    x1, y1, x2, y2 = (float(v) for v in box)
    center = np.asarray([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float32)
    sources: List[Tuple[float, str]] = []
    for episode in episodes:
        if not episode.closed or episode.exit_frame is None:
            continue
        gap = max(0.0, (int(frame) - int(episode.exit_frame)) / max(float(fps), 1.0))
        if gap > float(max_gap_sec):
            continue
        for participant in episode.participants.values():
            if not participant.anchor:
                continue
            predicted = np.asarray(participant.predicted_center(), dtype=np.float32)
            distance = float(np.linalg.norm(center - predicted) / max(participant.height, 1.0))
            if distance <= float(spatial_scale):
                sources.append((distance, participant.key))
    sources.sort(key=lambda item: (item[0], item[1]))
    result: List[str] = []
    seen = set()
    for _, key in sources:
        if key not in seen:
            result.append(key)
            seen.add(key)
    return result


def episode_sources(episode: OverlapEpisode) -> List[str]:
    return sorted(key for key, item in episode.participants.items() if item.anchor)
