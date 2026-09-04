from __future__ import annotations

from typing import Mapping


def sample_interval(config: Mapping[str, object], default: int = 5) -> int:
    """Return the validated frame interval used for ReID evidence gathering."""
    value = int(config.get("interval", default))
    return max(1, value)


def part_interval(config: Mapping[str, object], interval: int | None = None) -> int:
    """Return the validated interval used for illumination/part variants."""
    base = sample_interval(config) if interval is None else max(1, int(interval))
    value = int(config.get("part_interval", base))
    return max(base, value)
