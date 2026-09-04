"""
frame.py  --  the Frame carrier that travels through every live stage.

One object per captured frame, owned by exactly one stage at a time (capture ->
slot -> scheduler -> inference -> identity -> render -> writer). It carries the
pixels plus the metadata every downstream stage needs.

CRITICAL RULE (v5 plan §9): `ts` is the wall-clock time.time() stamped ONCE at
frame read, and is carried UNCHANGED to the payload/logs. Never regenerate a
timestamp later (after inference, at render, ...) or cross-camera temporal logic
(freshness, topology) silently skews. All cameras share this one machine clock,
which is exactly what makes cross-camera time comparable.

TWO CLOCKS, AND WHY (2026-08-03). `ts` answers "when did this frame reach the
pipeline" -- the right question for the machinery: scheduler freshness, writer
pacing, reconnect handling. For a LIVE camera it is also the right answer to "when
did this happen in the world", because the read follows the event by milliseconds.

For a RECORDED file it is not. Frames are decoded as fast as the disk allows (125+
fps in practice), so `ts` measures decode progress, not events. Two files read in
parallel get timestamps whose difference reflects thread scheduling -- so anything
cross-camera and time-sensitive, above all geometry's co-temporal pairing, would be
built on invented simultaneity that looks entirely plausible.

Hence `source_ts`: MEDIA time, `offset + frame_index / source_fps`, set only for
file sources. Consumers that ask "when did this happen in the world" use
`event_ts()`; the machinery keeps using `ts`. Never conflate them.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Frame:
    cam: str                 # camera / source name (also the output_<cam>.mp4 label)
    ts: float                # wall-clock time.time() at READ -- never regenerated
    frame_index: int         # per-camera monotonic counter (continues across reconnect)
    # MEDIA time for recorded sources: offset + frame_index / source_fps. None for a
    # live stream, where `ts` already is the event time. See the module docstring.
    source_ts: Optional[float] = None
    image: Any = None        # np.ndarray (CPU, BGR) or torch.Tensor (GPU); device below
    device: str = "cpu"      # "cpu" | "cuda:N" -- where `image` currently lives
    # Filled in by later stages; kept here so the frame is the single carrier.
    detections: Any = None   # per-stage results attached downstream
    meta: Dict[str, Any] = field(default_factory=dict)

    def age_ms(self, now: float) -> float:
        """Milliseconds since capture, measured against a wall-clock `now`
        (time.time()). Used by the scheduler's freshness check.

        Deliberately uses `ts`, not `event_ts()`: freshness is a question about the
        PIPELINE (is this frame worth spending inference on), so it must stay on the
        wall clock even when the source is a recording.
        """
        return (now - self.ts) * 1000.0

    def event_ts(self) -> float:
        """When this frame's contents happened, on a clock comparable ACROSS cameras.

        Media time for a recording, wall-clock for a live stream. This is what any
        temporal identity rule must use -- the co-presence veto, the geometric
        reachability check, the stored payload's `ts` -- because all of them compare
        one camera against another.
        """
        return self.ts if self.source_ts is None else self.source_ts
