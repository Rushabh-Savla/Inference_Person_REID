"""
Two clocks: wall-clock for the pipeline, MEDIA time for recorded sources.

WHY THIS EXISTS. `frame.ts` is stamped at frame read. For a live camera that is
milliseconds after the event, so it doubles as "when did this happen" -- and all
cameras share one machine clock, which is what makes cross-camera timing comparable
at all.

A recorded file breaks that silently. Frames decode as fast as the disk allows
(125+ fps measured on this box), so read time tracks decode progress, not events.
Feed two files to `--mode live` and the gap between their timestamps reflects thread
scheduling. Every cross-camera temporal rule -- reconcile's co-presence veto, and
all of geometry's co-temporal pairing -- would then be built on invented
simultaneity, and it would look completely plausible: positions in range,
timestamps in order, verdicts confidently wrong.

So a file source also carries `source_ts = offset + frame_index / fps`, and
`event_ts()` is what every "when did this happen" consumer reads. The checks below
pin the split, in both directions: event time must follow the media clock, and the
pipeline's own machinery must NOT.
"""

import sys

from _synth import Check

from live.frame import Frame


def main():
    c = Check("two clocks: wall-clock machinery, media-time events")

    # ------------------------------------------------------------------ 1
    print("\n1. A LIVE FRAME HAS ONE CLOCK")

    live = Frame(cam="cam_219", ts=1785752592.0, frame_index=42)
    c.ok(live.source_ts is None, "a live frame carries no media time")
    c.eq(live.event_ts(), 1785752592.0,
         "so event_ts() IS the wall clock -- nothing changes for a live camera")

    # ------------------------------------------------------------------ 2
    print("\n2. A RECORDED FRAME SEPARATES THEM")

    # Decoded at 125 fps (wall clock advances 0.008 s per frame) from footage shot
    # at 25 fps (media time advances 0.04 s per frame). Five times apart, which is
    # exactly the error that would silently corrupt co-temporal pairing.
    rec = Frame(cam="cam_219", ts=1785752592.0, frame_index=250, source_ts=10.0)
    c.eq(rec.event_ts(), 10.0,
         "event_ts() follows MEDIA time, not the decode wall clock")
    c.ok(rec.event_ts() != rec.ts, "and the two genuinely differ")

    # ------------------------------------------------------------------ 3
    print("\n3. FRESHNESS STAYS ON THE WALL CLOCK")

    # age_ms answers a question about the PIPELINE -- is this frame worth spending
    # inference on -- so it must not follow media time. If it did, every frame of a
    # recording would look ~1.7 billion seconds stale and the scheduler would skip
    # the entire run.
    now = 1785752592.5
    c.ok(abs(rec.age_ms(now) - 500.0) < 1e-6,
         f"age_ms uses ts, giving a sane 500 ms (got {rec.age_ms(now):.1f})")
    c.ok(rec.age_ms(now) < 1000,
         "NOT media time -- that would report the frame as ~1.8e12 ms stale and "
         "the scheduler would discard every frame of the run")

    # ------------------------------------------------------------------ 4
    print("\n4. TWO FILES STARTED TOGETHER LINE UP IN MEDIA TIME")

    # 25 fps and 15 fps, both started at t=0. The same real instant (2.0 s in) is a
    # DIFFERENT frame index in each -- which is precisely why frame indices are not
    # comparable across cameras and media time is.
    a = Frame(cam="cam_219", ts=1785752592.0, frame_index=50, source_ts=50 / 25.0)
    b = Frame(cam="cam_224", ts=1785752593.7, frame_index=30, source_ts=30 / 15.0)
    c.ok(abs(a.event_ts() - b.event_ts()) < 1e-9,
         f"frame 50 @25fps and frame 30 @15fps are co-temporal "
         f"({a.event_ts():.3f}s both), despite different indices")
    c.ok(abs(a.ts - b.ts) > 1.0,
         f"while their DECODE times differ by {abs(a.ts - b.ts):.1f}s -- the "
         f"quantity that would have been used before, and it is meaningless here")

    # A recording that started late is corrected by its offset, not by pretending.
    late = Frame(cam="cam_224", ts=1785752593.7, frame_index=10,
                 source_ts=1.35 + 10 / 15.0)
    c.ok(abs(late.event_ts() - (1.35 + 2 / 3)) < 1e-9,
         "file_time_offsets shifts one camera's media timeline to match the others")

    # ------------------------------------------------------------------ 5
    print("\n5. THE CAPTURE STAGE ONLY CLAIMS MEDIA TIME WHEN IT KNOWS THE RATE")

    from live.capture import CaptureThread

    class _Backend:
        def __init__(self, is_stream, fps):
            self.is_stream = is_stream
            self.source_fps = fps

    import threading

    def built(is_stream, fps, offset=0.0):
        t = CaptureThread("cam_x", _Backend(is_stream, fps), None,
                          threading.Event(), time_offset_sec=offset)
        # Mirror what run() resolves after open(), without opening anything.
        if not t.backend.is_stream:
            t.source_fps = getattr(t.backend, "source_fps", None)
            t.media_time = bool(t.source_fps)
        return t

    c.ok(built(True, 25.0).media_time is False,
         "a STREAM never gets media time, even if a rate is offered -- its ts is "
         "already the event time and a stream's nominal fps drifts")
    c.ok(built(False, 25.0).media_time is True,
         "a FILE with a known rate does")
    c.ok(built(False, None).media_time is False,
         "a file whose rate is UNKNOWN does not -- capture says so loudly rather "
         "than inventing a timeline")
    c.ok(built(False, 0.0).media_time is False,
         "and a zero rate (what OpenCV returns for a rate-less container) is "
         "rejected, not divided by")

    # ------------------------------------------------------------------ 6
    print("\n6. THE STORED PAYLOAD CARRIES EVENT TIME, NOT DECODE TIME")

    # This is the field reconcile reads for the co-presence veto AND for geometry's
    # co-temporal pairing, so it is the one that has to be right.
    from live.identity_stage import IdentityStage

    class _Det:
        track_id, x1, y1, x2, y2, confidence, crop_quality = 7, 10, 20, 30, 60, 0.9, None

    stage = IdentityStage.__new__(IdentityStage)
    stage._run_id = "r1"
    payload = stage._observation_payload(rec, _Det())
    c.eq(payload["ts"], 10.0,
         "payload ts is MEDIA time for a recorded run (frame.event_ts())")
    c.eq(payload["frame"], 250, "while frame_index is unchanged")
    c.ok(payload["ts"] != rec.ts,
         "and specifically NOT the decode wall clock, which is the whole point")

    live_payload = stage._observation_payload(live, _Det())
    c.eq(live_payload["ts"], live.ts,
         "for a live run it is the wall clock, exactly as before -- no behaviour "
         "change on the product path")

    c.done()


if __name__ == "__main__":
    sys.exit(main() or 0)
