"""
capture.py  --  STAGE: RTSP/file decode -> newest-frame slot (one thread/cam).

One `CaptureThread` per camera. It owns a `DecodeBackend`, reads frames as fast
as the source yields them, stamps each with a wall-clock `ts` (done by the
backend at read, per v5 §9), wraps it in a `Frame`, and drops it into a
`NewestSlot` (size 1). Downstream sees only the freshest frame -> latency never
accumulates from a slow consumer.

STREAM vs FILE (the key branch):
  * stream (rtsp/http): a failed read is a transient hiccup -> bounded,
    interruptible reconnect via the backend; `frame_index` keeps counting across
    the gap; only give up (mark the camera dead) after the retry budget.
  * file: a failed read is END-OF-FILE -> the camera is simply done (used for
    local testing of the live path against a recording).

Nothing here touches models, windows, or identity. It only produces frames.
"""

import threading
import time

from live.frame import Frame


class CaptureThread(threading.Thread):
    def __init__(self, cam, backend, slot, stop_event,
                 reconnect_base_delay=1.0, reconnect_max_delay=30.0,
                 reconnect_attempts=5, device="cpu", time_offset_sec=0.0):
        super().__init__(name=f"capture-{cam}", daemon=True)
        self.cam = cam
        self.backend = backend
        self.slot = slot
        self.stop_event = stop_event
        self.reconnect_base = float(reconnect_base_delay)
        self.reconnect_max = float(reconnect_max_delay)
        self.reconnect_attempts = int(reconnect_attempts)
        self.device = device
        self.frame_index = 0
        self.reconnects = 0
        self.dead = False          # source unavailable (see #57: not permanent)
        # #57: keep trying on a slow cadence after the fast retry budget is spent.
        self.retry_forever = True
        self.dead_retry_sec = 30.0
        self.finished = False      # file ended, or dead, or stopped
        # ---- MEDIA time, for recorded sources only (see frame.py) ----
        # `time_offset_sec` shifts this camera's media timeline so several files
        # recorded concurrently line up. 0 means "this file starts at t=0", which is
        # correct when every camera's recording was started together.
        self.time_offset_sec = float(time_offset_sec or 0.0)
        self.source_fps = None       # resolved at open(); None -> no media time
        self.media_time = False      # whether frames carry source_ts

    def _sleep_interruptible(self, seconds):
        """Wait, but wake immediately on stop (never a bare sleep -> §10)."""
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.stop_event.is_set():
            time.sleep(min(0.2, end - time.monotonic()))

    def run(self):
        tag = f"[capture:{self.cam}]"
        try:
            self.backend.open()
        except Exception as e:
            print(f"{tag} could not open source: {e}")
            self.dead = True
            self.finished = True
            return

        # Media time is resolved ONCE, after open, because the rate belongs to the
        # container and cannot change mid-file. Reported either way: which clock is
        # in play decides whether every cross-camera temporal rule means anything,
        # and silently having no media time on a file run is exactly the failure
        # this exists to prevent.
        if not self.backend.is_stream:
            self.source_fps = getattr(self.backend, "source_fps", None)
            if self.source_fps:
                self.media_time = True
                print(f"{tag} recorded source: MEDIA time active "
                      f"({self.source_fps:.3f} fps, offset "
                      f"{self.time_offset_sec:+.3f}s). Cross-camera timing uses "
                      f"frame_index/fps, NOT decode wall-clock.")
            else:
                print(f"{tag} recorded source but its frame rate is UNKNOWN -> no "
                      f"media time. Cross-camera temporal rules (co-presence veto, "
                      f"geometry) will be comparing DECODE times, which on a file "
                      f"run are meaningless. Re-encode the file with a valid fps, or "
                      f"do not trust any cross-camera timing from this run.")

        consecutive_failures = 0
        while not self.stop_event.is_set():
            try:
                ok, image, ts = self.backend.read()
            except Exception as e:
                print(f"{tag} read error: {e}")
                ok, image, ts = False, None, time.time()

            if ok and image is not None:
                consecutive_failures = 0
                self.slot.put(Frame(
                    cam=self.cam, ts=ts, frame_index=self.frame_index,
                    image=image, device="cpu",   # decode is CPU here (see backend)
                    source_ts=(self.time_offset_sec
                               + self.frame_index / self.source_fps
                               if self.media_time else None),
                ))
                self.frame_index += 1
                continue

            # ---- read failed ----
            if not self.backend.is_stream:
                print(f"{tag} end of file after {self.frame_index} frames.")
                break

            # Stream hiccup -> bounded, interruptible reconnect.
            consecutive_failures += 1
            if consecutive_failures > self.reconnect_attempts:
                # #57: death used to be PERMANENT after ~15s of retries, so a 20s
                # switch reboot or a brief network blip killed that camera for the
                # rest of the session -- and if every camera hit it, the whole run
                # ended. Retry forever on a slow cadence instead: a camera that
                # comes back rejoins the run. `dead` still gates the metrics and
                # the offline-overlay, so a genuinely gone camera is still visible.
                if self.retry_forever:
                    self.dead = True
                    print(f"{tag} unavailable after {self.reconnect_attempts} "
                          f"reconnects -> marked DEAD, but still retrying every "
                          f"{self.dead_retry_sec:.0f}s in case it comes back "
                          f"(#57).")
                    consecutive_failures = 0
                    if self.stop_event.wait(self.dead_retry_sec):
                        break
                    continue
                print(f"{tag} unavailable after {self.reconnect_attempts} "
                      f"reconnects -> marking camera dead.")
                self.dead = True
                break
            delay = min(self.reconnect_max,
                        self.reconnect_base * consecutive_failures)
            print(f"{tag} read failed ({consecutive_failures}/"
                  f"{self.reconnect_attempts}); reconnecting in {delay:.1f}s...")
            self._sleep_interruptible(delay)
            if self.stop_event.is_set():
                break
            try:
                self.backend.reconnect()
                self.reconnects += 1
            except Exception as e:
                print(f"{tag} reconnect error: {e}")

        try:
            self.backend.release()
        finally:
            self.finished = True
            print(f"{tag} done ({self.frame_index} frames, "
                  f"{self.reconnects} reconnects).")
