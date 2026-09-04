"""
video_source.py  --  STAGE 1 -> 2 of the pipeline:  VIDEO FILE / STREAM -> frames.

============================ THE BIG PICTURE ================================
A video FILE (like sample.mp4) is not a bunch of images -- it's a single,
compressed blob. To feed it to a neural network we must DECODE it into
individual still images called FRAMES. A video is just many frames shown
quickly (e.g. 25 frames per second = 25 FPS).

This file's ONE job: open a source (an mp4/.mov/.avi FILE, or an rtsp://
/http:// live STREAM) and hand out its frames one at a time, then stop cleanly.

We use OpenCV (`cv2`) for this. Each frame it gives us is a grid of pixels:
a NumPy array of shape (height, width, 3) -- the 3 being the Blue, Green, Red
colour channels. (OpenCV uses B-G-R order, not R-G-B. A historical quirk.)

------------------------------ FILES vs STREAMS ----------------------------
A FILE is finite and re-readable: a failed read means end-of-file, and we can
re-open it from the start later (the final re-render pass does exactly that).

A live STREAM (rtsp://, http://, ...) is neither. It has no "start" to seek to
and no natural "end"; a failed read is usually a transient network hiccup, not
the end. It also produces frames on its own clock -- if we process slower than
it emits, frames pile up in OpenCV's internal buffer and we fall further and
further BEHIND real time. So streams get three extra behaviours, all OFF for
files (which keep their original byte-for-byte behaviour):

  1. URL sources skip the on-disk existence check (there is no file to stat).
  2. A failed read triggers a bounded RECONNECT (re-open the capture) with
     backoff, instead of ending the stream on the first blip.
  3. DROP-STALE: we keep only the most RECENT frame each step (grab-and-drop
     the backlog) so latency stays bounded when N cameras outrun the CPU. The
     tradeoff is dropped frames under load, never growing lag.
============================================================================
"""

import os
import time

import cv2  # OpenCV: the library that decodes video and gives us frames.


def is_stream_path(path):
    """True for live-stream URLs (rtsp://, http://, ...) vs local file paths."""
    return isinstance(path, str) and "://" in path


# ------------------------------------------------------------------ RTSP options
#
# Two separate problems, one place (plan #28 / #29).
#
# TRANSPORT. Nothing in this project ever set an RTSP transport, so FFmpeg's
# default applied: UDP, where lost packets are simply gone. A dropped packet in
# H.265 costs a reference frame, and the smeared frames that follow feed BOTH the
# detector and the ReID crop -- so a corrupted crop poisons a tracklet prototype,
# and packet loss is random, which no threshold can explain. TCP retransmits.
#   Honesty about the evidence: J.4 measured ZERO decode errors across four LIVE
#   streams, so this is NOT the cause of the identity problems on this network --
#   the 294/682 and 207/1573 broken references were in recorded files, an artefact
#   of how those were made. TCP is set because it is the right default on a lossy
#   or busy network, not because it is a fix for something measured here.
#
# TIMEOUT is the one that has bitten. Without a socket timeout `cap.read()` can
# block INDEFINITELY, and the capture thread cannot re-check stop_event while
# blocked inside it -- so a single wedged camera makes Ctrl-C unable to stop that
# thread, and the interrupt that is supposed to reach reconcile never gets there.
# A timeout turns an indefinite block into a failed read, which the existing
# reconnect path already handles and which lets the loop see stop_event.
#
# Belt and braces, because they act at different layers:
#   * OPENCV_FFMPEG_CAPTURE_OPTIONS -> FFmpeg's own socket timeout, read at open
#     time by the FFmpeg backend, so it MUST be set before any VideoCapture is
#     constructed. Both `timeout` and `stimeout` are sent: FFmpeg renamed the RTSP
#     option, builds disagree about which they accept, and an unrecognised key is
#     ignored rather than fatal.
#   * CAP_PROP_OPEN/READ_TIMEOUT_MSEC -> OpenCV's own guard, passed per capture.
RTSP_ENV_VAR = "OPENCV_FFMPEG_CAPTURE_OPTIONS"

_STREAM_OPEN_PARAMS = []          # [propId, value, ...] for VideoCapture(...)


def apply_rtsp_options(transport="tcp", open_timeout_ms=5000,
                       read_timeout_ms=5000, ffmpeg_options=None, log=print):
    """Install RTSP transport + timeouts for every capture opened afterwards.

    Call ONCE at startup, before opening any source. Returns the option string
    actually installed (or None when disabled) so a run's log records it --
    transport is exactly the kind of setting that is invisible until it matters.

    An explicit `ffmpeg_options` string overrides the built one verbatim, so an
    FFmpeg build wanting different keys needs a config edit, not a code change.
    Set transport to "" / None and the timeouts to 0 to disable entirely and get
    the old behaviour back.
    """
    global _STREAM_OPEN_PARAMS

    params = []
    for prop, value in ((getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None),
                         open_timeout_ms),
                        (getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None),
                         read_timeout_ms)):
        if prop is not None and value and float(value) > 0:
            params += [int(prop), int(float(value))]
    _STREAM_OPEN_PARAMS = params

    if ffmpeg_options is None:
        parts = []
        if transport:
            parts.append(f"rtsp_transport;{transport}")
        # FFmpeg wants MICROseconds here, while every other timeout in this
        # project is in milliseconds. Converting in one place beats a config key
        # whose unit differs from its neighbours.
        if read_timeout_ms and float(read_timeout_ms) > 0:
            micros = int(float(read_timeout_ms) * 1000)
            parts.append(f"timeout;{micros}")
            parts.append(f"stimeout;{micros}")
        ffmpeg_options = "|".join(parts)

    if not ffmpeg_options:
        log("[video] RTSP transport/timeout options DISABLED by config "
            "(FFmpeg defaults apply: UDP, no socket timeout -- a wedged camera "
            "can then block a capture thread past Ctrl-C).")
        return None

    os.environ[RTSP_ENV_VAR] = ffmpeg_options
    log(f"[video] RTSP options: {ffmpeg_options}"
        + (f"  (+ OpenCV open/read timeout {open_timeout_ms}/{read_timeout_ms} ms)"
           if params else ""))
    return ffmpeg_options


def rtsp_options_from_config(cfg, log=print):
    """Apply the `source.rtsp` block (all keys optional)."""
    rtsp_cfg = ((cfg or {}).get("source", {}) or {}).get("rtsp", {}) or {}
    return apply_rtsp_options(
        transport=rtsp_cfg.get("transport", "tcp"),
        open_timeout_ms=rtsp_cfg.get("open_timeout_ms", 5000),
        read_timeout_ms=rtsp_cfg.get("read_timeout_ms", 5000),
        ffmpeg_options=rtsp_cfg.get("ffmpeg_options"),
        log=log,
    )


class VideoSource:
    """
    Wraps an OpenCV video capture for a local FILE or a live STREAM, so the
    rest of the program never touches OpenCV directly -- it just asks for
    frames.

    Usage:
        with VideoSource("sample.mp4") as cam:
            for frame in cam.frames():
                ...do something with `frame`...

        with VideoSource("rtsp://cam/stream", stop_event=ev) as cam:
            for frame in cam.frames():
                ...

    stop_event : optional threading.Event. When set, a stream stops waiting on
                 reconnects and `frames()` returns cleanly. Ignored for files.
    reconnect_attempts / reconnect_backoff / drop_stale : stream-only tuning
                 (see module docstring). Defaults are sane; overridden from
                 config.source.stream by main.py.
    """

    def __init__(self, path, stop_event=None, reconnect_attempts=5,
                 reconnect_backoff=1.0, drop_stale=True):
        # `path` is the source to read: a file path e.g. "sample.mp4", or a
        # stream URL e.g. "rtsp://...". It comes from config.yaml / --videos.
        self.path = path
        self.is_stream = is_stream_path(path)

        # Stream-only knobs (harmless/unused for files).
        self.stop_event = stop_event
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.reconnect_backoff = max(0.0, float(reconnect_backoff))
        self.drop_stale = bool(drop_stale)

        # `capture` will hold OpenCV's open-source object once we open it.
        # None means "not opened yet".
        self.capture = None

    def _stopped(self):
        return self.stop_event is not None and self.stop_event.is_set()

    def open(self):
        """
        Open the source and get it ready to read frames from.
        """
        # Fail early with a clear message if a FILE isn't actually there --
        # otherwise OpenCV fails silently and you get a confusing empty video.
        # A STREAM has no on-disk path to stat, so we skip this check for URLs
        # and let the connection attempt below be the thing that succeeds/fails.
        if not self.is_stream and not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Video file not found: {self.path!r}. "
                f"Put the file next to main.py, or set the correct path in "
                f"config.yaml under source.videos."
            )

        # This is the line that actually opens & prepares the source for decoding.
        self.capture = self._new_capture()

        # VideoCapture doesn't raise on a bad/corrupt file or an unreachable
        # stream; it just isn't "opened". So we check explicitly and complain.
        if not self.capture.isOpened():
            what = "stream" if self.is_stream else "video file"
            raise IOError(
                f"Could not open {what}: {self.path!r}. "
                + ("The URL may be unreachable, wrong, or need credentials."
                   if self.is_stream else
                   "The file may be corrupt or in an unsupported format.")
            )

        return self

    def _new_capture(self):
        """Open one capture with this project's stream settings applied.

        THE ONE place a VideoCapture is constructed. `open()` and `_reopen()` used
        to each build their own with a copy of the buffer-size hint, so a setting
        added to one silently did not apply after a reconnect -- which is exactly
        when a stream setting matters most.
        """
        if self.is_stream and _STREAM_OPEN_PARAMS:
            # The params form is the only way to pass open/read timeouts, and it
            # requires an explicit backend. FFmpeg is what handles rtsp:// here.
            cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG,
                                   list(_STREAM_OPEN_PARAMS))
        else:
            cap = cv2.VideoCapture(self.path)
        # For a live stream, keep OpenCV's internal buffer tiny so a slow consumer
        # reads the freshest frame instead of a growing backlog. Best-effort hint
        # (not all backends honour it); the drop-stale grab loop in frames() is the
        # real latency guard.
        if self.is_stream:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    def _reopen(self):
        """Release and re-open a stream capture (used for reconnect)."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.capture = self._new_capture()
        return self.capture.isOpened()

    def frames(self):
        """
        A GENERATOR that yields frames one at a time.

        A "generator" produces values lazily: each time the caller's for-loop
        asks for the next item, execution resumes here, grabs ONE frame, and
        hands it back. This means we never load the whole video into memory --
        we process it frame-by-frame.

        Each yielded `frame` is a NumPy array of shape (height, width, 3).

        FILES: identical to the original behaviour -- read frames in order,
        stop on the first failed read (end of file).

        STREAMS: keep only the most recent frame (drop-stale) so we never lag,
        and treat a failed read as a transient hiccup -> bounded reconnect,
        rather than the end of the stream. A set stop_event ends it cleanly.
        """
        if self.capture is None:
            self.open()

        if not self.is_stream:
            # ---- FILE PATH: unchanged from the original implementation. ----
            while True:
                ok, frame = self.capture.read()
                if not ok:
                    print("[video_source] End of video file.")
                    break
                yield frame
            return

        # ---- STREAM PATH ----
        consecutive_failures = 0
        while True:
            if self._stopped():
                print(f"[video_source] Stop requested; closing stream {self.path!r}.")
                break

            frame = self._read_latest()
            if frame is None:
                # Transient failure: attempt a bounded reconnect. reconnect_attempts
                # counts CONSECUTIVE failures before we give up on the stream.
                consecutive_failures += 1
                if consecutive_failures > self.reconnect_attempts:
                    print(f"[video_source] Stream {self.path!r} unavailable after "
                          f"{self.reconnect_attempts} reconnect attempts; stopping.")
                    break
                backoff = self.reconnect_backoff * consecutive_failures
                print(f"[video_source] Stream read failed "
                      f"({consecutive_failures}/{self.reconnect_attempts}); "
                      f"reconnecting in {backoff:.1f}s...")
                # Sleep in short slices so a stop_event is honoured promptly.
                slept = 0.0
                while slept < backoff and not self._stopped():
                    time.sleep(min(0.2, backoff - slept))
                    slept += 0.2
                if self._stopped():
                    break
                self._reopen()
                continue

            consecutive_failures = 0
            yield frame

    def _read_latest(self):
        """
        Read the FRESHEST available frame from a stream, dropping any backlog.

        OpenCV buffers decoded frames; if we call read() once per processed
        frame while processing slower than the stream's fps, we consume stale
        frames and lag grows without bound. So we grab() as many queued frames
        as are immediately available and retrieve() only the last one. Returns
        the frame, or None on a read failure (caller handles reconnect).
        """
        if not self.drop_stale:
            ok, frame = self.capture.read()
            return frame if ok else None

        # Grab (decode-and-discard) whatever is ALREADY BUFFERED, then retrieve the
        # most recent. The cap must be a TIME budget, not a frame count: with the
        # FFmpeg backend grab() BLOCKS waiting for the next frame instead of
        # returning False when the buffer is empty, so a fixed `for _ in range(5)`
        # discarded 4 of every 5 frames unconditionally -- a silent 5:1 decimation
        # of a healthy stream, not the backlog drain it was meant to be.
        #
        # Budgeting ~5ms fixes that: draining a real backlog is memcpy-fast and
        # stays inside the window, while a blocking grab on a live 25fps source
        # takes ~40ms and so ends the loop after the one frame we actually need.
        drain_budget = 0.005
        grabbed_any = False
        deadline = time.monotonic() + drain_budget
        while True:
            if not self.capture.grab():
                break
            grabbed_any = True
            if time.monotonic() >= deadline:
                break        # budget spent -> that grab blocked; nothing buffered
        if not grabbed_any:
            # Nothing buffered was grabbable -- fall back to a blocking read so
            # we still wait for the next frame rather than busy-spin.
            ok, frame = self.capture.read()
            return frame if ok else None
        ok, frame = self.capture.retrieve()
        return frame if ok else None

    def release(self):
        """Close the source and free the handle. Safe to call more than once."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    # ---- These two make `with VideoSource(...) as cam:` work. ----
    # A "context manager" guarantees release() runs when we leave the `with`
    # block -- even if the code inside it crashes -- so the source is always
    # closed properly.
    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
