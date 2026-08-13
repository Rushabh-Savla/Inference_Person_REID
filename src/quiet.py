"""
quiet.py  --  silence a C/C++ library's stderr for one specific attempt.

WHY THIS EXISTS. `cv2.VideoWriter(...)` for an unavailable codec fails by printing
to file descriptor 2 from inside FFmpeg and OpenCV's C++ layer, then returning an
object whose `isOpened()` is False. Python never sees an exception, so `try/except`
cannot suppress any of it:

    [h264_v4l2m2m @ 0x...] Could not find a valid device
    [h264_v4l2m2m @ 0x...] can't configure encoder
    [ERROR:0@77.204] global cap_ffmpeg_impl.hpp:3568 open Could not open codec ...
    [ERROR:0@77.204] global cap_ffmpeg_impl.hpp:3585 open VIDEOIO/FFMPEG: Failed ...

On this deployment h264 is never available, so that block appears on EVERY run,
immediately before the fallback that works and writes the video correctly. It has
twice been read as the run failing. Alarming output that means nothing is not free:
it trains people to ignore the console, which is where the real failures print.

SCOPE, DELIBERATELY NARROW. Use this ONLY around a call that is expected to fail and
whose failure is detected by its return value. Never wrap a whole stage: a genuine
error that prints to stderr and is then swallowed here would be invisible, which is
strictly worse than noise.

Only fd 2 is touched. Our own `print()` goes to fd 1 and is unaffected.
"""

import contextlib
import os
import sys


@contextlib.contextmanager
def suppressed_native_stderr():
    """Redirect file descriptor 2 to /dev/null for the duration of the block.

    Fail-safe: if the descriptor cannot be duplicated (a closed or exotic stderr --
    the live pipeline already wraps stdout, and a supervisor may hand us anything),
    the block simply runs with stderr intact. Silencing output is never worth
    risking the work inside.
    """
    try:
        sys.stderr.flush()
    except Exception:                                           # noqa: BLE001
        pass
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return
    devnull = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        yield
    except OSError:
        # Could not redirect -- run anyway rather than skipping the caller's work.
        yield
    finally:
        try:
            os.dup2(saved, 2)
        except OSError:
            pass
        os.close(saved)
        if devnull is not None:
            try:
                os.close(devnull)
            except OSError:
                pass
