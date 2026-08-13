"""
Plan items #28 / #29 -- RTSP transport and socket timeouts.

#29 IS THE ONE THAT COSTS RUNS. Without a socket timeout `cap.read()` can block
indefinitely, and a capture thread cannot re-check `stop_event` while it is
blocked inside that call. One wedged camera therefore makes Ctrl-C unable to stop
that thread -- and Ctrl-C is how the operator asks for the reconciled ids to be
computed. A timeout converts an indefinite block into a failed read, which the
existing reconnect path already handles.

#28 (TCP transport) is a sound default rather than a measured fix, and this test
says so: J.4 found ZERO decode errors across four live streams, so UDP packet loss
is not what has been corrupting identities on this network.

WHAT IS PINNED:
  1. the generated FFmpeg option string, including ms -> microseconds
  2. both `timeout` and `stimeout` are sent (FFmpeg renamed it; builds disagree,
     and an unrecognised key is ignored rather than fatal)
  3. the whole thing can be turned OFF, restoring FFmpeg's defaults
  4. an unreachable stream fails in BOUNDED time instead of hanging
  5. files are unaffected -- the file-batch path must keep its exact behaviour
  6. reconnect gets the same settings as the first open (they used to be built by
     two separate code paths, so a stream setting silently did not survive a
     reconnect -- precisely when it matters most)
"""

import os
import sys
import time

from _synth import Check

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import cv2                                                        # noqa: E402
from video_source import (RTSP_ENV_VAR, VideoSource,               # noqa: E402
                          apply_rtsp_options, rtsp_options_from_config)

QUIET = lambda *_a, **_k: None                                    # noqa: E731

# Reserved, unroutable documentation range (RFC 5737). A connect attempt here
# hangs rather than being refused, which is the wedged-camera case.
UNREACHABLE = "rtsp://192.0.2.1:554/stream"


def main():
    c = Check("#28/#29: RTSP transport + socket timeouts")

    print("\n1. THE OPTION STRING")
    os.environ.pop(RTSP_ENV_VAR, None)
    got = apply_rtsp_options(transport="tcp", open_timeout_ms=5000,
                             read_timeout_ms=5000, log=QUIET)
    c.eq(got, "rtsp_transport;tcp|timeout;5000000|stimeout;5000000",
         "tcp + both timeout spellings, ms converted to microseconds")
    c.eq(os.environ.get(RTSP_ENV_VAR), got,
         "installed in the env var FFmpeg reads at open time")

    c.eq(apply_rtsp_options(transport="udp", open_timeout_ms=0,
                            read_timeout_ms=0, log=QUIET),
         "rtsp_transport;udp",
         "no timeout configured -> transport only")
    c.eq(apply_rtsp_options(ffmpeg_options="rtsp_transport;http", log=QUIET),
         "rtsp_transport;http",
         "an explicit override is passed through verbatim")

    print("\n2. IT CAN BE TURNED OFF")
    os.environ.pop(RTSP_ENV_VAR, None)
    c.eq(apply_rtsp_options(transport="", open_timeout_ms=0, read_timeout_ms=0,
                            log=QUIET), None,
         "everything disabled -> nothing installed")
    c.ok(RTSP_ENV_VAR not in os.environ,
         "and the env var is not set, so FFmpeg's own defaults apply")

    print("\n3. THE CONFIG BLOCK")
    c.eq(rtsp_options_from_config({}, log=QUIET),
         "rtsp_transport;tcp|timeout;5000000|stimeout;5000000",
         "no source.rtsp block -> the documented defaults")
    c.eq(rtsp_options_from_config(
            {"source": {"rtsp": {"transport": "udp", "read_timeout_ms": 250}}},
            log=QUIET),
         "rtsp_transport;udp|timeout;250000|stimeout;250000",
         "config values are honoured")

    print("\n4. AN UNREACHABLE STREAM FAILS IN BOUNDED TIME (#29)")
    # The property is BOUNDED, not "takes exactly the timeout": on a host with no
    # route at all the connect can fail instantly, which is also fine. Asserting
    # the elapsed time equals the timeout would make this fail on such a machine
    # for no reason.
    budget_ms = 2000
    apply_rtsp_options(transport="tcp", open_timeout_ms=budget_ms,
                       read_timeout_ms=budget_ms, log=QUIET)
    t0 = time.time()
    opened = False
    try:
        VideoSource(UNREACHABLE).open()
        opened = True
    except (IOError, OSError):
        pass
    elapsed = time.time() - t0
    c.ok(not opened, "an unroutable stream does not report itself open")
    c.ok(elapsed < (budget_ms / 1000.0) + 3.0,
         f"gave up after {elapsed:.2f}s, inside the {budget_ms}ms budget "
         f"(+slack) rather than blocking indefinitely")

    print("\n5. FILES ARE UNAFFECTED")
    src = VideoSource("register_file.avi")
    c.ok(not src.is_stream, "a file is not treated as a stream")
    if os.path.exists("register_file.avi"):
        with src as cam:
            first = next(iter(cam.frames()), None)
        c.ok(first is not None and first.ndim == 3,
             f"and still decodes with the options installed "
             f"({None if first is None else first.shape})")
    else:
        c.ok(True, "register_file.avi absent here -- decode check skipped")

    print("\n6. RECONNECT USES THE SAME SETTINGS AS THE FIRST OPEN")
    # Both paths now go through _new_capture(); before, each built its own
    # VideoCapture and only open() applied the stream settings.
    import inspect
    body = inspect.getsource(VideoSource._reopen)
    c.ok("_new_capture" in body,
         "_reopen() builds its capture through the shared helper")
    c.ok("cv2.VideoCapture" not in body,
         "and does NOT construct one of its own (that is how settings drifted)")
    c.eq(inspect.getsource(VideoSource.open).count("cv2.VideoCapture"), 0,
         "open() likewise")

    print("\n7. STREAM CAPTURES CARRY OPENCV'S OWN TIMEOUTS TOO")
    c.ok(hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC")
         and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"),
         "this OpenCV exposes the open/read timeout properties")
    import video_source
    apply_rtsp_options(transport="tcp", open_timeout_ms=1234,
                       read_timeout_ms=4321, log=QUIET)
    params = video_source._STREAM_OPEN_PARAMS
    c.ok(1234 in params and 4321 in params,
         f"both are passed per stream capture ({params})")

    c.done()


if __name__ == "__main__":
    main()
