"""
Regression: finalization must reach the reconcile even when reporting/printing fails.

WHY THIS EXISTS. Two real 4-camera RTSP runs (2026-07-30, run_ids 20260730_081503
and 20260730_082045) produced clips and metrics and then nothing -- no final
summary, no reconcile, no decision log, no output video. Both were launched as

    python main.py ... 2>&1 | tee run1.log

which puts python and tee in ONE foreground process group. Ctrl-C goes to both,
tee dies first, and every subsequent print in python raises BrokenPipeError. The
first such print lands inside the shutdown sequence, and _report(final=True) runs
BEFORE _finalize_offline() -- so a cosmetic print failure abandoned the run's ids,
with the traceback going into the same dead pipe so nothing was visible.

A dropped SSH session or a closed terminal breaks stdout identically. InterruptGuard
does not help: it protects against SIGNALS, and explicitly does not swallow
exceptions.

These tests pin both halves of the fix:
  * _report is guarded, so a failure there cannot skip what follows it
  * stdout/stderr are wrapped for the finalization phase, so no print can raise
"""

import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from live.pipeline import LivePipeline, _QuietOnBrokenPipe

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


class Exploding(io.TextIOBase):
    """A stream that fails like a pipe whose reader has exited."""

    def __init__(self):
        self.writes = 0

    def write(self, data):
        self.writes += 1
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


def make_pipeline():
    """Minimal LivePipeline: no threads, no sources, reconcile 'enabled'."""
    p = LivePipeline([], {})
    p._t_start = 1.0            # non-None so _report is attempted
    p.reconcile_enabled = True
    p.captures, p.threads, p.writers, p.renderers = [], [], [], []
    return p


# ---------------------------------------------------------------------------
print("\n1. _QuietOnBrokenPipe absorbs a dead consumer")

boom = Exploding()
q = _QuietOnBrokenPipe(boom)
try:
    n = q.write("hello")
    q.write("again")
    q.flush()
    raised = False
except Exception as e:                                          # noqa: BLE001
    raised = repr(e)
check("write/flush never raise", raised is False, raised or "clean")
check("marked broken after the first failure", q.broken)
check("stops re-trying the dead stream", boom.writes == 1,
      f"underlying stream written to {boom.writes}x (want 1)")
check("write reports the full length so callers do not loop", n == len("hello"),
      f"returned {n}")

good = io.StringIO()
q2 = _QuietOnBrokenPipe(good)
q2.write("kept")
q2.flush()
check("a healthy stream is passed through untouched",
      good.getvalue() == "kept" and not q2.broken, repr(good.getvalue()))


# ---------------------------------------------------------------------------
print("\n2. A failing _report does NOT skip the reconcile")

p = make_pipeline()
called = {"finalize": False, "report": False}


def exploding_report(*_a, **_kw):
    called["report"] = True
    raise RuntimeError("metrics blew up")


p._report = exploding_report
p._finalize_offline = lambda: called.__setitem__("finalize", True)
p._shutdown_inner()

check("_report was attempted", called["report"])
check("_finalize_offline STILL ran", called["finalize"],
      "this is the whole point -- the ids matter, the report does not")


# ---------------------------------------------------------------------------
print("\n3. A broken stdout does NOT skip the reconcile (the real-world case)")

p = make_pipeline()
called2 = {"finalize": False}


def printing_report(*_a, **_kw):
    # Exactly what the real _report does: writes to stdout. With a dead pipe and
    # no wrapper this raised BrokenPipeError and killed the run.
    print("[live:metrics] final SUMMARY ...")


p._report = printing_report
p._finalize_offline = lambda: called2.__setitem__("finalize", True)

prev_out, prev_err = sys.stdout, sys.stderr
sys.stdout, sys.stderr = Exploding(), Exploding()
try:
    p._shutdown()          # installs the wrapper internally
    crashed = None
except Exception as e:                                          # noqa: BLE001
    crashed = repr(e)
finally:
    sys.stdout, sys.stderr = prev_out, prev_err

check("_shutdown survives a dead stdout", crashed is None, crashed or "clean")
check("_finalize_offline ran with stdout broken", called2["finalize"])
check("streams are restored afterwards",
      sys.stdout is prev_out and sys.stderr is prev_err)


# ---------------------------------------------------------------------------
print("\n4. Both failures at once (report raises AND stdout is dead)")

p = make_pipeline()
called3 = {"finalize": False}
p._report = exploding_report
p._finalize_offline = lambda: called3.__setitem__("finalize", True)

prev_out, prev_err = sys.stdout, sys.stderr
sys.stdout, sys.stderr = Exploding(), Exploding()
try:
    p._shutdown()
    crashed2 = None
except Exception as e:                                          # noqa: BLE001
    crashed2 = repr(e)
finally:
    sys.stdout, sys.stderr = prev_out, prev_err

check("_shutdown survives both", crashed2 is None, crashed2 or "clean")
check("_finalize_offline still ran", called3["finalize"])


# ---------------------------------------------------------------------------
print("\n5. Ordering invariant: reconcile is not gated on reporting")

import inspect
src = inspect.getsource(LivePipeline._shutdown_inner)
i_report = src.find("self._report(")
i_final = src.find("self._finalize_offline()")
check("_report still precedes _finalize_offline in the source",
      -1 < i_report < i_final, f"report@{i_report} finalize@{i_final}")
check("_report is wrapped in a try", "try:" in src[:i_report].rsplit("\n\n", 1)[-1],
      "guarded so its failure cannot abandon the ids")


failed = [n for n, ok in CHECKS if not ok]
print(f"\nShutdown reaches reconcile: {len(CHECKS) - len(failed)}/{len(CHECKS)} passed")
if failed:
    print("FAILED: " + "; ".join(failed))
    raise SystemExit(1)
print("OK")
