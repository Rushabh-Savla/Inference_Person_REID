r"""
============================================================================
INTERRUPT GUARD -- protect the finalization phase from a second Ctrl-C
============================================================================

Both entry paths (file-batch `main.py` and the live pipeline) do their most
important work AFTER the user stops the run:

    Ctrl-C  ->  stop the sources  ->  offline reconcile  ->  re-render
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                      this is where the CORRECT cross-camera
                                      ids are decided

The first Ctrl-C is the documented way to stop a live session, and it is
handled. The problem is the SECOND one: Python raises KeyboardInterrupt again,
this time in the middle of reconcile or re-render, so the run ends with
provisional (per-camera) ids and half-written videos -- exactly the result the
whole pipeline exists to avoid. Users press it because finalization is silent
for a while and looks like a hang.

So during finalization we make Ctrl-C non-fatal: it prints what is still
running and keeps going.

ESCAPE HATCH. `allow_force_after` presses still let the process die, but do NOT
rely on it as *the* way out: standard signals do not queue, so several Ctrl-C
presses in quick succession collapse into ONE delivery (measured). The reliable
abort is **Ctrl-\ (SIGQUIT)**, which this guard never blocks -- that is what the
messages tell the user, and it is why `allow_force_after` is a courtesy rather
than a contract.

WHY TWO MECHANISMS. A Python-level handler alone is not enough in this process.
The live pipeline runs OpenCV capture/writer and torch inference on many
threads, and library worker threads routinely block SIGINT. On Linux the kernel
delivers the signal to *some* thread that has it unblocked; when it lands on a
thread that has it blocked, the signal just sits there pending -- Python's
handler never runs, and the still-pending signal kills the process during
interpreter teardown (exit code -2) even though the run had completed. Measured
in exactly that way on the live path. So:

  1. `pthread_sigmask` blocks SIGINT for the main thread, and a small watcher
     thread calls `sigwait` to *consume* it. Consuming is the important part:
     nothing is left pending to kill us later, and no KeyboardInterrupt can be
     raised into the main thread's Python code.
  2. A Python-level SIGINT handler stays installed as well, for the case where
     the signal is delivered through Python's normal path (some other thread
     with SIGINT unblocked sets Python's flag).

Whichever route the kernel picks, the user gets a message and the finalization
completes.

Usage (main thread only -- that is where Python delivers signals):

    with InterruptGuard("reconcile + re-render"):
        reconcile_tracklets(...)
        render_final_videos(...)

Fail-open by design: on a platform without `pthread_sigmask`/`sigwait`
(Windows) only mechanism 2 is used, and if even that can't be installed (not
the main thread) the block still runs, just unprotected.
============================================================================
"""

import os
import signal
import sys
import threading


class InterruptGuard:
    """Context manager that makes Ctrl-C non-fatal for the wrapped block."""

    def __init__(self, what="finalizing outputs", allow_force_after=3):
        """
        what              : short description printed back to the user, e.g.
                            "reconcile + re-render".
        allow_force_after : how many Ctrl-C presses before the process is
                            allowed to die, so an insistent user can get out.
        """
        self.what = what
        self.allow_force_after = max(1, int(allow_force_after))
        self.presses = 0
        self._lock = threading.Lock()
        self._previous_handler = None
        self._handler_installed = False
        self._prev_mask = None
        self._mask_installed = False
        self._watcher = None
        self._done = threading.Event()

    # -- context manager ---------------------------------------------------
    def __enter__(self):
        # (1) block + sigwait: deterministic, independent of which thread the
        #     kernel picks and of the interpreter reaching a bytecode boundary.
        if hasattr(signal, "pthread_sigmask") and hasattr(signal, "sigtimedwait"):
            try:
                self._prev_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGINT})
                self._mask_installed = True
                self._watcher = threading.Thread(
                    target=self._watch, name="interrupt-guard", daemon=True)
                self._watcher.start()
            except (ValueError, OSError, AttributeError):
                self._mask_installed = False

        # (2) Python-level handler, for signals that arrive via Python's own
        #     flag mechanism (delivered to a thread that has SIGINT unblocked).
        try:
            self._previous_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._on_sigint)
            self._handler_installed = True
        except (ValueError, OSError, AttributeError):
            self._handler_installed = False
        return self

    def __exit__(self, exc_type, exc, tb):
        self._done.set()
        if self._watcher is not None:
            self._watcher.join(timeout=1.0)
            self._watcher = None
        if self._handler_installed:
            try:
                signal.signal(signal.SIGINT, self._previous_handler)
            except (ValueError, OSError, AttributeError):
                pass
            self._handler_installed = False
        if self._mask_installed:
            # Drain anything that arrived in the last instant, so a leftover
            # pending SIGINT cannot kill the process during teardown, then put
            # the old mask back.
            try:
                while signal.sigtimedwait({signal.SIGINT}, 0) is not None:
                    pass
            except (OSError, ValueError, AttributeError, InterruptedError):
                pass
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, self._prev_mask)
            except (ValueError, OSError, AttributeError):
                pass
            self._mask_installed = False
        return False        # never swallow exceptions from the block

    # -- the two delivery routes -------------------------------------------
    def _watch(self):
        """Consume blocked SIGINTs until the guarded block finishes."""
        # Inherit the block so sigwait -- not a handler -- receives the signal.
        try:
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        except (ValueError, OSError, AttributeError):
            return
        while not self._done.is_set():
            try:
                got = signal.sigtimedwait({signal.SIGINT}, 0.25)
            except InterruptedError:
                continue
            except (OSError, ValueError):
                return
            if got is not None:
                self._register_press()

    def _on_sigint(self, signum, frame):
        self._register_press()

    # -- shared press bookkeeping ------------------------------------------
    def _register_press(self):
        with self._lock:
            self.presses += 1
            left = self.allow_force_after - self.presses
        if left > 0:
            print(
                f"\n[guard] Ctrl-C ignored -- still {self.what}. Interrupting "
                f"now would throw away the reconciled cross-camera ids and "
                f"leave output_<cam>.mp4 unfinished. Please wait. "
                f"(If you really must abort, press Ctrl-\\ -- SIGQUIT, which "
                f"this guard never blocks.)",
                file=sys.stderr, flush=True,
            )
            return
        print(
            f"\n[guard] Force-quit requested -- abandoning {self.what}. The "
            f"videos will keep the provisional per-camera ids.",
            file=sys.stderr, flush=True,
        )
        self._force_quit()

    def _force_quit(self):
        """Escape hatch: really stop, from whichever thread saw the press."""
        # Undo everything we installed, then re-send the signal to ourselves so
        # the process dies the way an unguarded Ctrl-C would.
        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        except (ValueError, OSError, AttributeError):
            pass
        if self._mask_installed:
            try:
                signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
            except (ValueError, OSError, AttributeError):
                pass
        self._done.set()
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except OSError:
            os._exit(130)


def print_stop_hint(mode="live"):
    """One-line banner, printed at startup, that tells the user how to stop the
    run and why pressing Ctrl-C twice used to be destructive."""
    if mode == "live":
        print("[stop] To stop: press Ctrl-C ONCE. The run then reconciles "
              "cross-camera ids and re-renders output_<cam>.mp4 -- that step "
              "can take a while and prints as it goes. Extra Ctrl-C presses "
              "during it are IGNORED on purpose, so they can no longer cost "
              "you the reconciled ids (use Ctrl-\\ if you must abort anyway).")
    else:
        print("[stop] To stop early: press Ctrl-C ONCE (or 'q' in a window). "
              "Reconcile + re-render still runs afterwards; extra Ctrl-C "
              "presses during that step are IGNORED on purpose (use Ctrl-\\ "
              "if you must abort anyway and lose the reconciled ids).")
