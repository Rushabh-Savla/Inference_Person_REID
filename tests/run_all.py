"""
run_all.py  --  run EVERY logic test under tests/, not just the Stage-3 gate.

    python tests/run_all.py

Why this exists: `tests/live/run_stage3_acceptance.py` is deliberately scoped to
the Stage-3 acceptance criteria, so its SUITE lists only `test_stage3_s*.py`. The
Stage-2 and Stage-6 tests were therefore not run by anything -- they had to be
remembered and invoked by hand. That is how a reconcile defect shipped with a
passing test suite: nothing was running the file that would have caught it.

This discovers `tests/**/test_*.py`, runs each as its own process (they are plain
scripts using the `_synth.Check` harness, not pytest -- pytest is not a dependency),
and exits non-zero if any fails. All tests are deterministic and synthetic: no GPU,
no video, no Qdrant, no threads beyond what a test starts itself, so this is the
fast local gate to run before committing.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def discover():
    """tests/**/test_*.py, sorted for stable output."""
    found = []
    for root, _dirs, files in os.walk(HERE):
        for name in sorted(files):
            if name.startswith("test_") and name.endswith(".py"):
                found.append(os.path.join(root, name))
    return sorted(found)


def main():
    tests = discover()
    if not tests:
        print("run_all: no tests discovered -- that is itself a failure")
        return 1

    print("=" * 72)
    print(f"RUN ALL  --  {len(tests)} test file(s)")
    print("=" * 72)

    failed = []
    for path in tests:
        rel = os.path.relpath(path, HERE)
        print(f"\n>>> {rel}")
        # cwd = the test's own directory so `from _synth import ...` resolves the
        # same way it does when a test is run directly.
        proc = subprocess.run([sys.executable, os.path.basename(path)],
                              cwd=os.path.dirname(path))
        if proc.returncode != 0:
            failed.append(rel)

    print("\n" + "=" * 72)
    if failed:
        print(f"RESULT: FAIL  ({len(failed)}/{len(tests)} test file(s) failed)")
        for f in failed:
            print(f"  - {f}")
        return 1
    print(f"RESULT: PASS  (all {len(tests)} test file(s) passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
