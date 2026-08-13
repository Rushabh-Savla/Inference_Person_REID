"""
The geometric reachability check -- the unit that decides "one person could not
have been in both places".

Everything here is synthetic and deterministic: no GPU, no footage, no Qdrant, no
calibration file. The floor is a plain 2-D plane and positions are handed in
directly, which is exactly how the runtime consumes them (offline reconcile reads
recorded positions, it never derives any -- geometry/__init__.py invariant 1).

WHAT THESE CHECKS DEFEND. This project already built one hard geometric veto --
`live.topology`'s min-transit rule -- and had to disable it after it pruned the TRUE
cross-camera match (links 5 -> 1, topology_pruned=508). So the checks below are
weighted toward the ways a veto goes WRONG, not the ways it fires:

  * every unavailability path returns "no opinion", never "far apart";
  * both error budgets make the veto MORE permissive, never less;
  * the ceiling is measured from data, and refuses to be a guess;
  * metres are never asserted from an unscaled frame.
"""

import sys

from _synth import Check

from geometry.calibration import CalibrationRecord, MetricScaleUnavailable
from geometry.reachability import (IMPOSSIBLE, PLAUSIBLE, UNAVAILABLE,
                                   RecordedPosition, SpeedEnvelope,
                                   observed_speed_ceiling)


def pos(x, y, ts, error=0.0, group="room"):
    return RecordedPosition(x, y, error, group, ts)


def main():
    c = Check("geometry: the reachability envelope")

    # A ceiling of 2.0 units/s with safety 1.0 -> veto above 2.0 units/s. Clock
    # budget 0 for the arithmetic checks, so the numbers are exact.
    env = SpeedEnvelope(2.0, clock_error_sec=0.0, safety_factor=1.0, group="room")

    # ------------------------------------------------------------------ 1
    print("\n1. THE ARITHMETIC -- distance over elapsed, against the ceiling")

    r = env.check(pos(0, 0, 100.0), pos(1.0, 0, 101.0))
    c.eq(r.verdict, PLAUSIBLE, "1 unit in 1 s (1.0 units/s) is within a 2.0 ceiling")

    r = env.check(pos(0, 0, 100.0), pos(10.0, 0, 101.0))
    c.eq(r.verdict, IMPOSSIBLE, "10 units in 1 s (10.0 units/s) is not")
    c.ok(r.required_speed == 10.0 and r.limit == 2.0,
         f"and the reason carries the numbers ({r.required_speed} vs {r.limit})")

    # Same instant, far apart: required speed diverges, so the "one body cannot be
    # in two places" negative falls out of the SAME formula rather than needing a
    # second rule. This is the case the overlapping cameras actually produce.
    r = env.check(pos(0, 0, 100.0), pos(5.0, 0, 100.0))
    c.eq(r.verdict, IMPOSSIBLE,
         "5 units apart at the SAME instant is impossible -- no separate rule needed")

    # Shoulder-to-shoulder at the same instant must NOT be vetoed -- and note what
    # protects it: the position error radius, not the speed ceiling. With a
    # zero-error position, 0.3 units apart at one instant genuinely is two bodies.
    # In production that radius is the calibration's own held-out residual, so the
    # protection is measured rather than assumed.
    r = env.check(pos(0, 0, 100.0, error=0.25), pos(0.3, 0, 100.0, error=0.25))
    c.eq(r.verdict, PLAUSIBLE,
         "shoulder-to-shoulder at the same instant survives, via the position "
         "error radius")
    r = env.check(pos(0, 0, 100.0, error=0.25), pos(5.0, 0, 100.0, error=0.25))
    c.eq(r.verdict, IMPOSSIBLE,
         "while 5 units apart at that instant is still impossible -- the radius "
         "protects the ambiguous case without disarming the check")

    # Order must not matter: a veto that depended on argument order would apply to
    # some cluster pairs and not others, depending on iteration order.
    a, b = pos(0, 0, 100.0), pos(9.0, 0, 101.0)
    c.eq(env.check(a, b).verdict, env.check(b, a).verdict, "the check is symmetric")

    # ------------------------------------------------------------------ 2
    print("\n2. EVERY ERROR BUDGET MAKES THE VETO MORE PERMISSIVE")

    # 3 units in 1 s = 3.0 units/s, over the 2.0 ceiling -> impossible with exact
    # positions. Give each position a 0.6-unit error radius and the effective
    # distance drops to 1.8, so it is no longer impossible. That direction is the
    # whole safety argument: a sloppier calibration vetoes LESS.
    exact = env.check(pos(0, 0, 100.0), pos(3.0, 0, 101.0))
    fuzzy = env.check(pos(0, 0, 100.0, error=0.6), pos(3.0, 0, 101.0, error=0.6))
    c.eq(exact.verdict, IMPOSSIBLE, "3 units in 1 s is impossible with exact positions")
    c.eq(fuzzy.verdict, PLAUSIBLE,
         "the same pair with 0.6-unit position error is NOT vetoed (fails open)")

    # The clock budget does the same thing on the time axis.
    slack = SpeedEnvelope(2.0, clock_error_sec=1.0, safety_factor=1.0)
    c.eq(slack.check(pos(0, 0, 100.0), pos(3.0, 0, 101.0)).verdict, PLAUSIBLE,
         "a 1 s clock budget also turns that veto off (elapsed is GROWN)")

    # And so does the safety factor.
    safe = SpeedEnvelope(2.0, clock_error_sec=0.0, safety_factor=2.0)
    c.eq(safe.check(pos(0, 0, 100.0), pos(3.0, 0, 101.0)).verdict, PLAUSIBLE,
         "safety_factor 2.0 raises the veto line to 4.0 units/s")
    c.ok(safe.limit == 4.0, f"limit is ceiling x safety ({safe.limit})")

    # ------------------------------------------------------------------ 3
    print("\n3. UNAVAILABLE IS NOT 'FAR APART' -- every uncertainty fails open")

    c.eq(env.check(None, pos(0, 0, 100.0)).verdict, UNAVAILABLE,
         "a missing position gives no opinion")
    c.eq(env.check(pos(0, 0, None), pos(9.0, 0, 100.0)).verdict, UNAVAILABLE,
         "a missing timestamp gives no opinion (frame indices are not comparable)")
    c.eq(env.check(pos(0, 0, 100.0, group="room_a"),
                   pos(99.0, 0, 100.0, group="room_b")).verdict, UNAVAILABLE,
         "two DIFFERENT floor groups give no opinion -- their coordinates are not "
         "comparable, which is not the same as being far apart")
    c.eq(env.check(pos(0, 0, 100.0, group=None),
                   pos(9.0, 0, 100.0, group=None)).verdict, UNAVAILABLE,
         "a position with no group gives no opinion")

    # An envelope cannot be built from a guess -- the ceiling must be measured.
    for bad in (None, 0.0, -1.0):
        try:
            SpeedEnvelope(bad)
            c.ok(False, f"SpeedEnvelope({bad!r}) should have been refused")
        except ValueError:
            c.ok(True, f"SpeedEnvelope({bad!r}) is refused -- no guessed ceilings")

    # ------------------------------------------------------------------ 4
    print("\n4. THE CEILING IS MEASURED FROM SAME-TRACK MOTION")

    # One track walking a straight line at exactly 1.5 units/s, sampled every 0.5 s.
    # Same track = provably one person, so this needs no labels and no metric scale.
    walk = {("cam_a", 1): [(0.5 * i, 0.75 * i, 0.0) for i in range(60)]}
    stats = observed_speed_ceiling(walk)
    c.ok(stats is not None, "60 observations of one track are enough to measure")
    c.ok(abs(stats["median"] - 1.5) < 1e-6,
         f"median observed speed is the true 1.5 units/s (got {stats['median']:.4f})")
    c.ok(stats["ceiling_units_per_sec"] >= stats["p99"] - 1e-9,
         "the ceiling is a high percentile, not the median")

    # Too little data must produce None, which leaves the check unavailable rather
    # than inventing a ceiling from three points.
    c.ok(observed_speed_ceiling({("cam_a", 1): [(0.0, 0, 0), (0.5, 1, 0)]}) is None,
         "two observations are NOT enough -- returns None, so the check stays "
         "unavailable")

    # Pairs outside the dt window are excluded from the FIT: a 20 ms gap divides
    # position noise by a tiny dt and manufactures an enormous speed.
    jitter = {("cam_a", 1): [(i * 0.01, (i % 2) * 0.5, 0.0) for i in range(200)]}
    c.ok(observed_speed_ceiling(jitter) is None,
         "observations 10 ms apart are all rejected as too close to time -- no "
         "phantom ceiling from noise/dt")

    # ------------------------------------------------------------------ 5
    print("\n5. METRES ARE NEVER ASSERTED FROM AN UNSCALED FRAME")

    unscaled = CalibrationRecord({
        "calib_version": "v1",
        "groups": {"room": {"cameras": {"cam_a": {"H": [[1, 0, 0], [0, 1, 0],
                                                        [0, 0, 1]],
                                                  "image_size": [2560, 1440]}}}},
    })
    c.ok(not unscaled.is_metric, "a frame with no metric reference is NOT metric")
    c.ok("floor units" in unscaled.describe_units()
         and "metres" not in unscaled.describe_units().replace("metres unavailable", ""),
         f"and says so plainly: {unscaled.describe_units()!r}")
    try:
        unscaled.to_metres(1.0)
        c.ok(False, "to_metres() should have raised")
    except MetricScaleUnavailable as e:
        c.ok("trusted metric reference" in str(e),
             "to_metres() raises and the message says what to supply")

    scaled = CalibrationRecord({
        "calib_version": "v2",
        "metric_reference": {"source": "floor_plan", "description": "corridor",
                             "floor_units": 4.0, "metres": 2.0},
        "groups": {"room": {"cameras": {"cam_a": {"H": [[1, 0, 0], [0, 1, 0],
                                                        [0, 0, 1]],
                                                  "image_size": [2560, 1440]}}}},
    })
    c.ok(scaled.is_metric and scaled.metres_per_unit == 0.5,
         f"one reference gives a scale ({scaled.metres_per_unit} m per unit)")

    # Two references that disagree are a contradiction, not something to average.
    try:
        CalibrationRecord({
            "metric_reference": [
                {"source": "plan", "description": "a", "floor_units": 4.0,
                 "metres": 2.0},
                {"source": "tape", "description": "b", "floor_units": 4.0,
                 "metres": 3.0},
            ],
            "groups": {},
        })
        c.ok(False, "disagreeing metric references should have been refused")
    except ValueError as e:
        c.ok("disagree" in str(e),
             "two references that disagree by 40% are REFUSED, not averaged")

    # A half-written reference must not degrade to "no scale" silently.
    try:
        CalibrationRecord({"metric_reference": {"source": "plan", "metres": 2.0},
                           "groups": {}})
        c.ok(False, "an incomplete metric reference should have been refused")
    except ValueError as e:
        c.ok("missing" in str(e),
             "an incomplete reference is refused rather than ignored")

    c.done()


if __name__ == "__main__":
    sys.exit(main() or 0)
