"""
reachability.py -- could ONE person have been at both of these places, at those
two times?

This is the whole geometric contribution to identity. One formula:

    required_speed = distance / elapsed_time

and one comparison against a speed ceiling. If a proposed merge would require a
person to move faster than any person on this floor has ever been observed to
move, they are two people, and no cosine similarity can excuse it.

The formula covers BOTH cases the project cares about, which is why there is only
one of it:

    elapsed ~= 0   two detections at the same instant, metres apart -> required
                   speed is enormous -> IMPOSSIBLE. This is the near-absolute
                   "one person cannot be in two places" negative.
    elapsed  > 0   seen in one camera, then another. Distance over elapsed time is
                   the walking speed the merge implies. Too fast -> IMPOSSIBLE.
                   This is "they could not have got there in time".

WHY THIS IS NOT THE TOPOLOGY VETO AGAIN. `src/live/topology.py` did the second
case with HAND-SET minimum transit times per camera pair, and it was disabled after
it pruned the true cross-camera match (links collapsed 5 -> 1,
topology_pruned=508). It failed for a specific, fixable reason: these cameras are
adjacent and overlapping, so the real minimum transit time between them is
essentially ZERO, while the config claimed 2-3 seconds. The number was wrong, not
the idea. Measured floor positions cannot make that mistake -- two overlapping
cameras produce a distance near zero, so the check stays silent exactly where the
hand-set veto over-fired.

===================== EVERY ERROR BIASES TOWARD "POSSIBLE" =====================

A veto is UNRECOVERABLE. Reconcile cannot later un-split a person whose merge it
refused, and that is precisely how the topology veto destroyed a run. A MISSED veto
costs nothing new -- it leaves the false merge that already happens today. So the
arithmetic is deliberately lopsided:

    distance is SHRUNK   by both positions' error radii   (closer  -> slower -> ok)
    elapsed  is GROWN    by the clock-differential budget (longer  -> slower -> ok)
    ceiling  is RAISED   by a safety factor               (faster allowed -> ok)

A sloppier calibration therefore produces FEWER vetoes, never wrong ones. That is
the only safe way for this check to degrade.

======================== THE CEILING IS MEASURED, NOT GUESSED ==================

`observed_speed_ceiling()` reads the speed ceiling off the footage: consecutive
observations of ONE track are provably one person, so their speed distribution is
this floor's real answer, in whatever unit the floor frame happens to use. That
matters twice over:

  * it needs no metric scale -- distance and ceiling share the unknown unit, so it
    cancels exactly, and no metre-based threshold is ever asserted;
  * it is not a hypothesis. Every numeric constant this project guessed at has
    been reverted for hurting accuracy (four threshold changes and the topology
    veto). A ceiling read off the data cannot repeat that.

IMPORTS NOTHING FROM THIS PACKAGE, ON PURPOSE. Offline reconcile is allowed to
import this module and nothing else under `geometry/`, which is what mechanically
prevents it from recomputing geometry (see geometry/__init__.py invariant 1).
"""

from __future__ import annotations

import math

# Verdicts. Strings rather than an enum so they land in a JSONL decision log and a
# Qdrant payload unchanged, and read correctly in a log a human is scanning.
IMPOSSIBLE = "impossible"        # a hard veto: no person could do this
PLAUSIBLE = "plausible"          # geometry has no objection -- NOT evidence FOR a merge
UNAVAILABLE = "unavailable"      # no opinion; consumers must treat as neutral

# Default cross-camera timestamp uncertainty, in seconds. `ts` is when a frame was
# DECODED locally, not when photons hit the sensor, so two cameras' stamps for one
# instant differ by their encoder + network + decoder buffering. 0.5 s is
# deliberately generous: it is the ADR-003A ceiling rather than its 150 ms target,
# because over-stating it only makes the check more permissive.
DEFAULT_CLOCK_ERROR_SEC = 0.5

# Multiplies the measured ceiling. 1.5 leaves room for a person moving faster than
# anything in the calibration footage -- a run, a stumble -- without opening the
# envelope so wide that "same instant, across the room" stops registering.
DEFAULT_SAFETY_FACTOR = 1.5

# Bounds on the observation pairs used to MEASURE the ceiling.
#   below: two observations 20 ms apart are one frame apart, where position noise
#          divided by a tiny dt manufactures an enormous speed out of nothing;
#   above: a 3-second gap inside one track means the person was occluded and the
#          straight-line distance understates the path, so the speed is not a
#          speed. Neither is dropped from the pipeline -- only from the FIT.
MIN_FIT_DT_SEC = 0.10
MAX_FIT_DT_SEC = 2.00


class RecordedPosition:
    """A floor position as it was RECORDED during the run, plus its timestamp.

    Constructed from what the live run persisted -- never from a bbox. There is
    deliberately no code path here from a box to a position; that lives in
    geometry/floor.py, which reconcile does not import.
    """

    __slots__ = ("x", "y", "error", "group", "ts")

    def __init__(self, x, y, error, group, ts):
        self.x = float(x)
        self.y = float(y)
        self.error = float(error)
        self.group = group
        self.ts = None if ts is None else float(ts)

    @classmethod
    def from_payload(cls, payload):
        """Read the `floor` block out of a stored observation payload. None if absent.

        Tolerant by design: an observation stored before geometry existed, or by a
        run with geometry disabled, has no `floor` key and must produce None rather
        than an exception -- the store holds runs from both eras.
        """
        if not isinstance(payload, dict):
            return None
        floor = payload.get("floor")
        if not isinstance(floor, dict):
            return None
        x, y = floor.get("x"), floor.get("y")
        group = floor.get("group")
        if x is None or y is None or group is None:
            return None
        ts = payload.get("ts")
        try:
            return cls(x, y, floor.get("error", 0.0), group, ts)
        except (TypeError, ValueError):
            return None

    def __repr__(self):
        return (f"RecordedPosition({self.group} ({self.x:.2f}, {self.y:.2f}) "
                f"+-{self.error:.2f} @ {self.ts})")


class Reason:
    """Why the envelope said what it said -- for the decision log and for humans.

    Every rejected merge in this project used to leave no trace of WHICH rule
    refused, which is the documented reason four rounds of threshold tuning taught
    nobody anything. A verdict that cannot be explained afterwards is not usable
    evidence, so the reason travels with it.
    """

    __slots__ = ("verdict", "detail", "distance", "distance_effective",
                 "elapsed", "elapsed_effective", "required_speed", "limit")

    def __init__(self, verdict, detail, distance=None, distance_effective=None,
                 elapsed=None, elapsed_effective=None, required_speed=None,
                 limit=None):
        self.verdict = verdict
        self.detail = detail
        self.distance = distance
        self.distance_effective = distance_effective
        self.elapsed = elapsed
        self.elapsed_effective = elapsed_effective
        self.required_speed = required_speed
        self.limit = limit

    @property
    def impossible(self):
        return self.verdict == IMPOSSIBLE

    def as_dict(self):
        out = {"verdict": self.verdict, "detail": self.detail}
        for name in ("distance", "distance_effective", "elapsed",
                     "elapsed_effective", "required_speed", "limit"):
            v = getattr(self, name)
            if v is not None:
                out[name] = round(float(v), 4)
        return out

    def __repr__(self):
        if self.required_speed is None:
            return f"Reason({self.verdict}: {self.detail})"
        return (f"Reason({self.verdict}: {self.detail}; "
                f"{self.required_speed:.2f} vs limit {self.limit:.2f} units/s)")


class SpeedEnvelope:
    """The impossibility envelope: faster than this, and it is not one person.

    `max_speed` is in FLOOR UNITS per second, from `observed_speed_ceiling()` on
    this deployment's own footage. Passing a number invented from human-walking-speed
    tables would reintroduce exactly the guessed constant this class exists to
    avoid, so `from_calibration()` refuses when the record has no measurement.
    """

    def __init__(self, max_speed, clock_error_sec=DEFAULT_CLOCK_ERROR_SEC,
                 safety_factor=DEFAULT_SAFETY_FACTOR, group=None):
        if max_speed is None or not (max_speed > 0):
            raise ValueError(
                "[geometry] SpeedEnvelope needs a positive measured speed ceiling. "
                "Measure it with observed_speed_ceiling() -- do not supply a guess.")
        self.max_speed = float(max_speed)
        self.clock_error_sec = max(0.0, float(clock_error_sec))
        self.safety_factor = float(safety_factor)
        self.group = group

    @property
    def limit(self):
        """The speed above which a merge is declared impossible."""
        return self.max_speed * self.safety_factor

    @classmethod
    def for_group(cls, record, group, clock_error_sec=DEFAULT_CLOCK_ERROR_SEC,
                  safety_factor=DEFAULT_SAFETY_FACTOR):
        """Build the envelope for one floor group, or None if it is unmeasured.

        None means the reachability check is UNAVAILABLE for that group -- which
        fails open. That is the correct behaviour for a group whose footage has not
        yet been measured, and it is why this returns None instead of defaulting.
        """
        if record is None:
            return None
        entry = record.group_entry(group) if hasattr(record, "group_entry") else None
        if not entry:
            return None
        ceiling = entry.get("speed_ceiling_units_per_sec")
        if ceiling is None or not (float(ceiling) > 0):
            return None
        return cls(float(ceiling), clock_error_sec=clock_error_sec,
                   safety_factor=safety_factor, group=group)

    # ---- the check ---------------------------------------------------------

    def check(self, a, b):
        """-> Reason. The only decision this package makes.

        `a` and `b` are RecordedPosition. Order does not matter.
        """
        if a is None or b is None:
            return Reason(UNAVAILABLE, "one or both observations have no recorded "
                                       "floor position")
        if a.group is None or b.group is None:
            return Reason(UNAVAILABLE, "an observation has no floor group")
        if a.group != b.group:
            # Non-overlapping cameras share no floor points, so no homography can
            # relate them and their coordinates are not comparable. This is NOT
            # "far apart" -- it is unknown. Closing this gap needs a real-world
            # distance between the two groups AND a metric reference to convert it;
            # see CalibrationRecord.group_distance_metres.
            return Reason(UNAVAILABLE,
                          f"cameras are in different floor groups "
                          f"({a.group} vs {b.group}) -- no shared coordinates")
        if a.ts is None or b.ts is None:
            # Frame indices are per-camera and not comparable (15 fps vs 25 fps put
            # frame 1000 forty seconds apart), so without wall-clock ts there is no
            # elapsed time and the question cannot be asked.
            return Reason(UNAVAILABLE, "an observation has no wall-clock timestamp")

        distance = math.hypot(a.x - b.x, a.y - b.y)
        # Shrink the distance by what the positions could be wrong by. Both radii,
        # because either observation could be displaced toward the other.
        distance_eff = max(0.0, distance - a.error - b.error)

        elapsed = abs(a.ts - b.ts)
        # Grow the elapsed time by the clock budget: if the two cameras' stamps for
        # one instant can differ by this much, the person may have had that much
        # longer than the timestamps suggest.
        elapsed_eff = elapsed + self.clock_error_sec

        if elapsed_eff <= 0:
            # Genuinely simultaneous, with no clock slack to spend. Any separation
            # beyond the position error then requires infinite speed, which is the
            # continuous limit of the division below rather than a special case --
            # and it is the STRONGEST verdict this check can reach, so returning
            # "no opinion" here would throw away exactly the evidence the
            # overlapping cameras exist to provide.
            if distance_eff > 0:
                return Reason(
                    IMPOSSIBLE,
                    f"{distance:.2f} units apart at the same instant "
                    f"({distance_eff:.2f} after position error) -- one body cannot "
                    f"be in two places",
                    distance, distance_eff, elapsed, elapsed_eff, float("inf"),
                    self.limit)
            return Reason(
                PLAUSIBLE,
                "same instant, and within the positions' error radii -- people do "
                "stand shoulder to shoulder",
                distance, distance_eff, elapsed, elapsed_eff, 0.0, self.limit)

        required = distance_eff / elapsed_eff
        limit = self.limit
        if required > limit:
            return Reason(
                IMPOSSIBLE,
                f"would require {required:.2f} units/s over {distance:.2f} units in "
                f"{elapsed:.2f} s; nothing on this floor has exceeded "
                f"{limit:.2f} units/s",
                distance, distance_eff, elapsed, elapsed_eff, required, limit)
        return Reason(
            PLAUSIBLE,
            f"{required:.2f} units/s is within the {limit:.2f} units/s envelope",
            distance, distance_eff, elapsed, elapsed_eff, required, limit)

    def describe(self):
        return (f"speed ceiling {self.max_speed:.3f} units/s x safety "
                f"{self.safety_factor:.2f} = veto above {self.limit:.3f} units/s; "
                f"clock budget {self.clock_error_sec:.2f} s"
                f"{f'; group {self.group}' if self.group else ''}")


# ------------------------------------------------------------------ measurement

def observed_speed_ceiling(samples, min_dt=MIN_FIT_DT_SEC, max_dt=MAX_FIT_DT_SEC,
                           percentile=99.9):
    """Measure how fast people actually move on this floor, in floor units/second.

    `samples` maps a track key -> list of (ts, x, y), in any order. A track key is
    (camera, track_id): observations sharing one are one ByteTrack track, hence
    PROVABLY ONE PERSON walking -- no labels, no operator, no assumption.

    Returns a dict with the percentiles and `ceiling_units_per_sec`, or None when
    there is not enough data to measure. None makes the reachability check
    unavailable, which fails open -- the right outcome for footage too thin to
    measure on.

    ON ID SWITCHES. A ByteTrack track can jump onto a different person, which
    inserts a huge bogus speed. That inflates the ceiling, which makes the envelope
    MORE permissive and produces FEWER vetoes -- the safe direction. It is still
    worth watching: a p99.9 far above p99 usually means switches, not sprinters,
    and the returned percentiles are there so that is visible rather than baked
    silently into a threshold.
    """
    speeds = []
    used_tracks = 0
    for _key, obs in (samples or {}).items():
        rows = sorted((o for o in obs if o[0] is not None), key=lambda o: o[0])
        if len(rows) < 2:
            continue
        counted = False
        for (t0, x0, y0), (t1, x1, y1) in zip(rows, rows[1:]):
            dt = t1 - t0
            if not (min_dt <= dt <= max_dt):
                continue
            speeds.append(math.hypot(x1 - x0, y1 - y0) / dt)
            counted = True
        if counted:
            used_tracks += 1

    if len(speeds) < 30:
        return None

    speeds.sort()

    def pct(p):
        if not speeds:
            return None
        idx = min(len(speeds) - 1, max(0, int(round((p / 100.0) * (len(speeds) - 1)))))
        return speeds[idx]

    return {
        "n_pairs": len(speeds),
        "n_tracks": used_tracks,
        "median": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "p999": pct(99.9),
        "max": speeds[-1],
        "percentile_used": percentile,
        # The ceiling is a high percentile rather than the max: max is an
        # extreme-value statistic that grows with sample size, so a ceiling set
        # from it would drift every time a longer run is measured. This project
        # already learned that on `other MAX` (0.819 -> 0.936 on the same clip at
        # 48 vs 90 frames).
        "ceiling_units_per_sec": pct(percentile),
    }


def format_ceiling_report(stats, units="floor units"):
    """Human-readable block for the calibration tool's output."""
    if stats is None:
        return ("  NOT MEASURABLE -- fewer than 30 usable observation pairs.\n"
                "  The reachability check stays UNAVAILABLE (fails open) until a run\n"
                "  with more co-temporal, calibrated observations is measured.")
    return "\n".join([
        f"  {stats['n_pairs']} consecutive same-track pairs from "
        f"{stats['n_tracks']} track(s)",
        f"  observed speed ({units}/s):",
        f"     median {stats['median']:7.3f}   p95 {stats['p95']:7.3f}   "
        f"p99 {stats['p99']:7.3f}   p99.9 {stats['p999']:7.3f}   "
        f"max {stats['max']:7.3f}",
        f"  ceiling (p{stats['percentile_used']}) = "
        f"{stats['ceiling_units_per_sec']:.3f} {units}/s",
        f"  If p99.9 sits far above p99, suspect ByteTrack id switches rather than",
        f"  fast people. That inflates the ceiling, so the check gets more",
        f"  permissive -- safe, but worth knowing before trusting the number.",
    ])
