"""
calibration.py -- the floor-frame calibration record, and the metric-scale guard.

WHAT A RECORD IS. A JSON file (default `calibration/floor_frame.json`) holding,
per FLOOR GROUP, one homography per camera mapping that camera's image pixels onto
a shared floor coordinate system. A floor group is a set of cameras whose views
overlap enough that a common floor frame can be fitted -- here that is
`cam_219` + `cam_224`, which share a room. Cameras in no group have no geometry,
and that is reported as unavailable, never guessed.

CALIBRATE ONCE PER CAMERA GEOMETRY, NOT PER RUN. The record is keyed by camera
name and the pixel size it was authored in. Any run on those cameras reuses it, so
a live capture needs no calibration step -- which is the whole point: geometry that
has to be re-established per run is geometry nobody will use. Re-fit only when a
camera is physically moved, or `source.resize_width` changes the pixel space.

======================= THE METRIC-SCALE GUARD (read this) =====================

A homography fitted from camera imagery alone determines the floor plane up to an
ARBITRARY SCALE. The maths cannot know whether the floor is a corridor or a
cathedral. So distances out of a self-fitted frame are in FLOOR UNITS -- internally
consistent, comparable to each other, and NOT metres.

    Metric geometry cannot be established from monocular cameras alone. A single
    trusted metric reference must be provided before any metre-based threshold
    (0.5 m, 3.0 m, walking speed in m/s) is considered valid.

`units` is therefore "floor_units" until a `metric_reference` is recorded, and
every metre-facing accessor raises `MetricScaleUnavailable` until then. This is a
hard guard on purpose: the failure it prevents is silent, and a threshold quoted
in metres that is really in unknown units is worse than no threshold at all.

A metric reference is any trustworthy source of real-world scale. Full surveying
is NOT required. In order of preference:

  1. Verified floor plans or CAD drawings.
  2. Known architectural dimensions -- corridor width, door width, column
     spacing, floor-tile pitch -- provided they are verified on site.
  3. One or more independently measured reference distances (laser rangefinder,
     tape, measuring wheel) sufficient to establish AND validate world scale.

Recording one is a two-number edit: the distance between two identifiable floor
points as this frame measures it (floor units), and its real length. Give a second
independent pair when you have one and the loader cross-checks them, which is what
turns "established" into "validated".

WITHIN a floor group, nothing needs metres: the reachability check compares a
floor-unit distance against a floor-unit speed ceiling measured on the same
footage, so the unknown scale cancels exactly. Metres are needed only to relate
SEPARATE floor groups (`group_distances`), because a real-world distance supplied
by an operator arrives in metres and cannot be combined with floor units without a
scale.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

FLOOR_UNITS = "floor_units"
METRES = "metres"

DEFAULT_RECORD_PATH = os.path.join("calibration", "floor_frame.json")

# How far two independent metric references may disagree before the record is
# rejected. Two references that disagree by more than this are not measuring the
# same scale, and picking one of them silently would be the worst outcome.
MAX_METRIC_DISAGREEMENT = 0.05          # 5% of the mean


class MetricScaleUnavailable(RuntimeError):
    """Raised when metres are requested from a record that has no metric reference.

    The message is the whole value of this class -- it has to tell whoever hit it
    what to supply, because the answer is not "write more code".
    """

    def __init__(self, what="a metre-based value"):
        super().__init__(
            f"{what} was requested, but this floor frame has no metric reference.\n"
            "\n"
            "Metric geometry cannot be established from monocular cameras alone. A\n"
            "single trusted metric reference must be provided before any metre-based\n"
            "threshold (0.5 m, 3.0 m, walking speed in m/s) is considered valid.\n"
            "\n"
            "Any trustworthy source of real-world scale will do -- full surveying is\n"
            "NOT required. In order of preference:\n"
            "  1. verified floor plans or CAD drawings\n"
            "  2. known architectural dimensions (corridor/door width, column\n"
            "     spacing, floor-tile pitch), verified on site\n"
            "  3. one or more independently measured reference distances (laser\n"
            "     rangefinder, tape, measuring wheel)\n"
            "\n"
            "Record it with tools/fit_floor_frame.py --metric-reference, then this\n"
            "call succeeds. Until then, stay in floor units: the within-group\n"
            "reachability check needs no metres at all, because the distance and the\n"
            "speed ceiling share the same unknown unit and it cancels.")


class CalibrationRecord:
    """A parsed, validated floor-frame record.

    Read-only. Construct via `load_calibration()` or `CalibrationRecord(blob)`.
    """

    def __init__(self, blob, path=None):
        self._blob = dict(blob or {})
        self.path = path
        self.calib_version = self._blob.get("calib_version")
        self.created_at = self._blob.get("created_at")
        self.units = self._blob.get("units", FLOOR_UNITS)
        self.notes = self._blob.get("notes", "")
        self.source = self._blob.get("source", "")
        self._groups = self._blob.get("groups") or {}
        self._metric = self._blob.get("metric_reference")

        # camera -> group name. Built once; a camera in two groups is a config
        # error we refuse rather than resolve arbitrarily.
        self._group_of = {}
        for gname, g in self._groups.items():
            for cam in (g.get("cameras") or {}):
                if cam in self._group_of:
                    raise ValueError(
                        f"[geometry] camera {cam} appears in two floor groups "
                        f"({self._group_of[cam]}, {gname}). A camera stands on one "
                        f"floor frame; fix the record.")
                self._group_of[cam] = gname

        self._validate_metric_reference()

    # ---- metric scale ------------------------------------------------------

    def _validate_metric_reference(self):
        """Reject a half-recorded or self-contradicting metric reference.

        A malformed reference must not degrade to "no scale" quietly: someone went
        to the trouble of measuring, and silently ignoring it would produce
        floor-unit thresholds that everyone reads as metres.
        """
        if self._metric is None:
            self._metres_per_unit = None
            return

        refs = self._metric if isinstance(self._metric, list) else [self._metric]
        ratios = []
        for i, ref in enumerate(refs):
            for field in ("source", "description", "floor_units", "metres"):
                if ref.get(field) in (None, ""):
                    raise ValueError(
                        f"[geometry] metric_reference[{i}] is missing '{field}'. A "
                        f"reference needs the distance as THIS frame measures it "
                        f"(floor_units), its real length (metres), where the number "
                        f"came from (source), and what was measured (description).")
            fu = float(ref["floor_units"])
            m = float(ref["metres"])
            if fu <= 0 or m <= 0:
                raise ValueError(
                    f"[geometry] metric_reference[{i}] has a non-positive distance "
                    f"(floor_units={fu}, metres={m}).")
            ratios.append(m / fu)

        if len(ratios) > 1:
            spread = (max(ratios) - min(ratios)) / (sum(ratios) / len(ratios))
            if spread > MAX_METRIC_DISAGREEMENT:
                raise ValueError(
                    f"[geometry] the {len(ratios)} metric references disagree by "
                    f"{spread * 100:.1f}% (>{MAX_METRIC_DISAGREEMENT * 100:.0f}%): "
                    f"{', '.join(f'{r:.4f}' for r in ratios)} m per floor unit.\n"
                    f"        They are not measuring the same scale. Do not average "
                    f"them -- find which one is wrong. A disagreement this large "
                    f"usually means one pair of reference points is not on the floor "
                    f"plane, or the two points were identified in different frames.")
        self._metres_per_unit = sum(ratios) / len(ratios)

    @property
    def is_metric(self):
        """True when a validated metric reference exists, so metres are meaningful."""
        return self._metres_per_unit is not None

    @property
    def metres_per_unit(self):
        """Scale from floor units to metres. Raises when no reference is recorded."""
        if self._metres_per_unit is None:
            raise MetricScaleUnavailable("the floor-unit-to-metre scale")
        return self._metres_per_unit

    def to_metres(self, floor_units):
        """Convert a floor-unit distance to metres. Raises without a reference."""
        return float(floor_units) * self.metres_per_unit

    def describe_units(self):
        """One line for a log or a report -- never claims metres it does not have."""
        if self.is_metric:
            return (f"metres (1 floor unit = {self._metres_per_unit:.4f} m, from "
                    f"{self._metric_sources()})")
        return "floor units (UNSCALED -- no metric reference; metres unavailable)"

    def _metric_sources(self):
        refs = self._metric if isinstance(self._metric, list) else [self._metric]
        return "; ".join(str(r.get("source")) for r in refs)

    # ---- groups and cameras ------------------------------------------------

    @property
    def cameras(self):
        return sorted(self._group_of)

    @property
    def groups(self):
        return sorted(self._groups)

    def group_of(self, camera):
        """Which floor group a camera belongs to, or None if it is uncalibrated."""
        return self._group_of.get(camera)

    def same_group(self, cam_a, cam_b):
        """True only when both cameras are calibrated INTO THE SAME floor frame.

        Two cameras in different groups have no comparable coordinates, so every
        geometric question about them is unanswerable -- not "far apart".
        """
        ga, gb = self._group_of.get(cam_a), self._group_of.get(cam_b)
        return ga is not None and ga == gb

    def camera_entry(self, camera):
        g = self._group_of.get(camera)
        if g is None:
            return None
        return (self._groups[g].get("cameras") or {}).get(camera)

    def group_entry(self, group):
        return self._groups.get(group)

    def homography(self, camera):
        """3x3 image->floor matrix for a camera, or None if uncalibrated."""
        entry = self.camera_entry(camera)
        if not entry or entry.get("H") is None:
            return None
        H = np.asarray(entry["H"], dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"[geometry] {camera}: H must be 3x3, got {H.shape}.")
        return H

    def image_size(self, camera):
        """(width, height) the homography was authored in, or None.

        NOT optional. `source.resize_width` resizes frames BEFORE detection, so
        boxes -- and therefore foot points -- arrive in resized pixels. A
        homography authored at another resolution is silently wrong, with no error
        anywhere. floor.py rescales into this space, or reports unavailable.
        """
        entry = self.camera_entry(camera)
        if not entry:
            return None
        size = entry.get("image_size")
        if not size or len(size) != 2:
            raise ValueError(
                f"[geometry] {camera}: image_size is missing from the calibration "
                f"record. It is required -- a homography belongs to the pixel space "
                f"it was fitted in (see the docstring).")
        return (int(size[0]), int(size[1]))

    def speed_ceiling(self, camera):
        """Top plausible human speed on this floor, in FLOOR UNITS per second.

        Measured from the footage itself (see reachability.observed_speed_ceiling),
        not assumed, and stored per group because each group has its own unknown
        scale. None when the group has no measurement yet, which makes the
        reachability check unavailable rather than guessing a number.
        """
        g = self._group_of.get(camera)
        if g is None:
            return None
        v = self._groups[g].get("speed_ceiling_units_per_sec")
        return None if v is None else float(v)

    def position_error(self, camera):
        """Per-observation position uncertainty in floor units, from the fit.

        This is the calibration's own held-out residual, inflated for a clipped
        box by floor.py. It is subtracted from every measured distance, so a
        sloppier calibration produces FEWER vetoes rather than wrong ones.
        """
        entry = self.camera_entry(camera) or {}
        if entry.get("position_error_units") is not None:
            return float(entry["position_error_units"])
        g = self._group_of.get(camera)
        if g is not None:
            v = self._groups[g].get("position_error_units")
            if v is not None:
                return float(v)
        return None

    def group_distance_metres(self, cam_a, cam_b):
        """Real-world distance between two SEPARATE floor groups, in metres.

        This is the only place operator-supplied real-world distances live, and the
        only path to cross-group reachability -- non-overlapping cameras share no
        floor points, so no homography can relate them. Returns None when the pair
        is not recorded, which is today's state for cam_206 and cam_213.
        """
        ga, gb = self._group_of.get(cam_a), self._group_of.get(cam_b)
        if ga is None or gb is None or ga == gb:
            return None
        table = self._blob.get("group_distances") or {}
        for key in (f"{ga}|{gb}", f"{gb}|{ga}"):
            if key in table:
                entry = table[key]
                m = entry.get("metres") if isinstance(entry, dict) else entry
                return None if m is None else float(m)
        return None

    # ---- reporting ---------------------------------------------------------

    def summary(self):
        """Multi-line description for startup logs and calibration reports."""
        lines = [
            f"floor frame {self.calib_version or '(unversioned)'}"
            f"{'  from ' + os.path.relpath(self.path) if self.path else ''}",
            f"  units       {self.describe_units()}",
        ]
        if not self._groups:
            lines.append("  groups      NONE -- no camera has geometry")
        for gname in self.groups:
            g = self._groups[gname]
            cams = sorted((g.get("cameras") or {}))
            ceiling = g.get("speed_ceiling_units_per_sec")
            perr = g.get("position_error_units")
            lines.append(f"  group       {gname}: {', '.join(cams)}")
            lines.append(
                f"              speed ceiling "
                f"{'NOT MEASURED' if ceiling is None else f'{ceiling:.3f} units/s'}"
                f"   position error "
                f"{'unknown' if perr is None else f'{perr:.3f} units'}")
        return "\n".join(lines)


def load_calibration(path=None, required=False):
    """Load a floor-frame record.

    Returns None when the file does not exist and `required` is False -- geometry
    is optional, and its absence must degrade to today's behaviour rather than
    stopping a run.
    """
    path = path or DEFAULT_RECORD_PATH
    if not os.path.exists(path):
        if required:
            raise SystemExit(
                f"[geometry] no calibration record at {path}.\n"
                f"        Fit one from a completed run's stored observations -- it "
                f"needs no camera time and no measuring:\n"
                f"          python tools/fit_floor_frame.py <run_id>")
        return None
    with open(path) as f:
        blob = json.load(f)
    return CalibrationRecord(blob, path=path)


def write_calibration(record_blob, path=None):
    """Write a record, stamping `calib_version` and `created_at` if absent.

    Validates by round-tripping through CalibrationRecord first, so a malformed
    record is never left on disk for a later run to trip over.
    """
    path = path or DEFAULT_RECORD_PATH
    blob = dict(record_blob)
    blob.setdefault("calib_version", time.strftime("%Y%m%d_%H%M%S"))
    blob.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    blob.setdefault("units", FLOOR_UNITS)
    parsed = CalibrationRecord(blob, path=path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    return parsed
