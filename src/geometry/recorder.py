"""
recorder.py -- the ONLY place a floor position is ever computed.

Invariant 1 (geometry/__init__.py): the live run records geometry; offline
reconcile only consumes it. This module is the recording half. It runs inside the
live `IdentityStage` -- the single-threaded stage that already owns the store
writes -- and attaches a `floor` block to each observation payload as it is
persisted, plus a line to a sidecar log.

WHY THE POSITION IS WRITTEN TWICE, TO TWO PLACES

  * `payload["floor"]`, in Qdrant, beside the embedding. This is what offline
    reconcile reads. It travels with the observation, so a run reconciled today and
    the same run re-reconciled in six months see IDENTICAL geometry, which is the
    entire point of invariant 1.

  * `logs/geometry_<run_id>.jsonl`, a flat sidecar. This carries what reconcile
    does NOT need but analysis does -- above all `foot_img`, the position in IMAGE
    pixels. Because the image-space foot point is kept, re-deriving positions under
    a better calibration is a matrix multiply over a text file rather than a
    re-detection: a 20-second job instead of another capture. On a live path that
    records no video, this is the difference between iterating and re-staging.

    The sidecar is analysis input. It is NOT an input to reconcile, and re-deriving
    positions into it does not change any past run's identities -- that would
    violate invariant 1. A re-derivation produces a NEW run's geometry, for
    comparison.

DEGRADE, NEVER CRASH. A geometry failure must not cost a run its identities. Every
entry point swallows its own errors, disables itself after the first failure, and
says so once. This mirrors how `IdentityStage` treats store persistence -- except
that stage's silence on dimension mismatches is a known defect, so this one COUNTS
what it dropped and reports it in the run summary.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from geometry.floor import FloorFrame, foot_point


class GeometryRecorder:
    """Computes and records floor positions during a live run.

    Thread-safety: none, deliberately. `IdentityStage` is a single thread and is the
    sole caller, matching how it already reasons about the store ("this
    single-threaded stage is the sole writer, so no store lock is needed").
    """

    def __init__(self, record, log_path=None, run_id=None, log=print):
        self.frame = FloorFrame(record)
        self.record = self.frame.record
        self.run_id = run_id
        self.log_path = log_path
        self._log = log
        self._fh = None
        self._failed = False
        self._log_failed = False

        # Counters, reported in the run summary. A geometry subsystem that silently
        # produced nothing would look identical to one that was disabled, and this
        # project has already been bitten by exactly that shape of silence.
        self.recorded = 0
        self.unavailable = 0
        self.by_camera = defaultdict(int)
        self.unavailable_by_camera = defaultdict(int)

    # ---- lifecycle ---------------------------------------------------------

    @property
    def enabled(self):
        return (not self._failed) and self.frame.enabled

    def cameras_without_geometry(self, cameras):
        """Which of this run's cameras the calibration does not cover.

        Named at startup rather than discovered from a zero counter later: a camera
        quietly contributing no geometry is indistinguishable from geometry being
        off, and that ambiguity is what makes a silent failure expensive.
        """
        return [c for c in cameras if not self.frame.is_calibrated(c)]

    def _sidecar(self):
        if self._fh is not None or self._log_failed or not self.log_path:
            return self._fh
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)),
                        exist_ok=True)
            # Line-buffered: an unclean kill costs at most one record, and the
            # guarded finalize phase (pipeline.py's InterruptGuard) closes it
            # properly on a normal stop.
            self._fh = open(self.log_path, "a", buffering=1)
        except OSError as e:
            self._log_failed = True
            self._log(f"[geometry] sidecar log disabled ({e}); positions still go "
                      f"to the observation payload, so reconcile is unaffected.")
        return self._fh

    def close(self):
        """Flush and close the sidecar. Call inside the guarded finalize phase."""
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    # ---- the recording call ------------------------------------------------

    def annotate(self, payload, camera, bbox, ts, frame_index=None,
                 track_id=None, frame_size=None):
        """Attach a `floor` block to `payload` in place, and log the full record.

        Returns the FloorPosition, or None when geometry is unavailable for this
        observation. `payload` is left untouched in that case -- an absent `floor`
        key is how a consumer learns there was no position, and writing a null
        would be a claim we cannot support.
        """
        if not self.enabled:
            return None
        try:
            pos = self.frame.position(camera, bbox, frame_size=frame_size)
        except Exception as e:                                   # noqa: BLE001
            self._failed = True
            self._log(f"[geometry] DISABLED after error computing a position: {e}. "
                      f"The run continues; identities are unaffected, but this run "
                      f"records no geometry and offline reconcile will have no "
                      f"reachability check.")
            return None

        if pos is None:
            self.unavailable += 1
            self.unavailable_by_camera[camera] += 1
            return None

        payload["floor"] = pos.as_dict()
        payload["floor"]["calib_version"] = self.record.calib_version
        self.recorded += 1
        self.by_camera[camera] += 1
        self._write_sidecar(pos, camera, bbox, ts, frame_index, track_id)
        return pos

    def _write_sidecar(self, pos, camera, bbox, ts, frame_index, track_id):
        fh = self._sidecar()
        if fh is None:
            return
        fp = foot_point(bbox)
        rec = {
            "run_id": self.run_id,
            "camera": camera,
            "track_id": None if track_id is None else int(track_id),
            "frame": None if frame_index is None else int(frame_index),
            "ts": None if ts is None else float(ts),
            "bbox": [round(float(v), 2) for v in bbox] if bbox else None,
            # The most valuable field in the record: with the IMAGE-space foot
            # point kept, a better calibration can be applied offline without
            # re-running detection.
            "foot_img": [round(fp[0], 2), round(fp[1], 2)] if fp else None,
            "foot_source": pos.source,
            "clipped": pos.clipped,
            "floor": [round(pos.x, 4), round(pos.y, 4)],
            "floor_error": round(pos.error, 4),
            "group": pos.group,
            "calib_version": self.record.calib_version,
            "units": self.record.units,
        }
        try:
            fh.write(json.dumps(rec) + "\n")
        except (OSError, TypeError, ValueError) as e:
            self._log_failed = True
            self._log(f"[geometry] sidecar log disabled after a write error ({e}).")
            self.close()

    # ---- reporting ---------------------------------------------------------

    def stats(self):
        return {
            "geometry_recorded": self.recorded,
            "geometry_unavailable": self.unavailable,
            "geometry_by_camera": dict(self.by_camera),
            "geometry_unavailable_by_camera": dict(self.unavailable_by_camera),
            "geometry_failed": self._failed,
        }

    def summary(self):
        """Lines for the end-of-run report."""
        if not self.frame.enabled:
            return ["Geometry: no calibrated camera -- nothing recorded."]
        total = self.recorded + self.unavailable
        pct = (100.0 * self.recorded / total) if total else 0.0
        lines = [f"Geometry: {self.recorded}/{total} observations positioned "
                 f"({pct:.1f}%), units {self.record.units}"]
        for cam in sorted(set(self.by_camera) | set(self.unavailable_by_camera)):
            ok = self.by_camera.get(cam, 0)
            bad = self.unavailable_by_camera.get(cam, 0)
            lines.append(f"  {cam}: {ok} positioned, {bad} unavailable")
        if self._failed:
            lines.append("  !! recording was DISABLED mid-run by an error (above).")
        if self.recorded == 0 and total > 0:
            lines.append("  !! NOTHING was positioned. Offline reconcile will have "
                         "no reachability check. Check that the calibration's "
                         "image_size matches this run's frames.")
        return lines
