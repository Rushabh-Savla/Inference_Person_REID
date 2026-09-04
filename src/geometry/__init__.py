"""
geometry -- where people are standing, and whether one person could have been in
both places.

This package exists for ONE job: answer "could these two observations be the same
person, physically?" It is a CHECK on identity decisions, not a tracker. Nothing
here does 3D tracking, and nothing here creates or merges an identity -- it only
ever reports that a proposed merge is physically impossible.

    src/geometry/calibration.py   the floor-frame record + the metric-scale guard
    src/geometry/floor.py         bbox -> point on a shared floor  (needs cv2/H)
    src/geometry/reachability.py  two recorded points -> possible / impossible
    src/geometry/recorder.py      writes geometry during the LIVE run
    tools/fit_floor_frame.py      fits the floor frame from people's own feet

THIS FILE INTENTIONALLY IMPORTS NOTHING. It is documentation only, so that
`from geometry.reachability import ...` pulls in the reachability arithmetic
WITHOUT dragging in calibration or homography code. Invariant 1 below is enforced
by exactly that: reconcile can import the comparison, and cannot reach the
machinery that would let it recompute a position.

============================ THE THREE INVARIANTS =============================
Breaking any of them has already cost this project a regression. See ADR-003D.

1. THE LIVE RUN RECORDS GEOMETRY. OFFLINE RECONCILE ONLY CONSUMES IT.
   Reconcile must never load a calibration, never apply a homography, never
   re-derive a floor position from a box. It reads the positions the live run
   wrote into the observation payload and compares them.
   Allowed in reconcile:      geometry.reachability   (pure arithmetic)
   Forbidden in reconcile:    geometry.floor, geometry.calibration, cv2 homography
   Asserted by tests/live/test_geometry_not_recomputed.py.

   WHY. The live RTSP feed is never recorded, so a position that was not written
   during the run is gone forever -- and a reconcile that re-derived positions
   would silently produce DIFFERENT geometry from the run it is reconciling the
   moment a calibration is re-fitted. One run, one geometry, decided once.

2. UNITS ARE FLOOR UNITS UNTIL A METRIC REFERENCE IS RECORDED.
   A homography fitted from camera imagery alone fixes the floor plane up to an
   arbitrary scale, so its distances are not metres. `CalibrationRecord.is_metric`
   stays False and the metre-facing API raises until a trusted metric reference is
   supplied. The within-group reachability check needs no metres at all: the
   distance and the speed ceiling share the same unknown unit, and it cancels.

3. THE CHECK FAILS OPEN, ALWAYS.
   Uncalibrated camera, missing bbox, mismatched image size, cameras in different
   floor groups, no timestamp, no measured speed ceiling -> UNAVAILABLE, which
   every consumer treats as "no opinion". Both error budgets bias toward
   PLAUSIBLE. A wrong veto is unrecoverable -- reconcile cannot un-split a person
   it refused to merge -- while a missed veto merely leaves the false merge that
   already exists today.
"""
