"""
The floor frame -- bbox to a point on a shared floor, and every way it refuses.

The interesting content here is the REFUSALS. A homography always returns a number;
what makes this safe to hang a hard veto on is the set of cases where it declines to.
Each check below corresponds to a way a wrong position could reach the reachability
veto and split a real person permanently.

Synthetic and exact: the homography is built by hand, so the expected coordinates are
arithmetic rather than approximations. No footage, no calibration file, no cv2 fit.
"""

import sys

from _synth import Check

from geometry.calibration import CalibrationRecord
from geometry.floor import (CLIPPED_ERROR_MULTIPLIER, FloorFrame,
                            apply_homography, foot_point)

SIZE = [1000, 800]


def record(H_b=None, error=0.2, size_a=None, size_b=None, speed=1.0):
    """A two-camera record where cam_a IS the floor frame (identity homography)."""
    cams = {
        "cam_a": {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  "image_size": size_a or list(SIZE),
                  "position_error_units": error},
    }
    if H_b is not None:
        cams["cam_b"] = {"H": H_b, "image_size": size_b or list(SIZE),
                         "position_error_units": error}
    return CalibrationRecord({
        "calib_version": "test",
        "groups": {"room": {"cameras": cams,
                            "position_error_units": error,
                            "speed_ceiling_units_per_sec": speed}},
    })


def main():
    c = Check("geometry: the floor frame")

    # ------------------------------------------------------------------ 1
    print("\n1. THE FOOT POINT IS BOTTOM-CENTRE, AND A BAD BOX HAS NONE")

    c.eq(foot_point([10, 20, 30, 60]), (20.0, 60.0),
         "bottom-centre of a box is (mid-x, bottom-y)")
    for bad, why in (([10, 20, 10, 60], "zero width"),
                     ([10, 60, 30, 20], "inverted"),
                     ([10, 20, 30, 20], "zero height"),
                     (None, "missing"),
                     ([1, 2, 3], "wrong length"),
                     ([float("nan"), 0, 10, 10], "NaN")):
        c.ok(foot_point(bad) is None,
             f"a {why} box has no foot point -- no phantom position on the floor")

    # ------------------------------------------------------------------ 2
    print("\n2. THE HORIZON GUARD -- an off-plane point is missing, not enormous")

    # A homography whose third row sends v = 500 to the vanishing line. A foot point
    # estimated slightly too high in the image (a long coat, a clipped box) lands
    # there, and without the guard it becomes an astronomically large coordinate that
    # reads as "kilometres away" and vetoes a correct merge.
    H = [[1, 0, 0], [0, 1, 0], [0, -1.0 / 500.0, 1.0]]
    c.ok(apply_homography(__import__("numpy").asarray(H, dtype=float), 100, 500)
         is None, "a point ON the vanishing line returns None, not a huge number")
    got = apply_homography(__import__("numpy").asarray(H, dtype=float), 100, 250)
    c.ok(got is not None and abs(got[0] - 200.0) < 1e-9,
         f"a point safely below it projects normally (got {got})")

    # ------------------------------------------------------------------ 3
    print("\n3. AN UNCALIBRATED CAMERA HAS NO POSITION")

    frame = FloorFrame(record())
    c.ok(frame.is_calibrated("cam_a"), "cam_a is calibrated")
    c.ok(not frame.is_calibrated("cam_206"),
         "cam_206 is not in the record, so it is not calibrated")
    c.ok(frame.position("cam_206", [10, 20, 30, 60]) is None,
         "and asking for its position returns None -- appearance-only, fail-open")

    c.ok(FloorFrame(None).position("cam_a", [10, 20, 30, 60]) is None,
         "no record at all -> no positions anywhere")
    c.ok(not FloorFrame(None).enabled, "and the frame reports itself disabled")

    # A camera present in the record but with no measured position error cannot be
    # used: the error radius is what makes the veto safe, so a missing one must not
    # silently become zero.
    no_err = CalibrationRecord({
        "calib_version": "t",
        "groups": {"room": {"cameras": {
            "cam_a": {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      "image_size": list(SIZE)}}}}})
    c.ok(FloorFrame(no_err).position("cam_a", [10, 20, 30, 60]) is None,
         "a camera with NO measured position error yields no position -- a zero "
         "radius would make the veto maximally aggressive by accident")

    # ------------------------------------------------------------------ 4
    print("\n4. THE PIXEL SPACE IS PART OF THE CALIBRATION")

    # source.resize_width resizes frames BEFORE detection, so boxes arrive in resized
    # pixels. A homography authored at another resolution is silently wrong, with no
    # error anywhere -- ADR-003A section 1.3.
    p_native = frame.position("cam_a", [400, 300, 600, 700], frame_size=(1000, 800))
    c.ok(p_native is not None and p_native.x == 500.0 and p_native.y == 700.0,
         f"at the calibrated size the foot point maps straight through ({p_native})")

    # Half-size frame: the box is rescaled back into the calibrated space, so the
    # SAME physical foot lands on the same floor point.
    p_half = frame.position("cam_a", [200, 150, 300, 350], frame_size=(500, 400))
    c.ok(p_half is not None and abs(p_half.x - 500.0) < 1e-9
         and abs(p_half.y - 700.0) < 1e-9,
         f"a uniformly resized frame is rescaled, giving the same point ({p_half})")

    # A non-uniform change is not a resize and cannot be undone by one scale factor.
    c.ok(frame.position("cam_a", [200, 300, 300, 700], frame_size=(500, 800)) is None,
         "a frame whose ASPECT RATIO differs returns None rather than a tilted floor")

    # ------------------------------------------------------------------ 5
    print("\n5. A CLIPPED BOX IS KEPT, BUT TRUSTED LESS")

    # Feet outside the frame mean bottom-centre is not ground contact. Dropping the
    # detection would lose a person; widening its error radius keeps it while making
    # it unable to drive a veto on its own.
    clean = frame.position("cam_a", [400, 300, 600, 700])
    clipped_bottom = frame.position("cam_a", [400, 300, 600, 800])
    clipped_left = frame.position("cam_a", [0, 300, 200, 700])
    c.ok(clean is not None and not clean.clipped, "an interior box is not clipped")
    c.ok(clipped_bottom is not None and clipped_bottom.clipped,
         "a box touching the frame BOTTOM is clipped")
    c.ok(clipped_left is not None and clipped_left.clipped,
         "a box touching the frame LEFT edge is clipped")
    c.ok(abs(clipped_bottom.error - clean.error * CLIPPED_ERROR_MULTIPLIER) < 1e-9,
         f"and its error radius is {CLIPPED_ERROR_MULTIPLIER}x wider "
         f"({clean.error} -> {clipped_bottom.error}), so it vetoes less")

    # ------------------------------------------------------------------ 6
    print("\n6. THE PAYLOAD SHAPE IS WHAT RECONCILE READS")

    d = clean.as_dict()
    for field in ("x", "y", "error", "group", "source", "clipped"):
        c.ok(field in d, f"the recorded floor block carries {field!r}")
    c.eq(d["group"], "room", "the floor GROUP travels with the position -- two "
                             "groups' coordinates are never comparable")

    from geometry.reachability import RecordedPosition
    round_tripped = RecordedPosition.from_payload({"ts": 1.0, "floor": d})
    c.ok(round_tripped is not None
         and round_tripped.x == d["x"] and round_tripped.error == d["error"],
         "and it round-trips through the consumer reconcile actually uses")

    # ------------------------------------------------------------------ 7
    print("\n7. TWO CAMERAS, ONE FLOOR -- and cameras in different groups")

    # cam_b's floor is cam_a's shifted by (100, 50): the same physical spot seen from
    # either camera must land on the same floor point.
    shifted = FloorFrame(record(H_b=[[1, 0, 100], [0, 1, 50], [0, 0, 1]]))
    pa = shifted.position("cam_a", [400, 300, 600, 700])
    pb = shifted.position("cam_b", [300, 250, 500, 650])
    c.ok(pa is not None and pb is not None
         and abs(pa.x - pb.x) < 1e-9 and abs(pa.y - pb.y) < 1e-9,
         f"both cameras place the same spot at one floor point ({pa} / {pb})")
    c.eq(pa.group, pb.group, "and they report the same floor group")

    two_groups = CalibrationRecord({
        "calib_version": "t",
        "groups": {
            "room": {"cameras": {"cam_a": {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                           "image_size": list(SIZE),
                                           "position_error_units": 0.2}}},
            "corridor": {"cameras": {"cam_c": {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                               "image_size": list(SIZE),
                                               "position_error_units": 0.2}}},
        }})
    f2 = FloorFrame(two_groups)
    c.ok(not two_groups.same_group("cam_a", "cam_c"),
         "cameras in different floor groups are NOT in the same group")
    c.eq(f2.group_of("cam_a"), "room", "each camera reports its own group")
    c.ok(two_groups.group_distance_metres("cam_a", "cam_c") is None,
         "and no real-world distance between the groups is recorded yet, so "
         "cross-group reachability stays unavailable")

    # A camera listed in two groups is a contradiction, not something to resolve.
    try:
        CalibrationRecord({"groups": {
            "g1": {"cameras": {"cam_a": {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                         "image_size": list(SIZE)}}},
            "g2": {"cameras": {"cam_a": {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                         "image_size": list(SIZE)}}}}})
        c.ok(False, "a camera in two groups should have been refused")
    except ValueError as e:
        c.ok("two floor groups" in str(e),
             "a camera listed in two floor groups is refused -- it stands on one floor")

    # image_size is not optional: without it a homography cannot be checked against
    # the frames it is being applied to.
    try:
        CalibrationRecord({"groups": {"room": {"cameras": {
            "cam_a": {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}}}}).image_size("cam_a")
        c.ok(False, "a missing image_size should have been refused")
    except ValueError as e:
        c.ok("image_size" in str(e), "a record with no image_size is refused")

    c.done()


if __name__ == "__main__":
    sys.exit(main() or 0)
