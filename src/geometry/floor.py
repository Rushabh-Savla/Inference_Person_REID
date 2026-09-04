"""
floor.py -- a bounding box becomes a point on a shared floor.

    bbox (image pixels) --foot point--> (u, v) --H--> (x, y) on the floor frame

FAIL-OPEN IS THE CONTRACT. Every path that cannot produce a trustworthy position
returns None, and every consumer treats None as "no opinion". Uncalibrated camera,
missing box, degenerate box, a frame size the homography was not authored in, a
point at or beyond the plane's horizon -> None. There is no code path that guesses
a coordinate, because a wrong position produces a wrong VETO, and a veto is
unrecoverable: reconcile cannot un-split a person it refused to merge.

THE FOOT POINT IS BOX-BOTTOM-CENTRE, and that is a deliberate floor on accuracy
rather than a shortcut. Ankle keypoints would be better -- COCO 15/16 -- but
`live.inference.pose_ensemble: false` on the live path, deliberately, because pose
costs ~2x detection throughput and the resulting dropped frames fragment tracks
worse than better foot points repair. Since the live run is the only writer of
geometry (ADR-003D invariant 1), the live path's foot point is the only one that
matters, and it has no keypoints. The consequence is priced in honestly instead:
box-bottom error lands in `position_error_units`, which is SUBTRACTED from every
measured distance, so a poor foot point yields fewer vetoes, never wrong ones.

WHY A HOMOGRAPHY AND NOT 3D. People stand on one flat floor, so the box->floor map
is a planar homography: 8 degrees of freedom, no intrinsics, no distortion model,
no bundle adjustment, and cv2 already ships it. 3D tracking would be a different
project and would answer a question nobody here asked.
"""

from __future__ import annotations

import numpy as np

from geometry.calibration import CalibrationRecord

# A box whose bottom edge sits within this many pixels of the frame bottom, or
# whose left/right edge touches the frame border, is CLIPPED: the person's feet
# are probably outside the image, so bottom-centre is not their ground contact.
# We do not drop these -- a clipped box is still a person -- we widen their error.
CLIP_MARGIN_PX = 2
CLIPPED_ERROR_MULTIPLIER = 3.0


class FloorPosition:
    """One observation's position on the floor frame, with its uncertainty.

    `error` is in the same floor units as `x`/`y` and is the radius inside which
    the true position plausibly sits. Consumers subtract it from distances rather
    than treating the position as exact.
    """

    __slots__ = ("x", "y", "error", "group", "camera", "source", "clipped")

    def __init__(self, x, y, error, group, camera, source="box_bottom_centre",
                 clipped=False):
        self.x = float(x)
        self.y = float(y)
        self.error = float(error)
        self.group = group
        self.camera = camera
        self.source = source
        self.clipped = bool(clipped)

    def as_dict(self):
        """The shape written into the observation payload and the sidecar log.

        Deliberately flat and JSON-native: this is what offline reconcile reads,
        and reconcile must be able to consume it without importing this package.
        """
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "error": round(self.error, 4),
            "group": self.group,
            "source": self.source,
            "clipped": self.clipped,
        }

    def __repr__(self):
        return (f"FloorPosition({self.camera} {self.group} "
                f"({self.x:.2f}, {self.y:.2f}) +-{self.error:.2f})")


def foot_point(bbox):
    """Bottom-centre of a box, as (u, v) in image pixels. None for a bad box.

    A zero-area or inverted box is not a person standing anywhere, and inventing a
    point for it would put a phantom position on the floor.
    """
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not (x2 > x1 and y2 > y1):
        return None
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    return (0.5 * (x1 + x2), y2)


def _is_clipped(bbox, image_size):
    """Is the box cut off by a frame edge, so its bottom is not the feet?"""
    if image_size is None:
        return False
    w, h = image_size
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (y2 >= h - CLIP_MARGIN_PX
            or x1 <= CLIP_MARGIN_PX
            or x2 >= w - CLIP_MARGIN_PX)


def apply_homography(H, u, v):
    """One point through a 3x3 homography. None at or beyond the horizon.

    The guard is not theoretical. A floor homography sends the plane's vanishing
    line to infinity, and a foot point estimated slightly too high in the image (a
    long coat, a box clipped at the top of a stairwell) can land on the far side of
    it. Without the check that becomes an enormous finite coordinate, which then
    reads as "kilometres away" and vetoes a correct merge. A missing value is the
    honest answer.
    """
    denom = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        return None
    x = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / denom
    y = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / denom
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (x, y)


class FloorFrame:
    """Turns boxes into floor positions, for the cameras a record covers.

    Stateless apart from the calibration it holds, and cheap to call per detection:
    one 3x3 matrix-vector product. Construct once per run.
    """

    def __init__(self, record):
        if record is not None and not isinstance(record, CalibrationRecord):
            record = CalibrationRecord(record)
        self.record = record
        self._cache = {}                # camera -> (H, image_size, error, group)

    # ---- availability ------------------------------------------------------

    @property
    def enabled(self):
        return self.record is not None and bool(self.record.cameras)

    def is_calibrated(self, camera):
        return self.enabled and self.record.group_of(camera) is not None

    def group_of(self, camera):
        return self.record.group_of(camera) if self.enabled else None

    def _entry(self, camera):
        if camera in self._cache:
            return self._cache[camera]
        out = None
        if self.is_calibrated(camera):
            H = self.record.homography(camera)
            size = self.record.image_size(camera)
            err = self.record.position_error(camera)
            group = self.record.group_of(camera)
            if H is not None and err is not None:
                out = (H, size, float(err), group)
        self._cache[camera] = out
        return out

    # ---- the one useful method --------------------------------------------

    def position(self, camera, bbox, frame_size=None):
        """-> FloorPosition, or None when no trustworthy position exists.

        `frame_size` is the (width, height) the bbox was measured in. When it
        differs from the calibrated size the box is rescaled into the calibrated
        pixel space -- correct for a pure resize, which is what
        `source.resize_width` does. An aspect-ratio change is NOT a pure resize, so
        that returns None instead of a subtly wrong coordinate.
        """
        entry = self._entry(camera)
        if entry is None:
            return None
        H, calib_size, base_error, group = entry

        bbox = _rescale_bbox(bbox, frame_size, calib_size)
        if bbox is None:
            return None

        fp = foot_point(bbox)
        if fp is None:
            return None

        xy = apply_homography(H, fp[0], fp[1])
        if xy is None:
            return None

        clipped = _is_clipped(bbox, calib_size)
        error = base_error * (CLIPPED_ERROR_MULTIPLIER if clipped else 1.0)
        return FloorPosition(xy[0], xy[1], error, group, camera,
                             source="box_bottom_centre", clipped=clipped)


def _rescale_bbox(bbox, frame_size, calib_size):
    """Put a bbox into the pixel space the homography was authored in, or None."""
    if bbox is None or len(bbox) != 4:
        return None
    if frame_size is None or calib_size is None or tuple(frame_size) == tuple(calib_size):
        return bbox
    fw, fh = float(frame_size[0]), float(frame_size[1])
    cw, ch = float(calib_size[0]), float(calib_size[1])
    if fw <= 0 or fh <= 0:
        return None
    sx, sy = cw / fw, ch / fh
    # A pure resize scales both axes equally. Anything else (letterboxing, a crop,
    # a different sensor mode) is not something a single scale factor can undo, and
    # applying one anyway would tilt the floor plane.
    if abs(sx - sy) > 0.01 * max(sx, sy):
        return None
    return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
