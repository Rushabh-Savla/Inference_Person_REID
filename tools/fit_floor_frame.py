"""
Fit the shared floor frame from PEOPLE'S OWN FEET. No clicking, no floor plan, no
tape measure, no camera time.

    # fit from a finished run's stored observations
    python tools/fit_floor_frame.py <run_id>

    # inspect what it would do, write nothing
    python tools/fit_floor_frame.py <run_id> --dry-run

    # record a metric reference, so metres become available (see METRIC SCALE below)
    python tools/fit_floor_frame.py <run_id> \
        --metric-reference "floor_plan:corridor width cam_219 doorframe to wall:3.42:2.10"

=============================== WHAT THIS DOES ================================

Two cameras that see the same room see the same people standing on the same floor.
So the people ARE the calibration targets:

  1. find tracklet pairs across the two cameras that appearance says are confidently
     the same person, and that overlap in wall-clock time;
  2. take each pair's co-temporal foot points as one image->image correspondence;
  3. RANSAC a homography from cam_b's floor pixels into cam_a's;
  4. declare cam_a's pixel plane the shared FLOOR FRAME.

This is ADR-003A section 6's idea -- AutoMagicCalib's insight without its GPU:
derive calibration from tracked objects rather than a checkerboard. It inverts
MV3DT's dependency, where calibration is a precondition for identity. Here identity
produces calibration, which then constrains identity.

The inputs are already in Qdrant. Every observation has carried `bbox` and `ts`
since the live reconcile landed, so ANY finished multi-camera run can be calibrated
retroactively, with the cameras switched off.

============================ WHY IT IS NOT CIRCULAR ============================

It is fitted from appearance matches, and appearance is the thing we are trying to
constrain. That is a real objection, and it is answered by the VALIDATION rather
than by the fit:

  Two tracklets that co-occur in ONE camera are provably two different people --
  one body cannot be two simultaneous detections. So for a tracklet X in cam_a and
  two mutually co-present tracklets Y, Z in cam_b, X can be the same person as AT
  MOST ONE of them. A correct homography must put X near one and far from the other.

Every such triangle is a self-validating test with no operator, no labels and no
threshold chosen in advance. The fit uses high-confidence matches; the SCORE uses
provably-distinct people. A homography fitted from wrong matches fails the triangle
test, and this tool refuses to write a record that does.

A second, independent check: RANSAC's own inlier fraction. Correspondences from two
different people, or from foot points that are not on the floor plane, do not admit
one consistent homography.

================================ METRIC SCALE =================================

The output is in FLOOR UNITS -- cam_a pixels, internally consistent, NOT metres. A
homography fitted from imagery alone fixes the plane only up to scale, and the
reachability check needs no more than that: it compares a floor-unit distance to a
floor-unit speed ceiling measured on the same footage, so the unknown scale cancels.

    Metric geometry cannot be established from monocular cameras alone. A single
    trusted metric reference must be provided before any metre-based threshold
    (0.5 m, 3.0 m, walking speed in m/s) is considered valid.

Supply one with `--metric-reference "source:description:floor_units:metres"` when a
trustworthy source exists. Full surveying is NOT required. In order of preference:
verified floor plans or CAD; known architectural dimensions (corridor width, door
width, column spacing, tile pitch) verified on site; one or more independently
measured reference distances. Pass the flag twice and the two are cross-checked
against each other, which is what turns "established" into "validated".

READ-ONLY on the gallery. Writes exactly one file: the calibration record.
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import cv2                                                          # noqa: E402
import yaml                                                         # noqa: E402

from geometry.calibration import write_calibration                  # noqa: E402
from geometry.floor import foot_point                               # noqa: E402
from geometry.reachability import (format_ceiling_report,            # noqa: E402
                                  observed_speed_ceiling)

# Defaults. Every one of these is a knob on the FIT, not on any identity decision,
# so a bad value here shows up as a calibration this tool refuses to write.
DEFAULT_MIN_COSINE = 0.85       # how confident a match must be to be a target
DEFAULT_MAX_DT = 0.20           # seconds; a "co-temporal" foot-point pairing
DEFAULT_RANSAC_PX = 12.0        # reprojection tolerance, in cam_a pixels
MIN_CORRESPONDENCES = 12        # 4 fits exactly and measures nothing; demand more
MIN_INLIER_FRACTION = 0.55
MIN_TRIANGLES = 3               # below this the validation proves nothing
MIN_TRIANGLE_PASS_RATE = 0.80


def header(text):
    print("\n" + "=" * 76)
    print(text)
    print("=" * 76)


# ------------------------------------------------------------------ observations

def load_observations(store, run_id):
    """-> {(camera, track_id): [(ts, bbox, vector)]}, plus what was skipped.

    Reads vectors because the fit needs appearance to know which foot point in one
    camera corresponds to which in the other.
    """
    out = defaultdict(list)
    skipped = {"no_bbox": 0, "no_ts": 0, "other_run": 0}
    offset = None
    scroll_filter = None
    try:
        from qdrant_client import models as qmodels
        scroll_filter = qmodels.Filter(must=[qmodels.FieldCondition(
            key="run_id", match=qmodels.MatchValue(value=run_id))])
    except Exception:                                               # noqa: BLE001
        scroll_filter = None

    while True:
        try:
            pts, offset = store.client.scroll(
                store.collection, limit=1000, offset=offset,
                scroll_filter=scroll_filter, with_payload=True, with_vectors=True)
        except Exception:                                           # noqa: BLE001
            scroll_filter = None
            pts, offset = store.client.scroll(
                store.collection, limit=1000, offset=offset,
                with_payload=True, with_vectors=True)
        for p in pts:
            pl = p.payload or {}
            if pl.get("run_id") != run_id:
                skipped["other_run"] += 1
                continue
            cam, tid = pl.get("camera"), pl.get("track_id")
            if cam is None or tid is None:
                continue
            bbox, ts = pl.get("bbox"), pl.get("ts")
            if bbox is None:
                skipped["no_bbox"] += 1
                continue
            if ts is None:
                skipped["no_ts"] += 1
                continue
            vec = p.vector
            if isinstance(vec, dict):
                vec = next(iter(vec.values()), None)
            if vec is None:
                continue
            v = np.asarray(vec, dtype=np.float32).ravel()
            n = np.linalg.norm(v)
            if n <= 0:
                continue
            out[(cam, int(tid))].append((float(ts), [float(b) for b in bbox], v / n))
        if offset is None:
            break
    for key in out:
        out[key].sort(key=lambda r: r[0])
    return out, skipped


def inventory_failure(store, run_id):
    """Explain WHY a run has no observations, by inventorying the store.

    Three causes look identical from the outside and need different fixes:
      * wrong run id                       -> the list below has what you want
      * collection emptied (`--reset`)     -> the corpus is gone; capture again
      * collection rebuilt at a new width  -> ditto, and the width says when
    Naming which one it is costs one scroll and saves an afternoon.
    """
    lines = [f"[calib] no usable observations for run {run_id!r}.", ""]
    try:
        info = store.client.get_collection(store.collection)
        dim = getattr(info.config.params.vectors, "size", "?")
        total = store.count()
        lines.append(f"        collection {store.collection!r}: {total} point(s), "
                     f"dim={dim} (this build embeds at {store.dim})")
    except Exception as e:                                      # noqa: BLE001
        return "\n".join(lines + [f"        could not inspect the collection: {e}"])

    runs, with_bbox, with_ts = {}, {}, {}
    offset = None
    try:
        while True:
            pts, offset = store.client.scroll(store.collection, limit=1000,
                                              offset=offset, with_payload=True,
                                              with_vectors=False)
            for p in pts:
                pl = p.payload or {}
                r = pl.get("run_id")
                runs[r] = runs.get(r, 0) + 1
                if pl.get("bbox") is not None:
                    with_bbox[r] = with_bbox.get(r, 0) + 1
                if pl.get("ts") is not None:
                    with_ts[r] = with_ts.get(r, 0) + 1
            if offset is None:
                break
    except Exception as e:                                      # noqa: BLE001
        return "\n".join(lines + [f"        could not scroll the collection: {e}"])

    if not runs:
        lines += [
            "",
            "        The collection is EMPTY. Every stored run is gone -- the usual",
            "        cause is `python main.py --reset`, which wipes the store AND",
            "        THEN runs the pipeline, so it looks like a normal run.",
            "",
            "        A floor frame can only be fitted from a run that is still in",
            "        the store, so this needs a fresh capture with two cameras that",
            "        see the same room (cam_219 + cam_224).",
        ]
        return "\n".join(lines)

    lines += ["", "        runs present (usable = has BOTH bbox and ts, which a "
                  "foot point needs):"]
    for r, n in sorted(runs.items(), key=lambda kv: str(kv[0])):
        usable = min(with_bbox.get(r, 0), with_ts.get(r, 0))
        flag = "" if usable else "   <- NOT usable: no bbox/ts stored"
        lines.append(f"          {str(r):24} {n:7} point(s), {usable} usable{flag}")
    lines += [
        "",
        "        If the run you wanted is not listed, it is no longer in the store.",
        "        Pick one above, or capture a new one. Note a run needs TWO cameras",
        "        that see the SAME room -- a single-camera run cannot be fitted.",
    ]
    return "\n".join(lines)


def prototypes(obs):
    protos = {}
    for key, rows in obs.items():
        m = np.mean([r[2] for r in rows], axis=0)
        n = np.linalg.norm(m)
        if n > 0:
            protos[key] = m / n
    return protos


def temporal_overlap(rows_a, rows_b):
    """Seconds two observation lists overlap in wall-clock time."""
    if not rows_a or not rows_b:
        return 0.0
    return max(0.0, min(rows_a[-1][0], rows_b[-1][0])
               - max(rows_a[0][0], rows_b[0][0]))


def co_temporal_feet(rows_a, rows_b, max_dt):
    """Foot-point pairs seen within `max_dt` of each other. -> [((ua,va),(ub,vb))]."""
    pairs = []
    times_b = [r[0] for r in rows_b]
    for ts, bbox_a, _v in rows_a:
        i = np.searchsorted(times_b, ts)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(rows_b):
                dt = abs(rows_b[j][0] - ts)
                if best is None or dt < best[0]:
                    best = (dt, j)
        if best is None or best[0] > max_dt:
            continue
        fa = foot_point(bbox_a)
        fb = foot_point(rows_b[best[1]][1])
        if fa is not None and fb is not None:
            pairs.append((fa, fb))
    return pairs


# ------------------------------------------------------------------ the fit

# Correspondences are DEDUPLICATED by floor location, on a grid this many pixels
# across in the target camera. Roughly a person's width, so two points in one cell
# are the same place as far as the fit is concerned.
DEDUP_CELL_PX = 60
# At most this many correspondences per cell. A homography needs distinct LOCATIONS;
# repeats of one location add no information and, worse, out-vote everything else.
MAX_PER_CELL = 3
# Distinct occupied cells needed. THIS is the number that decides whether a fit is
# possible -- not the raw correspondence count.
MIN_DISTINCT_LOCATIONS = 8


def deduplicate_by_location(corr, cell_px=DEDUP_CELL_PX, cap=MAX_PER_CELL):
    """Thin correspondences so no single floor location can dominate the fit.

    WHY THIS IS NOT AN OPTIMISATION. Run 20260803_121136 produced 122
    correspondences, of which 106 came from ONE person who never moved: a
    38.6-second tracklet contributing the same floor location over and over. Those
    106 are mutually consistent with each other (they are the same point), so RANSAC
    scored them as a model and rejected everything else -- 8% inliers, and a
    flattering 9.7 px median held-out error that described nothing but the blob.

    A homography is determined by distinct locations. Duplicates of one location
    carry no extra information about the plane and actively bias the fit toward
    whatever is wrong about that one spot -- which is exactly the failure mode of a
    SEATED person, whose every observation is off-plane in the same direction.

    Keeps `cap` per grid cell, preferring the earliest, and returns
    (kept, n_cells).
    """
    seen = {}
    kept = []
    for pair in corr:
        (xa, ya) = pair[0]
        cell = (int(xa // cell_px), int(ya // cell_px))
        n = seen.get(cell, 0)
        if n >= cap:
            continue
        seen[cell] = n + 1
        kept.append(pair)
    return kept, len(seen)


def gather_correspondences(obs, protos, cam_a, cam_b, min_cosine, max_dt):
    """Foot-point correspondences from confident cross-camera matches.

    Reciprocal-best is required on top of the cosine floor, for the reason
    reconcile requires it: on out-of-domain footage many different people land in
    the same 0.8-0.9 band, and an absolute bar alone lets a look-alike contribute
    correspondences from the wrong person's feet.
    """
    keys_a = sorted(k for k in protos if k[0] == cam_a)
    keys_b = sorted(k for k in protos if k[0] == cam_b)
    if not keys_a or not keys_b:
        return [], []

    scores = {(a, b): float(protos[a] @ protos[b]) for a in keys_a for b in keys_b}
    best_for_a = {a: max(keys_b, key=lambda b: scores[(a, b)]) for a in keys_a}
    best_for_b = {b: max(keys_a, key=lambda a: scores[(a, b)]) for b in keys_b}

    used, corr = [], []
    for a in keys_a:
        b = best_for_a[a]
        if best_for_b[b] != a:
            continue
        score = scores[(a, b)]
        if score < min_cosine:
            continue
        overlap = temporal_overlap(obs[a], obs[b])
        if overlap <= 0:
            continue
        feet = co_temporal_feet(obs[a], obs[b], max_dt)
        if not feet:
            continue
        corr.extend(feet)
        used.append((a, b, score, overlap, len(feet)))
    return corr, used


def describe_spread(corr, image_size=None):
    """How the correspondence points are ARRANGED, not how many there are.

    Count is the number everyone looks at and it is the least informative. A
    homography maps a PLANE to a plane, so it needs points spread in two
    dimensions. Sixteen points along one person's six-second walk are nearly
    COLLINEAR, and a homography fitted from collinear points is degenerate --
    `findHomography` still returns a matrix, it is simply meaningless away from that
    line. The symptoms are a low inlier fraction and a huge held-out error, which
    look like "bad matches" and get misdiagnosed as such.

    Returns:
      collinearity : sqrt(smaller / larger principal spread), 0 = a perfect line,
                     1 = an isotropic blob. Below ~0.15 the fit cannot be trusted
                     however many points there are.
      coverage     : fraction of the frame's width/height the points span, which
                     says whether the map is being asked to extrapolate.
    """
    pts = np.asarray([c[0] for c in corr], dtype=np.float64)   # cam_a (dst) side
    out = {"n": len(pts)}
    if len(pts) < 3:
        out["collinearity"] = 0.0
        out["coverage"] = None
        return out
    centred = pts - pts.mean(axis=0)
    # Eigenvalues of the 2x2 covariance = variance along each principal axis.
    eig = np.linalg.eigvalsh(np.cov(centred, rowvar=False))
    lo, hi = float(max(eig.min(), 0.0)), float(max(eig.max(), 1e-12))
    out["collinearity"] = float(np.sqrt(lo / hi))
    span = pts.max(axis=0) - pts.min(axis=0)
    out["span_px"] = [float(span[0]), float(span[1])]
    if image_size:
        out["coverage"] = [float(span[0] / image_size[0]),
                           float(span[1] / image_size[1])]
    else:
        out["coverage"] = None
    return out


def report_spread(spread, n_pairs):
    """Print the arrangement, and say plainly when it cannot support a fit."""
    print(f"  arrangement of the {spread['n']} point(s) in the target frame:")
    if spread.get("span_px"):
        cov = spread.get("coverage")
        extra = (f"  ({cov[0] * 100:.0f}% x {cov[1] * 100:.0f}% of the frame)"
                 if cov else "")
        print(f"     span {spread['span_px'][0]:.0f} x "
              f"{spread['span_px'][1]:.0f} px{extra}")
    print(f"     collinearity {spread['collinearity']:.3f}   "
          f"(0 = all on one line, 1 = spread evenly in both directions)")
    if spread["collinearity"] < 0.15:
        print(f"  !! NEARLY COLLINEAR. A homography maps a PLANE to a plane, so")
        print(f"     points along a single path cannot determine one -- the fit is")
        print(f"     degenerate no matter how many points there are, and any inlier")
        print(f"     count or residual below is meaningless.")
        if n_pairs <= 2:
            print(f"     With only {n_pairs} matched tracklet pair(s), every point")
            print(f"     traces the same short walk. This needs FOOTAGE where people")
            print(f"     stand in many different places, not a longer walk.")
        return False
    return True


def fit_homography(corr, ransac_px):
    """RANSAC cam_b pixels -> cam_a pixels. -> (H, inlier_mask) or (None, None)."""
    if len(corr) < 4:
        return None, None
    src = np.array([[c[1]] for c in corr], dtype=np.float64)   # cam_b
    dst = np.array([[c[0]] for c in corr], dtype=np.float64)   # cam_a
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, float(ransac_px))
    if H is None:
        return None, None
    return H, (None if mask is None else mask.ravel().astype(bool))


def project(H, pt):
    d = H[2, 0] * pt[0] + H[2, 1] * pt[1] + H[2, 2]
    if not np.isfinite(d) or abs(d) < 1e-9:
        return None
    return ((H[0, 0] * pt[0] + H[0, 1] * pt[1] + H[0, 2]) / d,
            (H[1, 0] * pt[0] + H[1, 1] * pt[1] + H[1, 2]) / d)


def heldout_residuals(corr, ransac_px, folds=5):
    """Leave-a-fold-out reprojection error, in cam_a pixels.

    Residuals of the points the fit was computed FROM measure how well it memorised
    them; at 4 points that is exactly zero and means nothing. Holding folds out is
    what makes the error real, and it is the number that becomes
    `position_error_units`.
    """
    n = len(corr)
    if n < MIN_CORRESPONDENCES:
        return np.array([])
    idx = np.arange(n)
    rng = np.random.default_rng(0)          # fixed seed: a calibration must replay
    rng.shuffle(idx)
    errs = []
    for f in range(folds):
        test = set(idx[f::folds].tolist())
        train = [corr[i] for i in range(n) if i not in test]
        if len(train) < 8:
            continue
        H, _ = fit_homography(train, ransac_px)
        if H is None:
            continue
        for i in sorted(test):
            got = project(H, corr[i][1])
            if got is None:
                continue
            errs.append(math.hypot(got[0] - corr[i][0][0], got[1] - corr[i][0][1]))
    return np.array(errs, dtype=np.float64)


# ------------------------------------------------------------------ validation

def find_triangles(obs, protos, cam_a, cam_b):
    """(X in cam_a, Y in cam_b, Z in cam_b) where Y and Z are provably different.

    Y and Z co-occur in cam_b, so they are two people; X can match at most one. This
    needs no labels, which is the whole point -- it is the one check a homography
    fitted from appearance matches cannot flatter itself on.
    """
    keys_a = [k for k in protos if k[0] == cam_a]
    keys_b = [k for k in protos if k[0] == cam_b]
    out = []
    for i, y in enumerate(keys_b):
        for z in keys_b[i + 1:]:
            if temporal_overlap(obs[y], obs[z]) <= 0:
                continue
            for x in keys_a:
                if (temporal_overlap(obs[x], obs[y]) <= 0
                        or temporal_overlap(obs[x], obs[z]) <= 0):
                    continue
                out.append((x, y, z))
    return out


def median_floor_distance(obs, H, key_a, key_b, max_dt):
    """Median distance between two tracklets' co-temporal floor positions."""
    dists = []
    times_b = [r[0] for r in obs[key_b]]
    for ts, bbox_a, _v in obs[key_a]:
        i = np.searchsorted(times_b, ts)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(obs[key_b]):
                dt = abs(times_b[j] - ts)
                if best is None or dt < best[0]:
                    best = (dt, j)
        if best is None or best[0] > max_dt:
            continue
        fa = foot_point(bbox_a)
        fb = foot_point(obs[key_b][best[1]][1])
        if fa is None or fb is None:
            continue
        pb = project(H, fb)
        if pb is None:
            continue
        dists.append(math.hypot(pb[0] - fa[0], pb[1] - fa[1]))
    return float(np.median(dists)) if dists else None


def validate_triangles(obs, protos, H, cam_a, cam_b, max_dt):
    """-> (n_tested, n_passed, examples). A triangle passes when the two arms differ.

    "Differ" deliberately means only that one arm is CLEARLY nearer than the other
    (a factor of 2). Demanding an absolute distance would require the metric scale
    we do not have, and demanding a small ratio would grade the ReID model rather
    than the homography.
    """
    tested = passed = 0
    examples = []
    for x, y, z in find_triangles(obs, protos, cam_a, cam_b):
        dy = median_floor_distance(obs, H, x, y, max_dt)
        dz = median_floor_distance(obs, H, x, z, max_dt)
        if dy is None or dz is None:
            continue
        tested += 1
        lo, hi = sorted((dy, dz))
        ok = hi > 2.0 * max(lo, 1e-6)
        passed += 1 if ok else 0
        if len(examples) < 6:
            examples.append((x, y, z, dy, dz, ok))
    return tested, passed, examples


# ------------------------------------------------------------------ metric refs

def parse_metric_reference(spec):
    """"source:description:floor_units:metres" -> the record's dict form."""
    parts = spec.split(":")
    if len(parts) < 4:
        raise SystemExit(
            f"[calib] --metric-reference {spec!r} is malformed.\n"
            f"        Expected  source:description:floor_units:metres\n"
            f"        e.g.  "
            f"\"floor_plan:corridor width, cam_219 doorframe to far wall:3.42:2.10\"\n"
            f"        floor_units = the distance as THIS frame measures it;\n"
            f"        metres      = its real length, from a trusted source.")
    source, metres, floor_units = parts[0], parts[-1], parts[-2]
    description = ":".join(parts[1:-2])
    try:
        fu, m = float(floor_units), float(metres)
    except ValueError:
        raise SystemExit(f"[calib] --metric-reference {spec!r}: floor_units and "
                         f"metres must both be numbers.")
    return {"source": source, "description": description,
            "floor_units": fu, "metres": m, "verified_on_site": None}


# ------------------------------------------------------------------ entry point

def parse_args():
    p = argparse.ArgumentParser(
        description="Fit the shared floor frame from people's foot points.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("run_id", help="a completed run whose observations are in Qdrant")
    p.add_argument("--cam-a", default="cam_219",
                   help="the camera whose pixel plane becomes the floor frame")
    p.add_argument("--cam-b", default="cam_224", help="the co-visible camera")
    p.add_argument("--group", default=None,
                   help="floor-group name (default: room_<cam_a>_<cam_b>)")
    p.add_argument("--min-cosine", type=float, default=DEFAULT_MIN_COSINE,
                   help="prototype cosine a match needs to become a target")
    p.add_argument("--max-dt", type=float, default=DEFAULT_MAX_DT,
                   help="seconds; how close in time a foot-point pairing must be")
    p.add_argument("--ransac-px", type=float, default=DEFAULT_RANSAC_PX,
                   help="RANSAC reprojection tolerance, in cam_a pixels")
    p.add_argument("--metric-reference", action="append", default=[],
                   metavar="SOURCE:DESC:FLOOR_UNITS:METRES",
                   help="record a trusted metric scale; repeat to cross-check two")
    p.add_argument("--out", default=None, help="record path (default from config)")
    p.add_argument("--dry-run", action="store_true",
                   help="report everything, write nothing")
    p.add_argument("--url", default=None, help="Qdrant URL override")
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml")) as f:
        cfg = yaml.safe_load(f) or {}
    gcfg = cfg.get("geometry") or {}
    out_path = args.out or gcfg.get("calibration_path") or "calibration/floor_frame.json"
    if not os.path.isabs(out_path):
        out_path = os.path.join(root, out_path)

    from database.store import PersonVectorStore
    store_cfg = cfg.get("store") or {}
    store = PersonVectorStore(
        path=store_cfg.get("path", "qdrant_data"),
        url=args.url or os.environ.get("QDRANT_URL") or store_cfg.get("url") or None,
        api_key=os.environ.get("QDRANT_API_KEY") or None)

    header(f"OBSERVATIONS -- run {args.run_id}")
    obs, skipped = load_observations(store, args.run_id)
    if not obs:
        # Never just say "not found". A wrong run id, an emptied collection and a
        # collection rebuilt at a new width are three completely different
        # problems with three different fixes, and they are indistinguishable from
        # the outside. Inventory the store and name which one it is.
        raise SystemExit(inventory_failure(store, args.run_id))
    cams = sorted({k[0] for k in obs})
    print(f"  {sum(len(v) for v in obs.values())} observation(s), "
          f"{len(obs)} tracklet(s), cameras: {', '.join(cams)}")
    if skipped["no_bbox"] or skipped["no_ts"]:
        print(f"  skipped: {skipped['no_bbox']} without bbox, "
              f"{skipped['no_ts']} without ts (both are required to place a foot)")
    for cam in (args.cam_a, args.cam_b):
        if cam not in cams:
            raise SystemExit(
                f"[calib] camera {cam} is not in run {args.run_id} (has: "
                f"{', '.join(cams)}).\n"
                f"        A floor frame needs TWO cameras that see the same room. "
                f"Pick them with --cam-a / --cam-b.")

    protos = prototypes(obs)

    header(f"CORRESPONDENCES -- {args.cam_b} feet -> {args.cam_a} feet")
    corr, used = gather_correspondences(obs, protos, args.cam_a, args.cam_b,
                                       args.min_cosine, args.max_dt)
    print(f"  {len(used)} confident reciprocal-best cross-camera match(es) "
          f"at cosine >= {args.min_cosine:.2f}")
    for a, b, score, overlap, n in sorted(used, key=lambda t: -t[2]):
        print(f"     {a[0]}:{a[1]:<4} <-> {b[0]}:{b[1]:<4}  cosine {score:.3f}  "
              f"overlap {overlap:5.1f}s  -> {n} foot pair(s)")
    print(f"  {len(corr)} raw foot-point correspondence(s)")

    # Thin by location BEFORE counting anything, so a stationary person cannot
    # dominate. Distinct locations, not raw count, is what determines a plane.
    corr, n_locations = deduplicate_by_location(corr)
    print(f"  -> {len(corr)} after de-duplicating by floor location "
          f"({DEDUP_CELL_PX} px cells, max {MAX_PER_CELL} each)")
    print(f"  -> {n_locations} DISTINCT location(s) covered   "
          f"[need >= {MIN_DISTINCT_LOCATIONS}]")
    if n_locations < MIN_DISTINCT_LOCATIONS:
        raise SystemExit(
            f"[calib] only {n_locations} distinct floor location(s) "
            f"(need {MIN_DISTINCT_LOCATIONS}).\n"
            f"\n"
            f"        This is the number that matters, and it is NOT the raw\n"
            f"        correspondence count -- a person standing still for a minute\n"
            f"        produces hundreds of correspondences at ONE location, which\n"
            f"        determines nothing about the plane.\n"
            f"\n"
            f"        THE FIX IS FOOTAGE. One person is enough, and is actually\n"
            f"        ideal (no matching ambiguity). What is needed is COVERAGE:\n"
            f"          * walk to a spot, STAND STILL 3-4 seconds, feet visible\n"
            f"          * move to a clearly DIFFERENT spot; repeat 8-12 times\n"
            f"          * cover the room's extent -- each corner of the overlap,\n"
            f"            the middle, near one camera, near the other\n"
            f"          * NOBODY SEATED: the foot point is the bottom of the box,\n"
            f"            so a chair or desk edge encodes a plane that does not\n"
            f"            exist, and those points are self-consistent enough for\n"
            f"            RANSAC to prefer them\n"
            f"\n"
            f"        Add 30 seconds with two people standing in DIFFERENT places\n"
            f"        for the triangle validation.")
    if len(corr) < MIN_CORRESPONDENCES:
        raise SystemExit(
            f"[calib] {len(corr)} correspondences is too few (need "
            f"{MIN_CORRESPONDENCES}).\n"
            f"        Four points fit a homography EXACTLY, so the error becomes\n"
            f"        unmeasurable -- that is why this refuses rather than fits.\n"
            f"        Try a run with more people crossing both views, or relax\n"
            f"        --min-cosine (which admits less certain matches, so re-read\n"
            f"        the triangle validation carefully if you do).")

    header("THE FIT")
    H, mask = fit_homography(corr, args.ransac_px)
    if H is None:
        raise SystemExit(
            "[calib] findHomography failed. The usual causes are foot points that "
            "are not all on one plane (a mezzanine, stairs) or correspondences from "
            "two different people.")
    inliers = int(mask.sum()) if mask is not None else len(corr)
    frac = inliers / len(corr)
    print(f"  RANSAC: {inliers}/{len(corr)} inliers ({frac:.0%}) at "
          f"{args.ransac_px:.1f} px tolerance")
    residuals = heldout_residuals(corr, args.ransac_px)
    if residuals.size:
        print(f"  held-out reprojection error, in {args.cam_a} pixels:")
        print(f"     median {np.median(residuals):7.1f}   "
              f"p95 {np.percentile(residuals, 95):7.1f}   "
              f"max {residuals.max():7.1f}")
    else:
        print("  held-out error: NOT COMPUTABLE (too few correspondences)")
    if frac < MIN_INLIER_FRACTION or not usable_spread:
        why = (f"only {frac:.0%} of correspondences fit one homography "
               f"(need {MIN_INLIER_FRACTION:.0%})" if frac < MIN_INLIER_FRACTION
               else "the correspondences are nearly collinear")
        lines = [f"[calib] {why}, so this footage cannot produce a usable floor "
                 f"frame yet.", ""]
        # Order the advice by what the DATA says, not by a fixed list. Suggesting
        # "raise --min-cosine" when only one pair matched would delete the only
        # evidence there is -- which is exactly what the first version of this
        # message did.
        if spread["collinearity"] < 0.15:
            lines += [
                "        DIAGNOSIS: degenerate arrangement, not bad matching.",
                f"        collinearity {spread['collinearity']:.3f} means the points",
                "        lie along a line. A homography maps a plane to a plane and",
                "        cannot be recovered from one line, so the residuals above",
                "        describe nothing at all.",
                "",
                "        THE FIX IS FOOTAGE, not a flag. What is needed is people",
                "        STANDING IN MANY DIFFERENT PLACES while both cameras see",
                "        them -- 8-12 distinct spots, corners and middle, near and",
                "        far -- rather than a longer walk along one path.",
            ]
        if len(used) <= 2:
            lines += [
                "",
                f"        Only {len(used)} tracklet pair(s) cleared "
                f"--min-cosine {args.min_cosine:.2f}, so every point comes from one",
                "        person's path. Do NOT raise the gate -- that removes the",
                "        only evidence. Lowering it admits more pairs and more of",
                "        the floor, at the risk of matching the wrong person:",
                f"          python tools/fit_floor_frame.py {args.run_id} "
                f"--min-cosine 0.65",
                "        The triangle validation is what catches a wrong match, so",
                "        read it carefully if you do.",
            ]
        else:
            lines += [
                "",
                "        With several matched pairs, a low inlier fraction points at",
                "        correspondences from two different people, or foot points",
                "        off the floor plane (someone seated, or behind a desk).",
                "        RAISE --min-cosine to keep only the confident matches.",
            ]
        raise SystemExit("\n".join(lines))

    header("VALIDATION -- provably-different people (no labels used)")
    tested, passed, examples = validate_triangles(obs, protos, H, args.cam_a,
                                                 args.cam_b, args.max_dt)
    print(f"  {tested} triangle(s) tested, {passed} passed")
    for x, y, z, dy, dz, ok in examples:
        print(f"     {x[0]}:{x[1]} vs ({y[0]}:{y[1]}, {z[0]}:{z[1]}) -> "
              f"{dy:7.1f} / {dz:7.1f} px   {'PASS' if ok else 'FAIL'}")
    if tested < MIN_TRIANGLES:
        print(f"  !! fewer than {MIN_TRIANGLES} triangles -- this run cannot "
              f"VALIDATE the fit.")
        print(f"     A triangle needs two people co-present in {args.cam_b} while a")
        print(f"     third track runs in {args.cam_a}. Without them the homography")
        print(f"     is unverified, and an unverified homography drives a hard veto.")
        raise SystemExit("[calib] refusing to write an unvalidated calibration.")
    rate = passed / tested
    if rate < MIN_TRIANGLE_PASS_RATE:
        raise SystemExit(
            f"[calib] only {rate:.0%} of triangles passed (need "
            f"{MIN_TRIANGLE_PASS_RATE:.0%}).\n"
            f"        The homography cannot separate people it is PROVEN must be in "
            f"different places, so it is wrong -- writing it would drive a hard veto "
            f"off a bad map. Check for a camera moved mid-run, and for foot points "
            f"off the floor plane.")
    print(f"  PASS ({rate:.0%})")

    header("SPEED CEILING -- measured, not assumed")
    samples = defaultdict(list)
    for key, rows in obs.items():
        if key[0] == args.cam_a:
            for ts, bbox, _v in rows:
                fp = foot_point(bbox)
                if fp:
                    samples[key].append((ts, fp[0], fp[1]))
        elif key[0] == args.cam_b:
            for ts, bbox, _v in rows:
                fp = foot_point(bbox)
                if fp is None:
                    continue
                p = project(H, fp)
                if p:
                    samples[key].append((ts, p[0], p[1]))
    ceiling = observed_speed_ceiling(samples)
    print(format_ceiling_report(ceiling, units=f"{args.cam_a} px"))

    # position_error is the held-out residual, which is what a position may be wrong
    # by. The reachability check SUBTRACTS it from every distance, so a sloppier
    # calibration produces fewer vetoes rather than wrong ones.
    position_error = (float(np.percentile(residuals, 95)) if residuals.size
                      else float(args.ransac_px))

    group = args.group or f"room_{args.cam_a}_{args.cam_b}"
    size_a = image_size_for(obs, args.cam_a)
    size_b = image_size_for(obs, args.cam_b)
    blob = {
        "units": "floor_units",
        "metric_reference": ([parse_metric_reference(s)
                              for s in args.metric_reference] or None),
        "source": (f"tools/fit_floor_frame.py {args.run_id} "
                   f"--cam-a {args.cam_a} --cam-b {args.cam_b}"),
        "groups": {
            group: {
                "cameras": {
                    # cam_a IS the floor frame, so its map is the identity.
                    args.cam_a: {"H": np.eye(3).tolist(), "image_size": size_a,
                                 "n_points": len(corr),
                                 "position_error_units": round(position_error, 4)},
                    args.cam_b: {"H": [[float(v) for v in row] for row in H],
                                 "image_size": size_b,
                                 "n_points": len(corr),
                                 "position_error_units": round(position_error, 4)},
                },
                "position_error_units": round(position_error, 4),
                "speed_ceiling_units_per_sec": (
                    None if ceiling is None
                    else round(float(ceiling["ceiling_units_per_sec"]), 4)),
                "fit": {
                    "run_id": args.run_id,
                    "matches": [[f"{a[0]}:{a[1]}", f"{b[0]}:{b[1]}", round(s, 4)]
                                for a, b, s, _o, _n in used],
                    "n_correspondences": len(corr),
                    "n_distinct_locations": n_locations,
                    "collinearity": round(spread["collinearity"], 4),
                    "ransac_inliers": inliers,
                    "ransac_px": args.ransac_px,
                    "heldout_px_median": (round(float(np.median(residuals)), 3)
                                          if residuals.size else None),
                    "heldout_px_p95": (round(float(np.percentile(residuals, 95)), 3)
                                       if residuals.size else None),
                    "triangles_tested": tested,
                    "triangles_passed": passed,
                    "speed_stats": ceiling,
                },
            }
        },
        # The ONLY place operator-supplied real-world distances between separate
        # floor groups belong. Empty here because cam_206 and cam_213 overlap
        # nothing, so no imagery can relate them -- and using this needs a metric
        # reference, since a real-world distance arrives in metres.
        "group_distances": {},
        "notes": "",
    }

    header("RECORD")
    if args.dry_run:
        print("  --dry-run: nothing written. It would have gone to")
        print(f"    {os.path.relpath(out_path, root)}")
        return 0
    rec = write_calibration(blob, out_path)
    print(f"  -> {os.path.relpath(out_path, root)}")
    for line in rec.summary().splitlines():
        print(f"  {line}")
    if ceiling is None:
        print("\n  !! No speed ceiling was measurable, so the reachability check "
              "stays\n     UNAVAILABLE (fails open) until a richer run is fitted.")
    print("\n  Next:")
    print("    1. set geometry.enabled: true in config.yaml, then capture a run --")
    print("       positions are recorded live and cannot be added afterwards.")
    print("    2. turn on geometry.reconcile.enabled and WATCH the re-rendered video:")
    print("       python tests/calibration/rerender_from_clips.py <run_id> --geometry")
    return 0


def image_size_for(obs, camera):
    """Infer the pixel space these boxes were measured in, from the boxes.

    A homography belongs to the pixel space it was fitted in, and that space is
    whatever the run's frames were, so it has to be recorded. Taking the max box
    extent is a lower bound on the frame, which is enough to detect the case that
    matters -- a later run at a DIFFERENT resolution -- because a pure resize scales
    it proportionally. Rounded up to a standard size when it is close to one.
    """
    boxes = [b for key, rows in obs.items() if key[0] == camera
             for (_ts, b, _v) in rows]
    xs = [b[2] for b in boxes]
    ys = [b[3] for b in boxes]
    if not xs or not ys:
        return None
    w, h = int(math.ceil(max(xs))), int(math.ceil(max(ys)))
    for cw, ch in ((2560, 1440), (1920, 1080), (1280, 720), (3840, 2160)):
        if 0.94 * cw <= w <= cw and 0.94 * ch <= h <= ch:
            return [cw, ch]
    return [w, h]


if __name__ == "__main__":
    sys.exit(main())
