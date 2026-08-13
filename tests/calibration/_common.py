"""
Shared helpers for the calibration harness.

These scripts EXIST TO PRODUCE NUMBERS, not to pass. Their output is what
REMEDIATION_PLAN.md Part H records, and re-running them on new footage is how
that section stays honest.

The one methodological rule that matters, and the one that was got wrong first
time round: a "different person" pair is ONLY valid when the two tracks
CO-OCCUR IN THE SAME FRAME. A person cannot be two simultaneous detections, so
co-occurrence proves distinctness. Comparing tracks that never co-occur silently
mixes in fragments of the SAME person (which is a known defect here), and that
inflates every different-person statistic. `proven_distinct_pairs` enforces this.
"""

from __future__ import annotations

import os
import sys
from itertools import combinations

import cv2
import numpy as np

# ---------------------------------------------------------------- project paths

def project_root() -> str:
    """Repo root, derived from this file's location (tests/calibration/_common.py)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bootstrap() -> str:
    """Put `src` on sys.path the same way main.py does, and cd to the repo root
    so relative model paths (yolo11n.pt, src/reid/weights/...) resolve."""
    root = project_root()
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.chdir(root)
    return root


REID_WEIGHTS = "src/reid/weights/osnet_ain_x1_0.pth"


def _reid_cfg_value(key: str, env: str, default, ignore_env: bool = False):
    """One `reid.*` key as the pipeline would resolve it, with an env override.

    Read from config.yaml rather than hardcoded for the same reason
    detect_weights() is: a model or tap swap must not leave the harness silently
    measuring the OLD feature space and reporting it as shipped behaviour. The
    env override is how a comparison run measures more than one -- e.g.
        CALIB_REID_MODEL=osnet_ibn_x1_0 \
        CALIB_REID_WEIGHTS=src/reid/weights/osnet_ibn_x1_0.pth \
          python tests/calibration/measure_score_separation.py register_file.avi 90 6
    """
    override = os.environ.get(env)
    if override and not ignore_env:
        return override
    try:
        import yaml
        with open(os.path.join(project_root(), "config.yaml")) as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("reid") or {}).get(key) or default
    except Exception:                                          # noqa: BLE001
        return default


def reid_tap() -> str:
    """The feature tap the pipeline would use (#39). CALIB_REID_TAP overrides."""
    return _reid_cfg_value("tap", "CALIB_REID_TAP", "post_relu")


def reid_model(ignore_env: bool = False) -> str:
    """The BACKBONE the pipeline would load (`reid.model`, src/reid/backends.py).
    CALIB_REID_MODEL overrides -- pair it with CALIB_REID_WEIGHTS.

    ignore_env=True returns what config.yaml SHIPS regardless of the override,
    which is how a comparison run can tell "the model I am measuring" from "the
    model the store and thresholds are calibrated for"."""
    return _reid_cfg_value("model", "CALIB_REID_MODEL", "osnet_ain_x1_0",
                           ignore_env=ignore_env)


def reid_weights() -> str:
    """The checkpoint the pipeline would load. CALIB_REID_WEIGHTS overrides."""
    return _reid_cfg_value("weights", "CALIB_REID_WEIGHTS", REID_WEIGHTS)


POSE_WEIGHTS = "yolo11n-pose.pt"


def detect_weights() -> str:
    """The detector the pipeline would actually load.

    Read from config.yaml (`detector.model`) rather than hardcoded, so a model swap
    cannot leave the calibration harness silently measuring the OLD model and
    reporting it as the shipped behaviour. `CALIB_DETECT_WEIGHTS` overrides, which
    is how compare_detector_models.py measures more than one.
    """
    override = os.environ.get("CALIB_DETECT_WEIGHTS")
    if override:
        return override
    try:
        import yaml
        with open(os.path.join(project_root(), "config.yaml")) as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("detector") or {}).get("model") or "yolo11n.pt"
    except Exception:                                          # noqa: BLE001
        return "yolo11n.pt"


# Kept as a module-level name because the existing scripts read it directly. It is
# resolved at import time, AFTER bootstrap() has chdir'd to the repo root in the
# scripts that call it first -- project_root() is path-derived, so it does not care.
DETECT_WEIGHTS = detect_weights()


def load_config() -> dict:
    """config.yaml as a dict, or {} if it cannot be read."""
    try:
        import yaml
        with open(os.path.join(project_root(), "config.yaml")) as f:
            return yaml.safe_load(f) or {}
    except Exception:                                          # noqa: BLE001
        return {}


# ------------------------------------------------- argv helpers (shared)
#
# `_arg` used to be copy-pasted into each script as
#     if flag in sys.argv: return sys.argv[sys.argv.index(flag) + 1]
# which has two live bugs: a BOOLEAN flag written last on the command line raises
# IndexError instead of being honoured, and a boolean flag followed by another
# flag returns THAT flag as its value -- truthy, so it happens to work, while
# silently consuming nothing. Both are the kind of thing that costs a validation
# cycle rather than failing loudly, so they live here once, fixed.

def arg(flag: str, default=None):
    """Value after `flag`, or `default`. Never raises on a trailing flag."""
    argv = sys.argv
    if flag in argv:
        i = argv.index(flag) + 1
        if i < len(argv) and not argv[i].startswith("--"):
            return argv[i]
        raise SystemExit(f"[calib] {flag} expects a value "
                         f"(got {'nothing' if i >= len(argv) else argv[i]!r})")
    return default


def flag(name: str) -> bool:
    """True when the boolean flag `name` is present. Takes no value."""
    return name in sys.argv


# Every flag the reconcile tools understand. An argv token starting with `--` that
# is NOT here is a hard error, because the alternative is what already happened:
# `--same-rounds` was passed to a deployment whose code predated the flag, argv
# ignored it in silence, and two "different" sweeps came back byte-identical. A
# measurement that quietly measures the wrong thing is worse than a crash.
KNOWN_FLAGS = {
    "--cross", "--same", "--same-global", "--min-obs", "--scoring", "--top-frac",
    "--no-reciprocal", "--same-reciprocal", "--no-same-reciprocal",
    "--same-rounds", "--no-same-rounds", "--no-covisibility", "--covis-tolerance",
    "--geometry", "--no-geometry", "--geometry-safety",
}


def validate_flags(extra=()):
    """Refuse unknown `--flags`. Catches typos AND a stale deployment."""
    allowed = KNOWN_FLAGS | set(extra)
    unknown = [t for t in sys.argv[1:] if t.startswith("--") and t not in allowed]
    if unknown:
        raise SystemExit(
            f"[calib] unknown flag(s) {unknown}.\n"
            f"        Known: {', '.join(sorted(allowed))}\n"
            f"        If you expected one of these to exist, this deployment is "
            f"OLDER than the flag -- deploy the code and re-run. Silently ignoring "
            f"it would hand you a measurement of the wrong thing.")


def reconcile_settings(cfg=None, log=print, owns=(), extra_flags=()):
    """Reconcile kwargs AS PRODUCTION WOULD RUN THEM, plus argv overrides.

    `owns` names flags the CALLER parses itself, so they are left alone here. The
    sweep and the re-render take comma-separated LISTS for `--cross` and
    `--scoring` (they compare settings), which cannot collapse into one value.

    WHY EVERY OFFLINE TOOL MUST GO THROUGH THIS. The sweep and the re-render used
    to build their kwargs by hand and pass neither `covisibility` nor
    `same_camera_reciprocal_best`, both of which default OFF in
    reconcile_tracklets and are ON in config.yaml. So the only cheap feedback loop
    this project has was measuring a clustering algorithm that does not ship --
    with the cross-camera simultaneity veto disabled, the exact guard that
    dominates the production candidate space (1961 exclusions in run
    20260731_060425). See REMEDIATION_PLAN.md Part M.

    Overrides, all optional:
      --cross F              cross-camera bar (a single value; the sweep parses
                             its own comma list before calling this)
      --same-global F        global same-camera bar
      --same cam=F,cam=F     per-camera same-camera bars (REPLACES config's)
      --min-obs N            min_tracklet_observations
      --scoring MODE         prototype | max_exemplar | consensus
      --top-frac F           consensus_top_frac
      --no-reciprocal        turn Phase 2 mutual-best OFF
      --same-reciprocal      turn Phase 1 mutual-best ON
      --no-same-reciprocal   turn Phase 1 mutual-best OFF
      --same-rounds          iterate Phase 1 until it merges nothing (M.9.11)
      --no-same-rounds       single Phase 1 pass (the old behaviour)
      --no-covisibility      disable the cross-camera simultaneity veto
      --covis-tolerance S    override every numeric veto tolerance with S seconds
                             (pairs declared `covisible` stay covisible)
      --geometry             turn the geometric reachability veto ON for this run
      --no-geometry          turn it OFF
      --geometry-safety F    multiply the MEASURED speed ceiling by F to get the
                             veto line. Raise it to be more permissive; a wrong
                             veto cannot be undone, a missed one costs nothing new

    NOTE `--geometry` toggles POLICY only. It cannot conjure geometry into a run
    that was captured without it -- positions are recorded live and only there
    (geometry/__init__.py invariant 1). On such a run reconcile says so and
    proceeds on appearance alone.
    """
    from identity.reconcile import resolve_reconcile_kwargs

    validate_flags(extra_flags)
    kw = resolve_reconcile_kwargs(cfg if cfg is not None else load_config(), log=log)

    def mine(f):
        """The flag's value, unless the caller owns that flag."""
        return None if f in owns else arg(f)

    if mine("--cross") is not None:
        kw["threshold"] = float(mine("--cross"))
    if mine("--same-global") is not None:
        kw["same_camera_threshold"] = float(mine("--same-global"))
    if mine("--same") is not None:
        # MERGE over config.yaml's per_camera map, never REPLACE it. Replacing meant
        # `--same "cam_219=0.55"` silently dropped every OTHER camera's override back
        # to the global bar: on 20260804_120409 that pushed cam_224 from its
        # configured 0.80 to 0.90 without a word, so a run intended to move ONE bar
        # moved three, and a comparison against the unflagged baseline was measuring
        # two axes at once in opposite directions. Exactly the class of defect this
        # resolver's docstring exists to prevent, one level in.
        overrides = parse_same(mine("--same"))
        merged = dict(kw.get("same_camera_thresholds") or {})
        merged.update(overrides)
        kw["same_camera_thresholds"] = merged
        untouched = {c: v for c, v in merged.items() if c not in overrides}
        if untouched:
            log(f"  [settings] --same overrode {sorted(overrides)}; "
                f"config's other per-camera bars are KEPT: "
                + ", ".join(f"{c}={v:.2f}" for c, v in sorted(untouched.items())))
    if mine("--min-obs") is not None:
        kw["min_tracklet_observations"] = int(mine("--min-obs"))
    if mine("--scoring") is not None:
        kw["scoring"] = mine("--scoring")
    if mine("--top-frac") is not None:
        kw["consensus_top_frac"] = float(mine("--top-frac"))
    if flag("--no-reciprocal"):
        kw["require_reciprocal_best"] = False
    if flag("--same-reciprocal"):
        kw["same_camera_reciprocal_best"] = True
    if flag("--no-same-reciprocal"):
        kw["same_camera_reciprocal_best"] = False
    if flag("--same-rounds"):
        kw["same_camera_rounds"] = True
    if flag("--no-same-rounds"):
        kw["same_camera_rounds"] = False
    if flag("--no-covisibility"):
        kw["covisibility"] = (False, {})
    # Override every NUMERIC tolerance, leaving `covisible` pairs covisible. The
    # veto's own labelled evidence separates cleanly by magnitude -- wrong vetoes
    # at 1.2-1.6 s, right ones at 3.0-10.5 s (M.9.25) -- so this axis needs to be
    # sweepable, and `--no-covisibility` (all or nothing) could not express it.
    if mine("--covis-tolerance") is not None:
        tol = float(mine("--covis-tolerance"))
        enabled, pairs = kw["covisibility"]
        kw["covisibility"] = (enabled,
                              {k: (None if v is None else tol)
                               for k, v in pairs.items()})
    # Geometry POLICY overrides. The positions themselves were recorded during the
    # run and are not overridable from here -- by design.
    geo = dict(kw.get("geometry") or {})
    if flag("--geometry"):
        geo["enabled"] = True
    if flag("--no-geometry"):
        geo["enabled"] = False
    if mine("--geometry-safety") is not None:
        geo["safety_factor"] = float(mine("--geometry-safety"))
    kw["geometry"] = geo
    return kw


def parse_same(text):
    """"cam_213=0.80,cam_224=0.90" -> {'cam_213': 0.8, 'cam_224': 0.9}."""
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"[calib] --same expects cam=VALUE, got {part!r}")
        cam, _, value = part.partition("=")
        out[cam.strip()] = float(value)
    return out


def pick_video(preferred: str | None = None) -> str:
    """Resolve a clip to measure on.

    Defaults to register_file.avi: it is 2560x1440 (production resolution) and
    decodes with ZERO H.265 reference errors, unlike test_file.avi (294 broken
    frames) and test_v2.avi (207). Measuring feature quality on a clip with
    corrupted frames measures the corruption.
    """
    if preferred and os.path.exists(preferred):
        return preferred
    for cand in ("register_file.avi", "test_v2.avi", "test_file.avi",
                 "recorded_1_2.mp4"):
        if os.path.exists(cand):
            if preferred:
                print(f"[calib] {preferred!r} not found; using {cand!r}")
            return cand
    raise SystemExit(
        "[calib] No footage found in the repo root. Pass a path as the first "
        "argument, e.g.  python tests/calibration/<script>.py myclip.mp4")


# ---------------------------------------------------------------- frame loading

def sample_frames(video: str, count: int, stride: int) -> list[np.ndarray]:
    """Every `stride`-th frame, up to `count`. Use for appearance statistics,
    where temporal continuity does not matter."""
    cap = cv2.VideoCapture(video)
    frames, i = [], 0
    while len(frames) < count:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(f)
        i += 1
    cap.release()
    if not frames:
        raise SystemExit(f"[calib] could not decode any frames from {video!r}")
    return frames


def consecutive_frames(video: str, count: int) -> list[np.ndarray]:
    """The first `count` frames, unskipped. Use for anything involving TRACKING
    -- ByteTrack needs real frame-to-frame continuity, so a strided sample
    measures a different (and much harder) problem than the pipeline faces."""
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < count:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"[calib] could not decode any frames from {video!r}")
    return frames


# ---------------------------------------------------------------- box geometry

def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def containment(a, b) -> float:
    """Intersection over the area of the SMALLER box. Catches the
    torso-inside-full-body case, whose IoU is modest but which is still one
    person wearing two boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def det_box(d) -> tuple:
    return (d.x1, d.y1, d.x2, d.y2)


# ------------------------------------------------- embedding collection helpers

MIN_CROP_H, MIN_CROP_W = 64, 24     # mirrors reid.quality.min_height / min_width


def collect_track_embeddings(frames, detector, embed_fn, min_h=MIN_CROP_H,
                             min_w=MIN_CROP_W):
    """Run detect+track over `frames`, embed every usable crop, and return

        by_track : {track_id: [embedding, ...]}   in frame order
        cooccur  : {(tid_a, tid_b), ...}          pairs seen in the SAME frame
        per_frame: [{track_id: embedding}, ...]   one dict per frame

    `embed_fn(list_of_bgr_crops) -> (N, D) array` so callers can swap the
    feature tap without touching this function.
    """
    from detector import crop_person

    by_track: dict[int, list] = {}
    per_frame: list[dict] = []
    cooccur: set[tuple] = set()

    for frame in frames:
        dets = [d for d in detector.track(frame) if d.track_id is not None]
        crops, tids = [], []
        for d in dets:
            crop = crop_person(frame, d)
            if crop is None or crop.size == 0:
                continue
            h, w = crop.shape[:2]
            if h < min_h or w < min_w:
                continue
            crops.append(crop)
            tids.append(d.track_id)
        if not crops:
            per_frame.append({})
            continue
        embs = embed_fn(crops)
        this_frame = {}
        for tid, e in zip(tids, embs):
            by_track.setdefault(tid, []).append(e)
            this_frame[tid] = e
        per_frame.append(this_frame)
        for a, b in combinations(sorted(set(tids)), 2):
            cooccur.add((a, b))

    return by_track, cooccur, per_frame


def proven_distinct_pairs(by_track, cooccur, min_obs=6):
    """Track pairs that CO-OCCUR in at least one frame and both have enough
    observations. Co-occurrence is what proves they are different people.

    Also returns the pairs it had to EXCLUDE, because those are exactly the
    pairs that may be one fragmented person -- worth printing so the reader can
    see how much of the candidate space was discarded and why.
    """
    usable = {t: v for t, v in by_track.items() if len(v) >= min_obs}
    pairs = [(a, b) for (a, b) in cooccur if a in usable and b in usable]
    excluded = [p for p in combinations(sorted(usable), 2) if p not in cooccur]
    return usable, pairs, excluded


# ---------------------------------------------------------------- reporting

def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def describe(arr, label: str, width: int = 34) -> str:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return f"  {label:<{width}} (no samples)"
    return (f"  {label:<{width}} n={a.size:<6d} mean={a.mean():.3f} "
            f"p5={np.percentile(a, 5):.3f} median={np.median(a):.3f} "
            f"p95={np.percentile(a, 95):.3f} min={a.min():.3f} max={a.max():.3f}")


def operating_points(same, other, thresholds=(0.50, 0.60, 0.70, 0.80, 0.85, 0.90)):
    """correct-accept% / wrong-accept% at each threshold."""
    same, other = np.asarray(same), np.asarray(other)
    out = []
    for t in thresholds:
        tp = 100.0 * (same >= t).mean() if same.size else float("nan")
        fp = 100.0 * (other >= t).mean() if other.size else float("nan")
        out.append((t, tp, fp))
    return out


def print_operating_points(same, other, indent="    "):
    print(f"{indent}{'thr':>5}{'correct%':>10}{'WRONG%':>9}")
    for t, tp, fp in operating_points(same, other):
        print(f"{indent}{t:>5.2f}{tp:>10.1f}{fp:>9.1f}")


def margin(same, other) -> float:
    """Worst-case separation: same-person 5th percentile minus different-person
    95th percentile. Positive means a threshold exists that mostly works."""
    same, other = np.asarray(same), np.asarray(other)
    if same.size == 0 or other.size == 0:
        return float("nan")
    return float(np.percentile(same, 5) - np.percentile(other, 95))


def header(title: str, width: int = 76):
    print("\n" + "=" * width + f"\n{title}\n" + "=" * width)


def footnote_sample_size(usable, pairs, excluded):
    print(f"\n  SAMPLE: {len(usable)} tracks, {len(pairs)} proven-distinct pairs.")
    if excluded:
        print(f"  EXCLUDED {len(excluded)} never-co-visible pair(s) {sorted(excluded)}")
        print("  -- those may be ONE person fragmented, so including them would")
        print("     inflate the different-person statistics.")
    print("  Treat small samples as hypotheses. See REMEDIATION_PLAN.md Part H.")
