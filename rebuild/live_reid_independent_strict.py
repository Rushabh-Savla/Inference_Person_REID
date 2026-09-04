from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import cv2

from rebuild.live_reid_independent import (
    CameraWorker,
    IdentityEngine,
    Match,
    Profile,
    TrackState,
    build_floor,
    load_yaml,
)
from rebuild.live_reid_independent import MODEL_MIN, MODELS
from reid.extractor import ReIDExtractor
from reid.nvidia_swin import NVIDIASwinReIDExtractor
from reid.solider_reid import SOLIDERReIDExtractor


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class LockedExtractor:
    """Serialize access to a shared CUDA model while keeping extract_batch API."""

    def __init__(self, model, lock):
        self.model = model
        self.lock = lock

    def extract_batch(self, crops):
        with self.lock:
            return self.model.extract_batch(crops)

    def describe(self):
        return self.model.describe()


class StrictIdentityEngine(IdentityEngine):
    """Online identity engine that verifies the current GID on every feature sample.

    The base independent engine protects initial association. This stricter layer
    adds the missing production rule: an already assigned track is re-checked against
    its own GID every time new features are extracted. A replacement GID must win by
    a real multimodel feature margin for consecutive observations; a single bad crop
    never flips an identity. Persistent mismatch eventually returns the track to
    PENDING instead of continuing to claim an unsupported GID.
    """

    def __init__(self, cfg, debug_path):
        super().__init__(cfg, debug_path)
        self.switch_margin = float(cfg.get("switch_margin", 0.05))
        self.verify_threshold = float(cfg.get("verify_threshold", 0.50))
        self.verify_attr = float(cfg.get("verify_attribute_threshold", 0.34))
        self.verify_grace = max(1, int(cfg.get("verify_grace_samples", 3)))

    def _verify(self, match: Match) -> bool:
        attrs = float((
            match.upper_colour
            + match.lower_colour
            + match.head
            + match.pattern
        ) / 4.0)
        return (
            match.deep_support >= 2
            and match.deep_top2 >= self.verify_threshold
            and (attrs >= self.verify_attr or match.deep_top2 >= 0.68)
        )

    def observe(self, obs, state):  # noqa: C901
        with self.lock:
            self.stats["feature_observations"] += 1
            state.last_seen = obs.timestamp
            self.active[state.key] = state
            if not hasattr(state, "bad_count"):
                state.bad_count = 0
                state.switch_gid = None
                state.switch_count = 0

            # ---------------------------------------------------------------
            # Already assigned: VERIFY every new feature sample.
            # ---------------------------------------------------------------
            if state.gid is not None:
                current = self.profiles.get(state.gid)
                current_match = self.score(obs, current) if current is not None else None
                if current_match is not None and self._verify(current_match):
                    state.bad_count = 0
                    state.switch_gid = None
                    state.switch_count = 0
                    current.add(obs, self.bank_size, self.novelty)
                    self._debug(obs, [current_match], current_match, state)
                    return state.gid

                # Current GID does not match strongly enough. Search every OTHER
                # profile, but never switch on one noisy frame.
                alternatives = []
                for profile in self.profiles.values():
                    if profile.gid == state.gid:
                        continue
                    if self._same_camera_conflict(obs, profile):
                        continue
                    alternatives.append(self.score(obs, profile))
                alternatives.sort(key=lambda x: x.score, reverse=True)
                second = alternatives[1].score if len(alternatives) > 1 else 0.0
                replacement = alternatives[0] if alternatives else None
                if replacement is not None and self.acceptable(replacement, second):
                    old_score = current_match.score if current_match is not None else 0.0
                    if replacement.score >= old_score + self.switch_margin:
                        if getattr(state, "switch_gid", None) == replacement.gid:
                            state.switch_count += 1
                        else:
                            state.switch_gid = replacement.gid
                            state.switch_count = 1
                        if state.switch_count >= self.confirm_samples:
                            old_gid = state.gid
                            state.gid = replacement.gid
                            state.bad_count = 0
                            state.switch_gid = None
                            state.switch_count = 0
                            self.profiles[state.gid].add(obs, self.bank_size, self.novelty)
                            self.stats["existing_gid_assignments"] += 1
                            if replacement.same_camera:
                                self.stats["same_camera_repairs"] += 1
                            else:
                                self.stats["cross_camera_matches"] += 1
                            self._debug(obs, alternatives, replacement, state)
                            return state.gid
                else:
                    state.switch_gid = None
                    state.switch_count = 0

                # Do not immediately discard a known GID because one view is noisy.
                # Give it a short verification grace period; after that the track is
                # genuinely unsupported and becomes unassigned rather than lying.
                state.bad_count += 1
                self.stats["rejected_candidates"] += 1
                if state.bad_count < self.verify_grace:
                    self._debug(obs, [m for m in [current_match] if m], None, state)
                    return state.gid
                state.gid = None
                state.bad_count = 0
                state.pending_gid = None
                state.pending_count = 0
                self._debug(obs, [m for m in [current_match] if m], None, state)
                # Continue below, allowing a clean re-identification from the same
                # observation rather than waiting for the next sampling interval.

            # ---------------------------------------------------------------
            # Unassigned track: genuine multimodel ReID association.
            # ---------------------------------------------------------------
            ranked = []
            for profile in self.profiles.values():
                if self._same_camera_conflict(obs, profile):
                    continue
                ranked.append(self.score(obs, profile))
            ranked.sort(key=lambda x: x.score, reverse=True)
            second = ranked[1].score if len(ranked) > 1 else 0.0
            accepted = ranked[0] if ranked and self.acceptable(ranked[0], second) else None

            if accepted is not None:
                state.gid = accepted.gid
                state.pending_gid = None
                state.pending_count = 0
                state.bad_count = 0
                self.profiles[accepted.gid].add(obs, self.bank_size, self.novelty)
                self.stats["existing_gid_assignments"] += 1
                if accepted.same_camera:
                    self.stats["same_camera_repairs"] += 1
                else:
                    self.stats["cross_camera_matches"] += 1
                self._debug(obs, ranked, accepted, state)
                return state.gid

            if ranked:
                self.stats["rejected_candidates"] += 1
                candidate = ranked[0].gid
                if state.pending_gid == candidate:
                    state.pending_count += 1
                else:
                    state.pending_gid = candidate
                    state.pending_count = 1
                if state.pending_count < self.confirm_samples:
                    self._debug(obs, ranked, None, state)
                    return None

            # Novel identity: repeated, non-overlapping, feature-rich observations
            # that do not satisfy ANY existing profile.
            gid = self._new_gid()
            state.gid = gid
            state.pending_gid = None
            state.pending_count = 0
            profile = Profile(gid)
            profile.add(obs, self.bank_size, self.novelty)
            self.profiles[gid] = profile
            self.stats["new_gid_assignments"] += 1
            self._debug(obs, ranked, None, state)
            return gid


def parse_camera(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("camera must be NAME=RTSP_URL")
    name, source = value.split("=", 1)
    if not name or not source:
        raise argparse.ArgumentTypeError("camera must be NAME=RTSP_URL")
    return name, source


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict independent multimodel live person ReID")
    parser.add_argument("--config", default="rebuild/config_live_independent_strict.yaml")
    parser.add_argument("--camera", action="append", required=True, type=parse_camera)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(cfg.get("output_dir", "rebuild_outputs_live_independent"))
    run_dir = output_root / f"strict_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    engine = StrictIdentityEngine(cfg.get("identity", {}) or {}, log_dir / "match_debug.jsonl")
    floor = build_floor(cfg, run_id)
    models = cfg.get("models", {}) or {}
    device = models.get("device", "cuda")

    resnet = ReIDExtractor(
        weights=models["resnet_weights"],
        device=device,
        tap=models.get("resnet_tap", "post_relu"),
        max_batch=int(models.get("resnet_batch", 32)),
        model=models.get("resnet_model"),
    )
    swin = NVIDIASwinReIDExtractor(
        models["swin_weights"], device=device, max_batch=int(models.get("swin_batch", 16))
    )
    solider = SOLIDERReIDExtractor(
        models["solider_weights"], device=device, max_batch=int(models.get("solider_batch", 16))
    )
    model_lock = threading.Lock()
    engine_models = {
        "resnet": LockedExtractor(resnet, model_lock),
        "swin": LockedExtractor(swin, model_lock),
        "solider": LockedExtractor(solider, model_lock),
    }

    print("=" * 78)
    print("       STRICT INDEPENDENT MULTIMODEL LIVE PERSON RE-ID")
    print("=" * 78)
    print(f"run:        {run_id}")
    print(f"cameras:    {', '.join(name for name, _ in args.camera)}")
    print("models:     NVIDIA ResNet + NVIDIA Swin + SOLIDER")
    print("views:      full + light + upper + torso + lower")
    print("attributes: shirt colour + bottom colour + cloth pattern + headwear/hair appearance")
    print("geometry:   person bbox + optional calibrated shared floor geometry")
    print("sampling:   every N frames; per-track extraction blocked while overlapping")
    print("identity:   every feature sample re-verifies the current GID")
    print("cross-cam:  minimum 2/3 deep-model agreement + attribute gate + margin")
    print("dependency: independent of seif744/Inference_PersonReid")
    print("=" * 78)
    print(f"[models] ResNet:  {resnet.describe()}")
    print(f"[models] Swin:    {swin.describe()}")
    print(f"[models] SOLIDER: {solider.describe()}")

    stop_event = threading.Event()
    latest: Dict[str, object] = {}
    latest_lock = threading.Lock()
    workers = []
    for name, source in args.camera:
        worker_cfg = {
            **cfg.get("capture", {}),
            "detector": cfg["detector"],
            "fragment_gap_sec": (cfg.get("tracking", {}) or {}).get("fragment_gap_sec", 2.0),
        }
        worker = CameraWorker(
            name,
            source,
            worker_cfg,
            engine,
            stop_event,
            latest,
            latest_lock,
            run_dir,
            log_dir / f"{name}.jsonl",
            floor,
            args.show,
        )
        worker.engine_models = engine_models
        workers.append(worker)

    def stop(_sig=None, _frame=None):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    for worker in workers:
        worker.start()

    try:
        while any(worker.is_alive() for worker in workers):
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                stop_event.set()
                break
            if args.show:
                with latest_lock:
                    frames = {name: frame.copy() for name, frame in latest.items()}
                for name, frame in frames.items():
                    cv2.imshow(name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    stop_event.set()
                    break
            else:
                time.sleep(0.2)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=10.0)
        if args.show:
            cv2.destroyAllWindows()
        engine.close()

    print("\n" + "=" * 78)
    print("                     STRICT LIVE RUN COMPLETE")
    print("=" * 78)
    for worker in workers:
        print(
            f"{worker.name_id}: frames={worker.stats['frames']} "
            f"tracks={worker.stats['tracks']} features={worker.stats['features']} "
            f"overlap_skips={worker.stats['overlap_skips']}"
        )
    for name in (
        "feature_observations",
        "existing_gid_assignments",
        "new_gid_assignments",
        "same_camera_repairs",
        "cross_camera_matches",
        "rejected_candidates",
    ):
        print(f"{name:24s}: {engine.stats[name]}")
    print("FINAL GLOBAL IDS:")
    for gid, profile in sorted(engine.profiles.items(), key=lambda item: int(item[0][1:])):
        print(
            f"  {gid}: cameras={','.join(sorted(profile.cameras))} "
            f"tracks={len(profile.tracks)} observations={profile.observations}"
        )
    print(f"outputs: {run_dir}")
    print(f"debug:   {log_dir / 'match_debug.jsonl'}")
    errors = [f"{w.name_id}: {w.error}" for w in workers if w.error is not None]
    if errors:
        print("ERRORS:")
        for value in errors:
            print(f"  {value}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
