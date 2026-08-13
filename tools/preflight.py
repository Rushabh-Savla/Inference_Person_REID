"""
Is this deployment actually configured the way you think it is?

    python tools/preflight.py                 # check everything, load no models
    python tools/preflight.py --load-model    # also LOAD the ReID net and measure it
    python tools/preflight.py --fix-store     # rebuild a wrong-width Qdrant collection

============================ WHY THIS EXISTS =================================

Every failure this checks for is SILENT. Nothing crashes, nothing logs an error, and
every number downstream still looks like a number:

  * a Qdrant collection left at the PREVIOUS backbone's width. The store only
    *warns*, and `IdentityStage` swallows the resulting ValueError -- so a run
    against a stale collection quietly persists NOTHING and the offline reconcile
    then has nothing to reconcile. This is the single most likely way a first run
    after a backbone switch goes wrong.
  * `reid.model` and `reid.weights` naming different architectures. The backend
    raises on this one, so it is loud -- but only once the model is loaded, which
    is minutes into a run with the cameras already going.
  * thresholds inherited from a feature space that is no longer running. Cannot be
    verified automatically; reported with the evidence so the mismatch is visible.
  * output video disabled, or reconcile off, so a run produces no watchable result
    and the reason is three config blocks away.

It loads no models by default, so it is a few seconds and safe to run anywhere.
`--load-model` additionally builds the net and reports its MEASURED width and input
size rather than what config claims -- worth doing once per machine.

Exit code is 0 only when nothing would silently produce garbage.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import yaml                                                         # noqa: E402

OK, WARN, FAIL = "ok", "warn", "fail"
MARK = {OK: "  [ ok ]", WARN: "  [warn]", FAIL: "  [FAIL]"}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, state, what, detail=""):
        self.rows.append((state, what, detail))
        print(f"{MARK[state]} {what}")
        for line in (detail.splitlines() if detail else []):
            print(f"         {line}")
        return state

    def section(self, title):
        print("\n" + "=" * 76)
        print(title)
        print("=" * 76)

    @property
    def failed(self):
        return [r for r in self.rows if r[0] == FAIL]

    @property
    def warned(self):
        return [r for r in self.rows if r[0] == WARN]


def human_mb(path):
    try:
        return f"{os.path.getsize(path) / 1e6:.0f} MB"
    except OSError:
        return "?"


def check_reid(cfg, rep, load_model):
    rep.section("1. ReID MODEL")
    reid = cfg.get("reid") or {}
    name = reid.get("model")
    weights = reid.get("weights")
    tap = reid.get("tap")

    from reid.backends import BACKENDS, DEFAULT_BACKEND

    if not name:
        rep.add(FAIL, "reid.model is not set",
                f"It would fall back to {DEFAULT_BACKEND!r}, which is NOT what\n"
                f"ships. Set reid.model explicitly.")
        return None
    if name not in BACKENDS:
        rep.add(FAIL, f"reid.model {name!r} is not a registered backend",
                f"registered: {sorted(BACKENDS)}")
        return None
    rep.add(OK, f"reid.model = {name}")

    if not weights or not os.path.exists(weights):
        rep.add(FAIL, f"reid.weights missing: {weights!r}",
                "Fetch it per machine -- it is gitignored:\n"
                "  curl -L -o src/reid/weights/msmt_sbs_R101-ibn.pth \\\n"
                "    https://github.com/JDAI-CV/fast-reid/releases/download/"
                "v0.1.1/msmt_sbs_R101-ibn.pth")
        return None
    rep.add(OK, f"reid.weights present ({human_mb(weights)})", weights)

    # A FastReID backend REQUIRES tap n/a -- its head returns the post-bnneck
    # feature unconditionally, so any tap request is a misunderstanding the backend
    # raises on. Catching it here is seconds instead of minutes into a run.
    if name.startswith("fastreid") and str(tap or "").strip().lower() not in ("n/a", "na", ""):
        rep.add(FAIL, f"reid.tap = {tap!r} is invalid for a FastReID backend",
                "FastReID's EmbeddingHead returns the post-bnneck feature\n"
                "unconditionally at eval; there is no tap to choose. Set: tap: n/a")
    else:
        rep.add(OK, f"reid.tap = {tap!r}")

    # Cross-check the name against the checkpoint filename. Not authoritative --
    # --load-model is -- but it catches the common copy-paste error for free.
    base = os.path.basename(weights).lower()
    if name.startswith("fastreid") and "r101" in name.lower() and "r50" in base:
        rep.add(FAIL, "reid.model says R101 but the checkpoint filename says R50")
    elif name.startswith("osnet") and "osnet" not in base:
        rep.add(WARN, "reid.model is an OSNet variant but the checkpoint filename "
                      "does not mention osnet")

    dim = input_size = None
    # `reid.device` applies ONLY to the file-batch path; the live pipeline reads
    # `live.run.device` and never consults this one. Reporting both stops the
    # familiar "I set reid.device to cuda and it still ran on CPU".
    file_dev = reid.get("device")
    live_dev = (((cfg.get("live") or {}).get("run") or {}).get("device"))
    rep.add(OK, f"device: reid.device={file_dev!r} (file-batch path only), "
                f"live.run.device={live_dev!r} (used by --mode live)")

    if load_model:
        try:
            from reid.extractor import ReIDExtractor
            # "auto" is NOT a torch device string -- torch raises on it. None is
            # how the extractor is told to auto-pick, which is the same intent.
            dev = None if str(file_dev or "").strip().lower() in ("", "auto") \
                else file_dev
            ex = ReIDExtractor(weights=weights, device=dev, model=name, tap=tap)
            dim, input_size = ex.embedding_dim, ex.input_size
            rep.add(OK, "model LOADS and reports", ex.describe())
        except Exception as e:                                       # noqa: BLE001
            rep.add(FAIL, f"model failed to load: {type(e).__name__}: {e}")
    else:
        # Read the width off the class without instantiating the net.
        cls, preset = BACKENDS[name]
        dim = getattr(cls, "EMBEDDING_DIM", None)
        for attr in ("SBS_EMBEDDING_DIM", "OSNET_EMBEDDING_DIM"):
            dim = dim or getattr(cls, attr, None)
        rep.add(WARN, "model not loaded (pass --load-model to measure it)",
                "Without loading, the width below is inferred from config, not "
                "measured.")
        dim = 2048 if name.startswith("fastreid") else 512
        input_size = (384, 128) if name.startswith("fastreid") else (256, 128)
        rep.add(OK, f"expected embedding width {dim}-d at "
                    f"{input_size[0]}x{input_size[1]}")
    return dim


def check_store(cfg, rep, model_dim, fix):
    rep.section("2. QDRANT GALLERY")
    store_cfg = cfg.get("store") or {}
    if not store_cfg.get("enabled", False):
        rep.add(FAIL, "store.enabled is false",
                "No observations are persisted, so the OFFLINE RECONCILE cannot "
                "run\nand output videos carry provisional per-camera ids only.")
        return
    url = os.environ.get("QDRANT_URL") or store_cfg.get("url") or None
    path = store_cfg.get("path", "qdrant_data")
    rep.add(OK, f"backend: {url or f'embedded {path!r}'}")

    try:
        from database.store import PersonVectorStore, EMBEDDING_DIM
        store = PersonVectorStore(path=path, url=url,
                                 api_key=os.environ.get("QDRANT_API_KEY") or None,
                                 **({} if model_dim is None else {"dim": model_dim}))
    except Exception as e:                                           # noqa: BLE001
        rep.add(FAIL, f"cannot reach the store: {type(e).__name__}: {e}",
                "Start it:  docker compose up -d  &&  "
                "curl -s http://localhost:6333/readyz")
        return

    try:
        exists = store.client.collection_exists(store.collection)
    except Exception as e:                                           # noqa: BLE001
        rep.add(FAIL, f"cannot query the collection: {e}")
        return

    if not exists:
        rep.add(OK, f"collection {store.collection!r} does not exist yet",
                f"It will be created at {store.dim}-d on the first run.")
        return

    info = store.client.get_collection(store.collection)
    params = info.config.params.vectors
    got_dim = getattr(params, "size", None)
    metric = str(getattr(params, "distance", ""))
    count = store.count()
    rep.add(OK, f"collection {store.collection!r}: {count} point(s), "
                f"dim={got_dim}, metric={metric}")

    if model_dim is not None and got_dim is not None and int(got_dim) != int(model_dim):
        detail = (
            f"The collection is {got_dim}-d; this build embeds at {model_dim}-d.\n"
            f"THIS IS THE SILENT ONE. The store only warns, and IdentityStage\n"
            f"SWALLOWS the resulting ValueError -- so a run persists NOTHING and\n"
            f"the offline reconcile has nothing to reconcile. Every id in the\n"
            f"output video would be provisional.\n"
            f"\n"
            f"The {count} existing point(s) are from the old backbone and are not\n"
            f"comparable to new ones, so they have to go:\n"
            f"  python tools/preflight.py --fix-store")
        if fix:
            store.reset()
            rep.add(OK, f"collection REBUILT at {store.dim}-d "
                        f"(dropped {count} old point(s))")
        else:
            rep.add(FAIL, f"collection width MISMATCH: {got_dim}-d vs "
                          f"{model_dim}-d model", detail)
    elif model_dim is not None:
        rep.add(OK, f"collection width matches the model ({got_dim}-d)")

    if metric and "COSINE" not in metric.upper():
        rep.add(FAIL, f"collection metric is {metric}, not COSINE",
                "Every threshold in this project assumes cosine similarity.")

    if count == 0:
        rep.add(WARN, "the collection is EMPTY",
                "Normal before the first run. If you expected history, note that\n"
                "`main.py --reset` clears the store AND THEN runs the pipeline, so\n"
                "it looks like an ordinary run.")


def check_output(cfg, rep):
    rep.section("3. WATCHABLE OUTPUT (what you get from an RTSP run)")
    live = cfg.get("live") or {}
    out = live.get("output") or {}
    rec = live.get("reconcile") or {}

    if out.get("write_video", False):
        rep.add(OK, f"live.output.write_video: true -> "
                    f"{out.get('video_path', 'output_<cam>.mp4')}")
    else:
        rep.add(FAIL, "live.output.write_video is false -- an RTSP run writes no video")

    if rec.get("enabled", False):
        rep.add(OK, "live.reconcile.enabled: true",
                "Ids in the video are the RECONCILED ones (same person = same id\n"
                "and colour in every camera), not the provisional live ids.")
    else:
        rep.add(WARN, "live.reconcile.enabled is false",
                "The video will carry PROVISIONAL per-camera ids with no\n"
                "cross-camera identity. Almost certainly not what you want.")

    if rec.get("keep_frames", False):
        rep.add(OK, "live.reconcile.keep_frames: true",
                "Clips + sidecars are kept, so a run can be re-rendered at other\n"
                "settings with no camera time:\n"
                "  python tests/calibration/rerender_from_clips.py <run_id> --cross 0.5")
    else:
        rep.add(WARN, "live.reconcile.keep_frames is false",
                "You cannot re-render this run at another threshold afterwards --\n"
                "every setting question would cost a fresh capture.")

    codec = out.get("codec")
    if codec == "h264":
        rep.add(WARN, f"live.output.codec = {codec!r}",
                "This OpenCV build has no usable h264 encoder on the A6000; it\n"
                "falls back to mp4v automatically. Harmless, just noisy.")


def check_sources(cfg, rep):
    rep.section("4. CAMERAS")
    src = cfg.get("source") or {}
    env_urls = src.get("env_urls") or []
    if env_urls:
        missing = [k for k in env_urls if not os.environ.get(k)]
        if missing:
            rep.add(FAIL, f"source.env_urls names {missing}, absent from the "
                          f"environment",
                    "Add them to the untracked .env; a missing one silently drops "
                    "that camera.")
        else:
            rep.add(OK, f"source.env_urls resolves: {env_urls}")
    else:
        rep.add(WARN, "source.env_urls is empty",
                "Cameras are then passed on the command line, where the RTSP\n"
                "credentials are visible to every user via `ps` and land in shell\n"
                "history (issue #30). Prefer .env + env_urls.")

    rtsp = src.get("rtsp") or {}
    if str(rtsp.get("transport", "")).lower() == "tcp":
        rep.add(OK, "source.rtsp.transport = tcp (lost packets are retransmitted)")
    else:
        rep.add(WARN, f"source.rtsp.transport = {rtsp.get('transport')!r}",
                "UDP drops packets silently; a lost H.265 reference frame poisons\n"
                "both detection and the ReID crop.")
    if rtsp.get("read_timeout_ms"):
        rep.add(OK, f"rtsp read timeout {rtsp['read_timeout_ms']} ms")
    else:
        rep.add(FAIL, "no rtsp.read_timeout_ms",
                "Without it cap.read() can block forever, and a wedged camera\n"
                "makes Ctrl-C unable to reach the reconcile.")

    if src.get("resize_width"):
        rep.add(WARN, f"source.resize_width = {src['resize_width']}",
                "Boxes arrive in RESIZED pixels. Any geometry calibration authored\n"
                "at another resolution is silently wrong.")


def check_thresholds(cfg, rep):
    rep.section("5. THRESHOLDS (reported, not verifiable)")
    ident = cfg.get("identity") or {}
    recon = ident.get("reconcile") or {}
    live_id = ((cfg.get("live") or {}).get("identity") or {})
    rep.add(OK, "offline reconcile",
            f"cross-camera {recon.get('threshold', ident.get('threshold'))}   "
            f"same-camera {recon.get('same_camera_threshold')}   "
            f"per-camera {recon.get('per_camera') or '{}'}")
    rep.add(OK, "live engine",
            f"cross-camera {live_id.get('cross_camera_threshold')}   "
            f"same-camera {live_id.get('same_camera_threshold')}")
    rep.add(WARN, "these were derived in the PREVIOUS backbone's feature space",
            "Nothing can verify a threshold automatically, and a run's identity\n"
            "COUNT cannot either unless you know how many people were present --\n"
            "a lesson this project has now learned twice.\n"
            "Headcount-free evidence that they are too high: at\n"
            "same_camera_threshold 0.90 only 50% of labelled same-person fragments\n"
            "merge (100% at 0.75), and FastReID's different-person p95 is 0.40\n"
            "against OSNet's 0.59, so the bars sit higher than this space needs.\n"
            "Get GROUND TRUTH first -- watch a run's output_<cam>.mp4 and record\n"
            "verdicts with review_links.py <run_id> --label -- then sweep:\n"
            "  python tests/calibration/sweep_reconcile_thresholds.py <run_id> \\\n"
            "    --cross 0.45,0.55,0.63 --same \"cam_219=0.65,cam_224=0.65\"")


def check_geometry(cfg, rep):
    rep.section("6. GEOMETRY (optional; inert when off)")
    g = cfg.get("geometry") or {}
    if not g.get("enabled", False):
        rep.add(OK, "geometry.enabled: false -- not recording, changes nothing")
    else:
        path = g.get("calibration_path")
        if path and os.path.exists(path):
            try:
                from geometry.calibration import load_calibration
                recd = load_calibration(path)
                rep.add(OK, "geometry recording ON with a calibration",
                        recd.summary())
            except Exception as e:                                   # noqa: BLE001
                rep.add(FAIL, f"calibration at {path} is unreadable: {e}")
        else:
            rep.add(FAIL, f"geometry.enabled is true but no calibration at {path!r}",
                    "Nothing will be recorded. Fit one from a finished run:\n"
                    "  python tools/fit_floor_frame.py <run_id>")
    if ((g.get("reconcile") or {}).get("enabled", False)):
        rep.add(WARN, "the geometric reachability VETO is ON",
                "It can only refuse merges, and a refusal cannot be undone. Watch\n"
                "the re-rendered video before trusting it.")


def check_device(cfg, rep):
    rep.section("7. DEVICE")
    try:
        import torch
        cuda = torch.cuda.is_available()
        rep.add(OK if cuda else WARN,
                f"torch {torch.__version__}, cuda_available={cuda}",
                "" if cuda else
                "CPU only: FastReID R101-IBN is ~0.76 s/crop, so load-shedding\n"
                "drops nearly every frame. Correctness testing only -- any\n"
                "conclusion about identity quality drawn here is wrong.")
    except Exception as e:                                           # noqa: BLE001
        rep.add(FAIL, f"torch unavailable: {e}")
    det = (cfg.get("detector") or {}).get("model")
    if det and (os.path.exists(det) or not os.path.sep in str(det)):
        rep.add(OK, f"detector.model = {det}"
                    f"{'' if os.path.exists(det) else '  (auto-downloads)'}")


def main():
    p = argparse.ArgumentParser(description="Verify this deployment's configuration.")
    p.add_argument("--load-model", action="store_true",
                   help="build the ReID net and report its MEASURED width/input size")
    p.add_argument("--fix-store", action="store_true",
                   help="DESTRUCTIVE: drop and recreate a wrong-width collection")
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml")) as f:
        cfg = yaml.safe_load(f) or {}

    rep = Report()
    model_dim = check_reid(cfg, rep, args.load_model)
    check_store(cfg, rep, model_dim, args.fix_store)
    check_output(cfg, rep)
    check_sources(cfg, rep)
    check_thresholds(cfg, rep)
    check_geometry(cfg, rep)
    check_device(cfg, rep)

    rep.section("VERDICT")
    if rep.failed:
        print(f"  {len(rep.failed)} BLOCKING problem(s) -- a run would silently "
              f"produce garbage:")
        for _s, what, _d in rep.failed:
            print(f"    - {what}")
        print(f"\n  {len(rep.warned)} warning(s) besides.")
        return 1
    print(f"  No blocking problems. {len(rep.warned)} warning(s) to read above.")
    print("\n  Then, to see the RTSP output:")
    print("    python main.py --mode live          # Ctrl-C ONCE when done")
    print("    -> output_<cam>.mp4 per camera, with reconciled ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
