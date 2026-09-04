"""
main.py  --  the ENTRY POINT. Run this file to start the pipeline(s).

    python main.py                                   # file-batch on config videos
    python main.py --videos a.mp4 b.mp4              # file-batch on given files
    python main.py --mode live --videos rtsp://...   # real-time streaming pipeline

Routing (see main(), `live.run.mode`):
  * STREAM URLs (rtsp:// / http:// ...) are handled by the real-time pipeline in
    src/live/ -- it reads the feed live (never recording the raw stream) and, on
    stop, runs the offline reconcile to settle correct cross-camera ids.
  * LOCAL FILES are handled by the file-batch flow below.

This module is the "conductor" for the FILE-BATCH flow: it reads config.yaml and,
for EACH video, runs the same per-frame pipeline:

    STAGE 1-2   VideoSource      video file  ->  frame
    STAGE 3-4   PersonDetector   frame       ->  [Detection...] (+ track IDs)
    STAGE 5     drawing          detections  ->  annotated frame

MULTIPLE VIDEOS AT ONCE (and how we show them):
Each video gets its OWN detector (so their track IDs never mix). The heavy work
(decode + detect + track + draw) runs on one
worker THREAD per video, all at the same time. But OpenCV's windows are NOT
thread-safe, so the worker threads never call imshow directly -- instead each
worker drops its latest annotated frame into a shared buffer, and the MAIN
thread reads that buffer and shows every camera in its own window. This lets
you watch all cameras live, with the tracking boxes, safely.
"""

import argparse
from datetime import datetime
import os
import sys
import threading
import time
import yaml   # reads our config.yaml (installed alongside ultralytics)
import cv2

# Put the `src` folder on the import path so EVERY module -- here, in reid/, and
# in database/ -- imports the same bare way (`from detector import ...`). `src`
# is the single source root. Run `python main.py` from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from video_source import VideoSource, is_stream_path
from detector import PersonDetector, resolve_detector_cfg
from drawing import draw_detections, draw_hud
from interrupt_guard import InterruptGuard, print_stop_hint
from quiet import suppressed_native_stderr


def load_dotenv(path=".env"):
    """
    Minimal .env loader (no python-dotenv dependency). Reads KEY=VALUE lines from
    an untracked .env file into os.environ WITHOUT overriding vars already set in
    the real environment (so prod env vars win over a stray local .env). Secrets
    like QDRANT_API_KEY live here; the file is gitignored.
    """
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def load_config(path="config.yaml"):
    """Read config.yaml into a plain Python dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _run_filter(run_id):
    """#20: a server-side `run_id` filter, or None when it cannot be built.

    Every gallery read used to pull the WHOLE collection and drop other runs in
    Python. That cost grows with every run forever, and it is paid during
    finalization while the operator waits. None means "no filter" and the callers
    keep their existing Python-side check, so this is safe on any client.
    """
    if run_id is None:
        return None
    try:
        from qdrant_client import models as qmodels
        return qmodels.Filter(must=[qmodels.FieldCondition(
            key="run_id", match=qmodels.MatchValue(value=run_id))])
    except Exception:                                          # noqa: BLE001
        return None


def print_run_summary(store, jobs, cfg, run_id=None):
    """
    Print WHAT the run produced -- CONSOLE ONLY, no files written. Reads the
    gallery back from the store: total observations, distinct people
    (reid_ids), and which people were seen in more than one camera (the
    product's core result). Then lists the annotated videos on disk.
    """
    from collections import defaultdict

    print("\n===== RUN SUMMARY =====")
    if run_id is not None:
        print(f"  Run id: {run_id}")

    if store is not None:
        try:
            cam_obs = defaultdict(int)
            gid_obs = defaultdict(int)
            # reid_id -> camera -> set of track ids seen for that person there.
            gid_cameras = defaultdict(lambda: defaultdict(set))
            offset = None
            while True:
                pts, offset = store.client.scroll(
                    store.collection, limit=1000, offset=offset,
                    scroll_filter=_run_filter(run_id),
                    with_payload=True, with_vectors=False)
                for p in pts:
                    pl = p.payload or {}
                    if run_id is not None and pl.get("run_id") != run_id:
                        continue
                    cam_obs[pl.get("camera")] += 1
                    gid = pl.get("reid_id", pl.get("global_id"))
                    if gid is None:
                        continue
                    gid_obs[gid] += 1
                    gid_cameras[gid][pl.get("camera")].add(pl.get("track_id"))
                if offset is None:
                    break

            total = sum(cam_obs.values())
            cross_ids = sorted(g for g, cams in gid_cameras.items() if len(cams) > 1)
            print(f"  Store: {total} observations -> "
                  f"{len(gid_obs)} distinct people (reid_ids)")
            print(f"  Cross-camera people: {len(cross_ids)}")
            for gid in cross_ids:
                parts = []
                for camera in sorted(gid_cameras[gid]):
                    tracks = sorted(t for t in gid_cameras[gid][camera] if t is not None)
                    tracks_txt = " + ".join(f"track {int(t):04d}" for t in tracks)
                    parts.append(f"{camera} ({tracks_txt})")
                print(f"    GID {gid}: " + " + ".join(parts))
        except Exception as e:
            print(f"  Store: (could not summarize: {e})")

    # Annotated videos, if written.
    if cfg.get("display", {}).get("save_annotated"):
        vids = [f"output_{name}.mp4" for name, *_ in jobs
                if os.path.exists(f"output_{name}.mp4")]
        if vids:
            print(f"  Annotated videos: {', '.join(vids)}")
    print("=======================\n")


def get_screen_size():
    """
    Return the monitor's (width, height) in pixels so we can fit windows on
    screen. Uses tkinter (standard library). Falls back to 1080p if unavailable.
    """
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()  # don't actually show tkinter's own window
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return (1920, 1080)


# Video file extensions we recognise when scanning a directory.
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg")


def _is_url(path):
    """True for stream URLs (rtsp://, http://) which have no on-disk file to check."""
    return isinstance(path, str) and "://" in path


def _camera_name_from_source(path):
    """Derive a stable, friendly camera name from a source.

    Files keep their basename (unchanged -- the file-batch path relies on this).
    STREAM URLs are named by their host's last IP octet -- rtsp://.../ch01/0 has
    no useful basename ('0'), whereas cam_<octet> is stable, unique per camera,
    carries NO credentials, and lines up with the config's cam_<octet> convention
    and the live.topology edges (so a person's camera label matches the graph)."""
    s = str(path)
    if _is_url(s):
        from urllib.parse import urlparse
        host = (urlparse(s).hostname or "").strip()
        if host:
            octet = host.rsplit(".", 1)[-1]
            if octet:
                return f"cam_{octet}"
        return "cam"
    return os.path.splitext(os.path.basename(s))[0]


def normalize_sources(videos):
    """
    Turn a list of video entries into a clean list of (name, path) with UNIQUE
    names (names become output_<name>.mp4, so collisions would overwrite each
    other).

    Each entry may be either:
      - a plain string path/URL  -> name is derived from the file/host, OR
      - a dict {name, path}      -> we use the given friendly name.
    """
    result = []
    used = set()
    for entry in videos:
        if isinstance(entry, dict):
            path = entry["path"]
            name = entry.get("name") or _camera_name_from_source(path)
        else:
            path = entry
            name = _camera_name_from_source(path)

        name = name or "camera"
        # De-duplicate: cam, cam_2, cam_3, ...
        base, k = name, 1
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        result.append((name, path))
    return result


def _env_url_sources(cfg):
    """#30: stream URLs (with credentials) from environment variables.

    A URL on the command line is visible in `ps` to every user on the machine and
    is written to shell history. Naming the env var in config and keeping the
    secret in the untracked .env keeps it out of both.
    """
    names = ((cfg or {}).get("source", {}) or {}).get("env_urls") or []
    out, missing = [], []
    for name in names:
        value = os.environ.get(str(name))
        if value:
            out.append(value)
        else:
            missing.append(str(name))
    if missing:
        print(f"[main] source.env_urls names {missing} but they are not set in the "
              f"environment or .env -- those cameras are SKIPPED. (Set them in "
              f".env as NAME=rtsp://user:pass@host/path.)")
    if out:
        print(f"[main] {len(out)} camera URL(s) taken from the environment "
              f"(credentials kept out of `ps` and shell history).")
    return out


def resolve_sources(args, cfg):
    """
    Decide which videos to process, dynamically. Precedence:
        --videos <paths...>   >   --videos-dir <dir>   >   config source.videos

    So the pipeline runs on ANY videos the user points at, without editing code
    or config. Returns (sources, origin_description). Validates that local files
    exist (stream URLs are skipped). Exits with a clear message on bad input.
    """
    if args.videos:
        sources = normalize_sources(list(args.videos))
        origin = "--videos"
    elif args.videos_dir:
        if not os.path.isdir(args.videos_dir):
            raise SystemExit(f"[main] --videos-dir is not a directory: {args.videos_dir}")
        files = [os.path.join(args.videos_dir, f)
                 for f in sorted(os.listdir(args.videos_dir))
                 if f.lower().endswith(VIDEO_EXTS)]
        if not files:
            raise SystemExit(f"[main] No video files ({', '.join(VIDEO_EXTS)}) "
                             f"found in {args.videos_dir}")
        sources = normalize_sources(files)
        origin = f"--videos-dir {args.videos_dir}"
    else:
        # #30: env-provided stream URLs take precedence over source.videos, so a
        # deployment keeps its credentials in .env and never types them into a
        # command line that `ps` and shell history both record.
        env_urls = _env_url_sources(cfg)
        if env_urls:
            sources = normalize_sources(env_urls)
            origin = "source.env_urls (credentials from the environment)"
        else:
            sources = normalize_sources(cfg.get("source", {}).get("videos", []))
            origin = "config.yaml (source.videos)"

    missing = [p for _, p in sources if not _is_url(p) and not os.path.exists(p)]
    if missing:
        raise SystemExit("[main] These video files were not found:\n  "
                         + "\n  ".join(missing))
    return sources, origin


def build_gid_map(store, run_id):
    """
    Read the store (AFTER reconciliation) and return the FINAL global id for each
    camera-local track: {(camera, track_id): global_id}. One track should carry a
    single global id, but we take the most common as a safety net. This is what
    lets the re-render label the same person with the same GID in every camera.
    """
    from collections import defaultdict, Counter
    if store is None:
        return {}
    votes = defaultdict(Counter)
    offset = None
    while True:
        pts, offset = store.client.scroll(
            store.collection, limit=1000, offset=offset,
            scroll_filter=_run_filter(run_id),       # #20: server-side
            with_payload=True, with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            if run_id is not None and pl.get("run_id") != run_id:
                continue
            gid = pl.get("reid_id", pl.get("global_id"))
            if gid is None:
                continue
            votes[(pl.get("camera"), pl.get("track_id"))][gid] += 1
        if offset is None:
            break
    return {key: counter.most_common(1)[0][0] for key, counter in votes.items()}


def render_final_videos(jobs, cfg, shared, store, run_id,
                        gid_map=None, out_pattern="output_{name}.mp4",
                        fps_by_camera=None, quality=None, links=None):
    """
    SECOND PASS -- write output_<camera>.mp4 with the FINAL (post-reconciliation)
    global ids. The live pass only captured box geometry; cross-camera identity is
    settled afterwards by reconcile.py. Re-drawing here means a person who walked
    from cam A to cam B carries the SAME "GID n" (and the same colour) in BOTH
    output videos -- the core proof the pipeline works. We re-decode the source
    frames (cheap; no model) and draw the captured boxes, so boxes/track ids match
    the live pass exactly.

    gid_map / out_pattern exist so a PAST run can be re-rendered offline at
    different reconcile settings (tests/calibration/rerender_from_clips.py):
      * gid_map -- supply {(camera, track_id): gid} directly, which is exactly what
        reconcile_tracklets returns. Skips build_gid_map, so a re-render never has
        to write ids into the store first and can therefore leave the gallery
        untouched while producing a watchable video.
      * out_pattern -- write somewhere other than output_<cam>.mp4, so two settings
        can be rendered side by side and compared instead of overwriting.

    fps_by_camera : {camera: fps} -- each camera's OWN playback rate (#45/#46). A
        single global output_fps tagged all four cameras at 20, so cam_224 at a real
        14.8 fps played 1.35x fast and cam_219 at 24.2 played 1.21x slow. Four videos
        on four wrong time scales cannot be compared by eye, which is how identity is
        actually judged. Falls back to cfg display.output_fps per camera.
    """
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    if gid_map is None:
        gid_map = build_gid_map(store, run_id)
    resize_width = cfg["source"].get("resize_width", 0)
    fps = float(cfg["display"].get("output_fps", 20.0))
    fps_by_camera = fps_by_camera or {}
    quality = quality or {}         # {(camera, track_id): {"fit":..,"margin":..}}
    links = links or {}             # {(camera, track_id): {"score","bar","with","linked"}}

    def render_one(name, path):
        annos = shared["annotations"].get(name)
        if not annos:
            # #64: this used to `return` silently. With four cameras, one producing
            # no video and no message is easy to miss entirely -- and "the file is
            # missing" is indistinguishable from "the file is stale from an earlier
            # run", which is exactly the trap that hid two lost runs.
            print(f"[render] {name}: NO annotations captured -> NO video written. "
                  f"That camera contributed nothing to this run (never connected, "
                  f"or every frame was dropped).")
            return
        out_path = out_pattern.format(name=name)
        writer = None
        try:
            with VideoSource(path=path) as cam:
                for frame_index, frame in enumerate(cam.frames()):
                    if frame_index >= len(annos):
                        break  # only re-render the frames the live pass processed
                    if resize_width and frame.shape[1] > resize_width:
                        scale = resize_width / frame.shape[1]
                        frame = cv2.resize(
                            frame,
                            (resize_width, int(round(frame.shape[0] * scale))),
                            interpolation=cv2.INTER_AREA)

                    dets = []
                    for anno in annos[frame_index]:
                        if isinstance(anno, dict):
                            x1 = anno["x1"]
                            y1 = anno["y1"]
                            x2 = anno["x2"]
                            y2 = anno["y2"]
                            track_id = anno.get("track_id")
                            conf = anno.get("confidence", 0.0)
                        else:
                            x1, y1, x2, y2, track_id, conf = anno[:6]

                        dets.append(
                            SimpleNamespace(
                                # cv2.rectangle needs ints. Detection declares int
                                # coordinates, so the live path already satisfies
                                # this -- but these can also arrive from a
                                # persisted .annotations.json, where a float would
                                # otherwise fail deep inside OpenCV with an
                                # unreadable overload-resolution error.
                                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                                track_id=track_id, confidence=conf,
                                global_id=gid_map.get((name, track_id)),
                                reid_id=gid_map.get((name, track_id)),
                                # Gap to the nearest OTHER cluster. None when
                                # reconcile did not report it.
                                reid_margin=(quality.get((name, track_id)) or {}
                                             ).get("margin"),
                                # The score that actually LINKED this tracklet to
                                # another one, and the bar it cleared. reid_fit is
                                # GONE: it scored a tracklet against a cluster
                                # prototype containing itself, so it read exactly
                                # 1.000 for anyone who linked to nothing -- maximum
                                # confidence on precisely the unverified boxes.
                                reid_link_score=(links.get((name, track_id)) or {}
                                                 ).get("score"),
                                reid_link_bar=(links.get((name, track_id)) or {}
                                               ).get("bar"),
                                reid_link_with=(links.get((name, track_id)) or {}
                                                ).get("with"),
                                reid_linked=(links.get((name, track_id)) or {}
                                             ).get("linked", True),
                                # Same numbers, under the names drawing.py's
                                # UNRESOLVED branch reads. Suppressed tracklets get
                                # no reid_id, so they take that branch instead.
                                reid_best_score=(links.get((name, track_id)) or {}
                                                 ).get("score"),
                                reid_threshold=(links.get((name, track_id)) or {}
                                                ).get("bar"),
                                reid_best_candidate=(links.get((name, track_id)) or {}
                                                     ).get("with"),
                                reid_reason=(links.get((name, track_id)) or {}
                                             ).get("reason"),
                                # #32/#33: no id from reconcile => unresolved, so the
                                # renderer must not print a bare track number that
                                # reads as an identity.
                                unresolved=(gid_map.get((name, track_id)) is None),
                            )
                        )
                    frame = draw_detections(frame, dets)
                    frame = draw_hud(frame, person_count=len(dets))

                    if writer is None:
                        h, w = frame.shape[:2]
                        out_fps = float(fps_by_camera.get(name) or fps)
                        # #63: output.codec was accepted in config and then ignored
                        # here -- mp4v was hardcoded, so `codec: h264` did nothing.
                        # Try the configured codec, fall back to mp4v (which always
                        # works) rather than silently producing no video.
                        want = str(cfg.get("display", {}).get("codec")
                                   or "mp4v").lower()
                        fourcc_name = {"h264": "avc1", "avc1": "avc1",
                                       "mp4v": "mp4v"}.get(want, "mp4v")
                        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                        if fourcc_name != "mp4v":
                            # Expected to fail on a build without an h264 encoder,
                            # and it reports that from C++ straight to fd 2 -- four
                            # red lines that have twice been read as the run
                            # failing. Silence the PROBE only; the mp4v fallback
                            # below is never suppressed.
                            with suppressed_native_stderr():
                                writer = cv2.VideoWriter(out_path, fourcc, out_fps,
                                                         (w, h))
                                opened = writer.isOpened()
                        else:
                            writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))
                            opened = writer.isOpened()
                        if not opened and fourcc_name != "mp4v":
                            print(f"[render] {name}: codec {want!r} unavailable in "
                                  f"this OpenCV build; using mp4v (expected on this "
                                  f"deployment, not an error).")
                            writer = cv2.VideoWriter(
                                out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                out_fps, (w, h))
                    writer.write(frame)
        except Exception as e:
            print(f"[render] {name}: ERROR re-rendering: {e}")
        finally:
            if writer is not None:
                writer.release()
                print(f"[render] Final annotated video -> {out_path}")

    max_workers = min(len(jobs), os.cpu_count() or len(jobs) or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(render_one, name, path) for name, path, *_ in jobs]
        for fut in futures:
            fut.result()


# =============================================================================
# The WORKER: runs the whole pipeline for ONE video, on its own thread.
# It does NOT touch any window. It only:
#   - reads frames, detects+tracks, draws boxes, and
#   - publishes its latest annotated frame into `shared` for the main thread.
# =============================================================================
def process_video(name, path, detector, embedder, store, identity,
                  locks, cfg, shared, stop_event, run_id):
    trk_cfg = cfg["tracker"]
    # 0 = process the whole video; a positive number stops early (handy for a
    # quick CPU test without waiting for hundreds of frames).
    max_frames = cfg["source"].get("max_frames", 0)
    # 0 = keep native resolution. A positive width downscales every frame (keeping
    # aspect) BEFORE detect/track/embed -- QHD footage on CPU is dominated by
    # pixel count, so e.g. 1280 is dramatically faster for dev with little ReID
    # cost. Everything downstream uses the same resized frame, so coords stay
    # consistent (crops are smaller too).
    resize_width = cfg["source"].get("resize_width", 0)
    show_window = cfg["display"].get("show_window", True)

    tag = f"[{name}]"
    frame_index = 0
    processed_here = 0   # how many fresh embeddings THIS camera has checked
    # Per-frame box geometry captured for the FINAL re-render pass. We record the
    # boxes + track ids (NOT the pixels) as we go, so after cross-camera
    # reconciliation we can re-draw the video with the final global ids without
    # re-running detection. annotations[f] = dicts with box coords, track id,
    # and confidence.
    annotations = []

    try:
        with VideoSource(path=path) as cam:
            print(f"{tag} Pipeline running on '{path}'.")
            for frame in cam.frames():        # STAGE 1-2: one frame at a time

                # Stop early if the user pressed 'q' in the main thread.
                if stop_event.is_set():
                    break

                # ---- Optional downscale (dev speed on CPU) -----------------
                # Do it FIRST so every later stage sees the same smaller frame.
                if resize_width and frame.shape[1] > resize_width:
                    scale = resize_width / frame.shape[1]
                    frame = cv2.resize(
                        frame, (resize_width, int(round(frame.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA)

                # ---- STAGE 3 (+4): detect, with or without track IDs --------
                if trk_cfg["enabled"]:
                    detections = detector.track(frame)   # Stage 4: with IDs
                else:
                    detections = detector.detect(frame)  # Stage 3: no IDs

                # ---- STAGE 5/6: embed tracked people + check Qdrant ---------
                # Uses the CLEAN frame (before boxes are drawn). We embed each
                # track (throttled + cached by TrackEmbedder) and hand fresh
                # embeddings to the Identity Service, which searches its own
                # gallery and assigns a stable GLOBAL id (or falls back to just
                # persisting raw observations when identity assignment is off).
                if embedder is not None:
                    with locks["model"]:            # shared model: one at a time
                        embedder.process(frame, detections, frame_index)
                    fresh = embedder.last_embedded   # tracks re-embedded this frame

                    if identity is not None and fresh:
                        # Turn fresh embeddings into stable GLOBAL ids. The service
                        # searches the gallery AND commits under the chosen id, so
                        # it is the sole writer -- serialize it with one lock.
                        with locks["identity"]:
                            for d in fresh:
                                gid = identity.assign(
                                    name, d.track_id, d.embedding, frame_index,
                                    run_id=run_id,
                                    crop_quality=d.crop_quality)
                                d.global_id = gid
                                d.reid_id = gid
                        processed_here += len(fresh)
                    elif store is not None and fresh:
                        # Identity off: just persist the raw observations.
                        payloads = [
                            {
                                "camera": name,
                                "track_id": d.track_id,
                                "frame": frame_index,
                                "run_id": run_id,
                                "crop_quality": d.crop_quality,
                            }
                            for d in fresh
                        ]
                        with locks["store"]:
                            store.add_many([d.embedding for d in fresh], payloads)
                        processed_here += len(fresh)

                # ---- Record box geometry for the final re-render pass --------
                # Captured BEFORE drawing (clean coords). The annotated video is
                # NOT written here: it is written in a second pass after
                # reconciliation so its labels use the FINAL global ids (so the
                # same person carries the same GID across both cameras' videos).
                annotations.append([
                    {
                        "x1": d.x1,
                        "y1": d.y1,
                        "x2": d.x2,
                        "y2": d.y2,
                        "track_id": d.track_id,
                        "confidence": d.confidence,
                    }
                    for d in detections
                ])

                if show_window:
                    # ---- STAGE 5: draw (modifies frame in place) ----------
                    frame = draw_detections(frame, detections)
                    frame = draw_hud(frame, person_count=len(detections))

                    # ---- Publish latest frame for the main thread to show --
                    # We store a COPY so the display thread can read it while
                    # we move on to the next frame. Skip this entirely in
                    # headless runs; the final output video is rendered later.
                    with shared["lock"]:
                        shared["frames"][name] = frame.copy()

                frame_index += 1

                # ---- Heartbeat: prove the pipeline is alive on long videos ---
                # Full videos on CPU take minutes; without this the terminal
                # looks frozen. Log progress every 30 frames.
                if frame_index % 30 == 0:
                    print(f"{tag} frame {frame_index}: {len(detections)} people, "
                          f"{processed_here} fresh embeddings checked so far")

                # Stop early if a frame cap was set (for quick tests).
                if max_frames and frame_index >= max_frames:
                    print(f"{tag} Reached max_frames={max_frames}, stopping.")
                    break

    except Exception as e:
        # One video failing shouldn't kill the others -- log and move on.
        print(f"{tag} ERROR: {e}")
    finally:
        # Hand the captured geometry to the main thread for the re-render pass,
        # and mark this camera as finished.
        with shared["lock"]:
            shared["annotations"][name] = annotations
            shared["done"].add(name)
            shared["processed"][name] = frame_index
        print(f"{tag} Done. Processed {frame_index} frames.")


# =============================================================================
# The DISPLAY loop: runs on the MAIN thread. Reads each camera's latest frame
# from the shared buffer and shows it in its own window. This is the ONLY place
# that calls OpenCV's window functions.
# =============================================================================
def run_display(names, cfg, shared, stop_event, total_videos):
    disp_cfg = cfg["display"]
    window_scale = disp_cfg.get("window_scale", 1.0)

    screen_w, screen_h = get_screen_size()
    created = set()   # which windows we've already created + sized
    # Some environments (headless servers, a WSL without WSLg/X) can't open a
    # GUI window -- cv2 raises then. Rather than crash the whole run (which would
    # also lose the saved videos), we fall back to a headless wait: the session
    # keeps running and is stopped with Ctrl-C, which still finalizes outputs.
    gui_ok = True

    while True:
        # Take a quick snapshot of the latest frames under the lock.
        with shared["lock"]:
            frames = dict(shared["frames"])
            done = set(shared["done"])

        if gui_ok:
            try:
                for name, frame in frames.items():
                    if name not in created:
                        # Size each window to fit the screen. When there are several
                        # cameras, give each a slice of the screen width so they fit
                        # side by side (0.9 / N), keeping aspect ratio.
                        h, w = frame.shape[:2]
                        frac = 0.9 / max(1, total_videos)
                        fit = min(1.0, (screen_w * frac) / w, (screen_h * 0.9) / h)
                        scale = fit * window_scale
                        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow(name, max(1, int(w * scale)), max(1, int(h * scale)))
                        created.add(name)

                    cv2.imshow(name, frame)

                # waitKey both refreshes the windows AND reads the keyboard. 'q' quits.
                if (cv2.waitKey(30) & 0xFF) == ord("q"):
                    print("[display] 'q' pressed -> stopping all videos.")
                    stop_event.set()
                    break
            except cv2.error as e:
                gui_ok = False
                print(f"[display] Live windows unavailable ({e}); continuing "
                      f"HEADLESS. Press Ctrl-C to stop and finalize the saved "
                      f"videos.")

        if not gui_ok:
            # No windows: can't read 'q', so wait for stop (Ctrl-C sets it) or
            # for every source to finish on its own.
            if stop_event.is_set():
                break
            time.sleep(0.03)

        # Exit once every video has finished. (We've already shown whatever
        # frames each produced; a video that failed instantly just won't have
        # a window.)
        if len(done) == total_videos:
            print("[display] All videos finished.")
            break

    # Keep the final frames up until a key is pressed, so nothing vanishes.
    if gui_ok and created and not stop_event.is_set():
        print("[display] Press any key in a window to close.")
        cv2.waitKey(0)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-camera person ReID pipeline.")
    p.add_argument(
        "--reset", action="store_true",
        help="Wipe the vector store collection before running, so a test starts "
             "from a clean gallery. DESTRUCTIVE -- deletes all stored embeddings.",
    )
    p.add_argument(
        "--videos", nargs="+", metavar="VIDEO", default=None,
        help="One or more video files or RTSP/HTTP stream URLs to process, "
             "space-separated. Overrides config source.videos. Each camera's name "
             "is derived from the file name. Example: "
             "--videos cam_a.mp4 cam_b.mp4",
    )
    p.add_argument(
        "--videos-dir", metavar="DIR", default=None,
        help="Process every video file in this directory "
             f"({'/'.join(e.lstrip('.') for e in VIDEO_EXTS)}). "
             "Overrides config source.videos.",
    )
    p.add_argument(
        "--mode", choices=["auto", "live", "batch"], default=None,
        help="Override live.run.mode. auto (default; live if a source is a URL) | "
             "live (v5 real-time src/live/ pipeline) | batch (file pipeline). "
             "Handy for testing the live pipeline without editing config.yaml.",
    )
    return p.parse_args()


def main():
    # ---- Load settings ------------------------------------------------------
    args = parse_args()
    load_dotenv()          # pull secrets (QDRANT_API_KEY, ...) from untracked .env
    cfg = load_config()
    # RTSP transport + socket timeouts, BEFORE anything opens a capture: FFmpeg
    # reads its options at open time, so this cannot be done later. Applies to both
    # the live pipeline and the file-batch flow, since both open through
    # VideoSource. Files ignore it. See source.rtsp and plan #28/#29.
    from video_source import rtsp_options_from_config
    rtsp_options_from_config(cfg)
    det_cfg = cfg["detector"]
    trk_cfg = cfg["tracker"]
    disp_cfg = cfg["display"]

    sources, src_origin = resolve_sources(args, cfg)
    if not sources:
        print("[main] No videos to process. Pass --videos <paths...>, "
              "--videos-dir <dir>, or set source.videos in config.yaml.")
        return
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[main] Preparing {len(sources)} video(s) from {src_origin}: "
          f"{', '.join(n for n, _ in sources)}")
    print(f"[main] Run id: {run_id}")

    # ---- pipeline routing --------------------------------------------------
    # `live.run.mode` selects the path; a capability report is logged so the
    # chosen device/decode is visible even on a headless server.
    #   live  -> the real-time streaming pipeline (src/live/), files OR streams
    #   auto  -> live pipeline if ANY source is a stream URL, else file-batch
    #   batch -> file-batch (files only; stream URLs are rejected)
    # Stream URLs are ALWAYS handled by the live pipeline (it reads RTSP live and
    # settles correct ids with the offline reconcile on stop); the file-batch
    # path below only ever sees local files.
    live_cfg = cfg.get("live", {}) or {}
    run_cfg = live_cfg.get("run", {}) or {}
    run_mode = args.mode or run_cfg.get("mode", "auto")   # --mode overrides config
    try:
        from live.capabilities import capability_report
        cap = capability_report(
            requested_device=run_cfg.get("device", "auto"),
            requested_nvdec=(live_cfg.get("capture", {}) or {}).get("nvdec", "auto"),
        )
        print(f"[main] Capability report: {cap}")
    except Exception as e:
        print(f"[main] (capability report unavailable: {e})")

    has_stream = any(is_stream_path(p) for _, p in sources)
    if run_mode == "live" or (run_mode == "auto" and has_stream):
        # Real-time streaming pipeline (src/live/): reads each source live (never
        # recording the raw RTSP feed) and, on stop, runs the offline reconcile to
        # settle correct cross-camera ids and re-render output_<cam>.mp4.
        from live.pipeline import LivePipeline
        LivePipeline(sources, cfg).run()
        print("[main] All done.")
        return
    if has_stream:
        raise SystemExit(
            "[main] batch mode cannot process stream URLs. Use --mode live "
            "(or run.mode: auto) for RTSP/HTTP streams.")

    # ---- Build one detector PER video ---------------------------------------
    # Each detector has its OWN tracker memory (IDs never bleed between videos).
    jobs = []  # each: (name, path, detector)
    for name, path in sources:
        # Per-camera overrides (angle/lighting differ per view) merged over the
        # global detector block -- see detector.resolve_detector_cfg.
        cam_det_cfg = resolve_detector_cfg(det_cfg, name)
        detector = PersonDetector(
            model_path=cam_det_cfg["model"],
            confidence_threshold=cam_det_cfg["confidence_threshold"],
            person_class_id=cam_det_cfg["person_class_id"],
            tracker_config=trk_cfg["config"],
            pose_ensemble=cam_det_cfg.get("pose_ensemble"),
            iou=cam_det_cfg.get("iou", 0.7),
        )
        jobs.append((name, path, detector))

    # ---- Build the SHARED ReID model + Qdrant access ------------------------
    # ONE extractor for the whole run: the model is the expensive resource, so we
    # load it once, share it across camera threads, and guard it with a lock
    # (worker threads must not run the model concurrently). ONE store, because all
    # cameras must write to the SAME gallery. But each camera gets its OWN
    # TrackEmbedder: its cache is keyed by track_id, and track_ids are per-camera
    # (cam A's track 1 is not cam B's track 1), so the caches must not mix.
    extractor = None
    store = None
    identity = None
    embedders = {}
    locks = {
        "model": threading.Lock(),      # shared model: one forward pass at a time
        "store": threading.Lock(),      # raw store writes (identity-off fallback)
        "identity": threading.Lock(),   # identity: mints ids + writes gallery
    }

    reid_cfg = cfg.get("reid", {})
    if reid_cfg.get("enabled"):
        from reid.extractor import ReIDExtractor
        from reid.service import TrackEmbedder
        # Resolve reid.device through the SAME helper the live path uses for
        # live.run.device, so both keys accept the same vocabulary
        # (auto | cpu | cuda | cuda:N) and behave identically: "auto" -> GPU when
        # one is visible else CPU, and an unavailable "cuda" degrades to CPU with
        # a warning here instead of failing later on the first .to(device).
        # Falling back to the raw config value keeps this non-fatal if the live
        # package cannot be imported (ReIDExtractor auto-picks when given None).
        try:
            from live.capabilities import detect_device
            reid_device = detect_device(reid_cfg.get("device", "auto"))
        except Exception as e:
            print(f"[main] (device auto-detect unavailable: {e})")
            reid_device = reid_cfg.get("device")
        print(f"[main] Loading ReID model on {reid_device} "
              f"(shared across all cameras)...")
        extractor = ReIDExtractor(
            weights=reid_cfg["weights"],
            device=reid_device,
            # #39/#56: the feature tap and the batch ceiling. Changing the tap
            # re-scales every threshold, so it is recorded in the run banner below.
            tap=reid_cfg.get("tap", "post_relu"),
            max_batch=int(reid_cfg.get("max_batch", 32)),
            # Which backbone. Recorded in the banner for the same reason as the
            # tap: it defines the feature space every threshold refers to.
            model=reid_cfg.get("model"),
        )
        print(f"[main] ReID model: {extractor.describe()}")
        print(f"[main] ReID tap: {reid_cfg.get('tap', 'post_relu')} "
              f"-- thresholds are model- and tap-specific; never compare score "
              f"logs across either.")
        for name, _ in sources:
            embedders[name] = TrackEmbedder(
                extractor,
                interval=reid_cfg.get("interval", 10),
            interval_sec=reid_cfg.get("interval_sec", 0.0),
                ttl=reid_cfg.get("ttl", 300),
                quality=reid_cfg.get("quality"),
                max_embeddings_per_track=reid_cfg.get("max_embeddings_per_track", 0),
                warmup_embeddings=reid_cfg.get("warmup_embeddings", 3),
            )
        print("[main] ReID enabled -> embedding tracked people (no ids assigned).")

    store_cfg = cfg.get("store", {})
    if store_cfg.get("enabled"):
        from database.store import PersonVectorStore
        # URL: env var wins over config.yaml. API key: env only (never committed).
        url = os.environ.get("QDRANT_URL") or store_cfg.get("url") or None
        api_key = os.environ.get("QDRANT_API_KEY") or None
        path = store_cfg.get("path", "qdrant_data")
        # Size the collection from the MODEL THAT IS ACTUALLY LOADED rather than
        # from a constant, so swapping `reid.model` for a different-width
        # backbone cannot create (or silently reuse) a mis-sized collection --
        # the store's dimension guard then reports the mismatch at startup
        # instead of on the first insert.
        store = PersonVectorStore(
            path=path, url=url, api_key=api_key,
            **({"dim": extractor.embedding_dim} if extractor is not None else {}))
        backend = url if url else f"LOCAL '{path}'"
        print(f"[main] Vector store ready at {backend} "
              f"(existing points: {store.count()}).")

        if args.reset:
            store.reset()
            print(f"[main] --reset: cleared the store (now {store.count()} points).")

    # The Identity Service needs the store as its gallery. It is SHARED across
    # cameras (global ids are only "global" if every camera decides from the same
    # gallery), and it is the sole writer once enabled.
    id_cfg = cfg.get("identity", {})
    if id_cfg.get("enabled") and store is not None:
        from identity.service import IdentityService
        # NOTE: the online decision core stays OFF here (online_cfg=None). This is
        # the FILE-BATCH path -- identity is a sticky live-assign during the pass
        # plus the offline reconcile afterwards, which is what gives the saved MP4
        # its final cross-camera ids. (Streams are handled by src/live/ instead.)
        identity = IdentityService(
            store,
            threshold=id_cfg.get("threshold", 0.85),
            top_k=id_cfg.get("top_k", 10),
            bank_size=id_cfg.get("bank_size", 20),
            min_score_gap=id_cfg.get("min_score_gap", 0.03),
            rerank_cfg=id_cfg.get("rerank"),
            verification_cfg=id_cfg.get("verification"),
            online_cfg=None,
        )
        print("[main] Identity Service enabled -> assigning GLOBAL ids.")

    # ---- Shared state between worker threads and the display loop ------------
    # frames: {name: latest annotated frame}   done: {names that finished}
    shared = {
        "lock": threading.Lock(),
        "frames": {},
        "done": set(),
        "annotations": {},   # {name: [per-frame box geometry]} for the re-render
        "processed": {},     # {name: frames processed} -- detects a total failure
    }
    stop_event = threading.Event()  # set to True to ask all workers to stop

    # ---- Start one worker thread per video ----------------------------------
    threads = []
    for name, path, detector in jobs:
        t = threading.Thread(
            target=process_video,
            args=(name, path, detector, embedders.get(name), store,
                  identity, locks, cfg, shared, stop_event, run_id),
            name=name,
        )
        t.start()
        threads.append(t)

    # ---- Display on the main thread (or just wait, if windows are off) -------
    # Ctrl-C is caught so a deliberately-stopped live session still finalizes:
    # we set stop_event and fall through to the reconcile + re-render below,
    # producing the saved videos just like pressing 'q' does.
    show_window = disp_cfg.get("show_window", True)
    print_stop_hint("batch")
    try:
        if show_window:
            run_display([n for n, *_ in jobs], cfg, shared, stop_event, len(jobs))
        else:
            # Headless: nothing to show, just wait for all videos to finish.
            for t in threads:
                t.join()
    except KeyboardInterrupt:
        print("\n[main] Interrupted (Ctrl-C) -> stopping all sources, then "
              "reconciling + rendering the final videos. Do NOT press Ctrl-C "
              "again; that would discard the reconciled ids.")

    # ---- FINALIZATION PHASE (guarded against further Ctrl-C) ----------------
    # From here to print_run_summary is what actually produces the deliverable:
    # winding the workers down, deciding the correct cross-camera ids, and
    # re-rendering the videos with them. A Ctrl-C landing anywhere in here used
    # to abort with provisional ids and half-written MP4s, so the whole phase
    # runs under InterruptGuard: extra presses print a warning instead of
    # raising, and 3 in a row still force-quit (see src/interrupt_guard.py).
    with InterruptGuard("reconciling ids + rendering the final videos"):
        _finalize_run(threads, stop_event, shared, jobs, cfg, disp_cfg, id_cfg,
                      store, identity, run_id)

    print("[main] All done.")


def _finalize_run(threads, stop_event, shared, jobs, cfg, disp_cfg, id_cfg,
                  store, identity, run_id):
    """Wind the workers down, reconcile ids across cameras, re-render the final
    videos, print the summary. Always called inside an InterruptGuard so a
    stray Ctrl-C can't leave the run with provisional ids."""
    # Ask workers to stop (if the user quit) and wait for them to wind down.
    stop_event.set()
    for t in threads:
        t.join(timeout=5)

    cv2.destroyAllWindows()

    # ---- Fail loudly if EVERY camera produced nothing -----------------------
    # process_video swallows per-camera exceptions so one bad source can't kill
    # the others -- but if ALL of them processed 0 frames (broken install, wrong
    # paths, unreadable streams), we must NOT silently reconcile an empty gallery
    # and exit 0. Surface it as an error so an unattended run is detectable.
    # (A partial failure is left to the per-camera ERROR logs above; a legitimate
    # people-free video still processes frames, so 0 total frames == real failure.)
    processed = shared.get("processed", {})
    if jobs and processed and sum(processed.values()) == 0:
        raise SystemExit(
            "[main] ERROR: every camera processed 0 frames "
            f"({', '.join(sorted(processed))}) -- nothing to reconcile. "
            "Check the video paths / install and re-run.")

    # ---- STAGE 7b: offline reconciliation ----------------------------------
    # The live Identity Service decides each track once and never revisits it, so
    # a person seen in two cameras AT THE SAME TIME is minted twice and can't be
    # matched live. Now that every camera has finished and the gallery is fully
    # populated, merge global ids that are the same person across cameras.
    recon_cfg = id_cfg.get("reconcile", {}) if id_cfg else {}
    # #34: filled by reconcile with per-tracklet fit/margin; stays empty when
    # reconcile does not run, in which case the labels simply carry no score.
    reid_quality = {}
    reid_links = {}
    if identity is not None and recon_cfg.get("enabled"):
        from identity.reconcile import (describe_reconcile_kwargs,
                                        reconcile_tracklets,
                                        resolve_reconcile_kwargs)
        # EVERY reconcile setting comes from the one resolver, so this path, the
        # live pipeline and both offline tools cannot drift. They HAD drifted:
        # this call site defaulted min_tracklet_observations to 1 (the pipeline
        # used 3) and fell back to identity.threshold=0.85 (the pipeline used
        # 0.63). See resolve_reconcile_kwargs and REMEDIATION_PLAN.md Part M.
        recon_kwargs = resolve_reconcile_kwargs(cfg)
        print("[main] Reconciling identities across cameras... "
              "(do NOT press Ctrl-C -- this is where the final ids are decided)")
        print(f"[main] reconcile settings: {describe_reconcile_kwargs(recon_kwargs)}")
        reconcile_tracklets(store, run_id=run_id, quality_out=reid_quality,
                            link_out=reid_links, **recon_kwargs)

    # ---- Final render: annotated videos with the reconciled global ids ------
    # Done AFTER reconciliation so the same person carries the same GID (and
    # colour) across every camera's output video.
    if disp_cfg.get("save_annotated"):
        print("[main] Rendering annotated videos with final global ids...")
        render_final_videos(jobs, cfg, shared, store, run_id,
                            quality=reid_quality, links=reid_links)

    print_run_summary(store, jobs, cfg, run_id=run_id)


if __name__ == "__main__":
    # This guard means main() only runs when you execute `python main.py`,
    # and is REQUIRED for threads to behave correctly.
    try:
        main()
    except KeyboardInterrupt:
        # Only reachable when the user force-quit through InterruptGuard (3
        # Ctrl-C presses during finalization). Exit quietly with the standard
        # SIGINT code instead of dumping a traceback.
        print("\n[main] Force-quit during finalization -- the annotated videos "
              "may be missing or keep provisional per-camera ids.")
        sys.exit(130)
