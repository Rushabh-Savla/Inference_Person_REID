"""
pipeline.py  --  LivePipeline: wire the stages, run, shut down cleanly.

Stage 1 backbone (headless; the deliverable is output_<cam>.mp4):

  per camera:  DecodeBackend -> CaptureThread -> NewestSlot            \
  shared:      BatchScheduler -> inference_queue -> InferenceStage ->   > identity_queue
  shared:      IdentityStage -> per-cam render_queue                   /
  per camera:  RenderStage -> writer_queue -> WriterStage -> output_<cam>.mp4

Startup order (v5 §12): capability report -> WARM-UP (dummy batch through
detector + ReID, BEFORE opening any source, so first real frames don't pay init
latency) -> start consumer stages -> start captures last.

Shutdown (v5 §10, race-free, in try/finally): stop -> join capture -> drain/join
scheduler+inference+identity -> join render -> flush+release writers (finalize
every MP4) -> report. A stop (Ctrl-C, all files ended, or max_duration) never
corrupts an output. N-camera generic: everything is per-camera + over the active
set; adding a camera is just another source entry.
"""

import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np

# src/ is already on sys.path (main.py inserts it); import the reused components.
from detector import PersonDetector, resolve_detector_cfg
from reid.extractor import ReIDExtractor
from reid.service import TrackEmbedder

from live.capabilities import capability_report
from live.decode_backend import make_decode_backend
from live.queues import NewestSlot, DropOldestQueue
from live.capture import CaptureThread
from live.scheduler import BatchScheduler
from live.inference import InferenceStage
from live.identity_stage import IdentityStage
from live.topology import FailOpenTopology, GraphTopology
from live.render import RenderStage
from live.writer import WriterStage
from interrupt_guard import InterruptGuard, print_stop_hint


class _QuietOnBrokenPipe:
    """Text-stream wrapper that goes silent instead of raising on a dead consumer.

    WHY THIS EXISTS (two real runs were lost to it): `python main.py ... | tee log`
    puts python and tee in ONE foreground process group, so Ctrl-C is delivered to
    both. tee has no SIGINT handler and dies immediately; every subsequent print in
    python then raises BrokenPipeError. The first such print is inside the shutdown
    sequence, and `_report(final=True)` runs BEFORE `_finalize_offline()` -- so a
    cosmetic print failure aborted the run before the reconciled ids were ever
    decided, and the traceback went into the same dead pipe so nothing was visible.
    A dropped SSH session or a closed terminal does exactly the same thing.

    Printing is never worth the deliverable. Installed only around finalization, so
    normal output is untouched; once the stream breaks we stop trying.
    """

    def __init__(self, stream):
        self._stream = stream
        self.broken = False

    def write(self, data):
        if self.broken:
            return len(data)
        try:
            return self._stream.write(data)
        except (BrokenPipeError, OSError, ValueError):
            self.broken = True
            return len(data)

    def flush(self):
        if self.broken:
            return
        try:
            self._stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.broken = True

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:                                      # noqa: BLE001
            return False

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _NoFps:
    """Stand-in so the per-camera rate still reaches the scheduler when a camera
    has no embedder (ReID disabled)."""

    def set_fps(self, fps):
        pass


class LivePipeline:
    def __init__(self, sources, cfg):
        self.sources = sources            # [(cam_name, url_or_path), ...]  any N
        self.cfg = cfg
        self.live_cfg = cfg.get("live", {}) or {}
        self.stop_event = threading.Event()
        self.threads = []                 # (name, thread) in start order
        self.captures = []                # CaptureThread refs (for finished/reconnect)
        self.writers = []                 # WriterStage refs (finalize on shutdown)
        self.renderers = []               # RenderStage refs (metrics)
        # metrics wiring (populated in run(); read by the reporter)
        self._m = {}                      # {slots, inference_queue, identity_queue,
                                          #  render_queues, writer_queues, scheduler,
                                          #  inference, identity, max_batch}
        self._peaks = {"inference_q": 0, "identity_q": 0, "writer_q": 0}
        self._t_start = None
        # ---- offline-reconcile flow (live.reconcile) ----
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._recon_cfg = (self.cfg.get("identity", {}) or {}).get("reconcile", {}) or {}
        live_recon = self.live_cfg.get("reconcile", {}) or {}
        self.reconcile_enabled = bool(live_recon.get("enabled", False))
        self._sample_stride = int(live_recon.get("sample_stride", 1) or 1)
        self._keep_frames = bool(live_recon.get("keep_frames", False))
        # #65: a per-camera ceiling on the transient clip. 0 = unlimited (the old
        # behaviour). Measured growth is ~1.2-2.2 MB/s per camera, so four cameras
        # fill ~21 GB/hour with nothing stopping them.
        self._clip_cap_bytes = int(float(live_recon.get("max_clip_gb", 0) or 0) * 1e9)
        self.store = None                 # built in run() when reconcile is on
        self._clip_paths = {}             # {cam: transient processed-frame clip}
        self.identity_stage = None        # ref for the final store-count report

    # ---- config helpers ----------------------------------------------------
    def _g(self, group, key, default):
        return (self.live_cfg.get(group, {}) or {}).get(key, default)

    def _build_topology(self):
        """Build the cross-camera veto from the live.topology config block.
        Returns a GraphTopology when enabled with edges, else FailOpenTopology
        (the fail-open invariant: no data -> nothing is ever blocked)."""
        tcfg = self.live_cfg.get("topology", {}) or {}
        edges = tcfg.get("edges") or []
        if not tcfg.get("enabled", False) or not edges:
            print("[live] cross-camera topology veto: OFF (appearance-only; intended, "
                  "not an error -- enable live.topology with measured transit times to use it).")
            return FailOpenTopology()
        parsed = []
        for e in edges:
            try:
                a, b, sec = e[0], e[1], float(e[2])
                parsed.append((str(a), str(b), sec))
            except (TypeError, IndexError, ValueError):
                print(f"[live] topology: skipping malformed edge {e!r} "
                      f"(expected [cam_a, cam_b, min_transit_sec]).")
        if not parsed:
            return FailOpenTopology()
        topo = GraphTopology(edges=parsed)
        cams = ", ".join(sorted(topo.known()))
        print(f"[live] topology veto ACTIVE: {len(parsed)} edge(s) over "
              f"{len(topo.known())} camera(s) [{cams}]; any other camera is fail-open.")
        return topo

    def _build_geometry(self, cameras):
        """Build the GeometryRecorder from the top-level `geometry` config block.

        Returns None when geometry is off or no calibration exists -- and that is a
        normal, supported state: every consumer treats missing geometry as "no
        opinion" and the run behaves exactly as it did before geometry existed.

        THE LIVE RUN IS THE ONLY WRITER OF GEOMETRY. Offline reconcile reads the
        positions this records and never re-derives them (geometry/__init__.py
        invariant 1). So if this returns None, the run's reachability check is gone
        for good -- it cannot be recovered afterwards, because the raw feed is never
        recorded. That is why every failure below is LOUD.
        """
        gcfg = self.cfg.get("geometry", {}) or {}
        if not gcfg.get("enabled", False):
            print("[live] geometry: OFF (geometry.enabled is false) -> no floor "
                  "positions recorded, so offline reconcile has no reachability "
                  "check. Appearance-only, exactly as before.")
            return None
        try:
            from geometry.calibration import load_calibration
            from geometry.recorder import GeometryRecorder
            record = load_calibration(gcfg.get("calibration_path") or None)
        except Exception as e:                                     # noqa: BLE001
            print(f"[live] geometry: could NOT load the calibration ({e}) -> no "
                  f"floor positions will be recorded for this run.")
            return None
        if record is None:
            print("[live] geometry: enabled, but no calibration record exists -> "
                  "nothing will be recorded. Fit one from a completed run (no "
                  "camera time needed): python tools/fit_floor_frame.py <run_id>")
            return None

        log_path = (gcfg.get("log_path") or "logs/geometry_<run_id>.jsonl")
        log_path = log_path.replace("<run_id>", str(self.run_id))
        rec = GeometryRecorder(record, log_path=log_path, run_id=self.run_id)
        print("[live] geometry ACTIVE:")
        for line in record.summary().splitlines():
            print(f"[live]   {line}")
        missing = rec.cameras_without_geometry(cameras)
        if missing:
            # Named at startup, not inferred from a zero counter afterwards: a
            # camera silently contributing no geometry looks identical to geometry
            # being off, and that ambiguity is expensive to debug later.
            print(f"[live]   NOT calibrated (fail-open, appearance-only): "
                  f"{', '.join(sorted(missing))}")
        return rec

    def _build_store(self, dim: Optional[int] = None):
        """Build the Qdrant gallery for the offline reconcile, from the top-level
        `store` config (same backend the file path uses: env QDRANT_URL/API_KEY >
        store.url > store.path). Returns None (and disables reconcile) if the
        store is off or unreachable -- degrade, never crash.

        dim : vector size, taken from the loaded ReID model so the collection is
              always sized for the backbone actually running. None keeps the
              store's own default."""
        store_cfg = self.cfg.get("store", {}) or {}
        if not store_cfg.get("enabled", False):
            print("[live] store.enabled is false -> cannot persist observations; "
                  "offline reconcile disabled.")
            return None
        try:
            from database.store import PersonVectorStore
            url = os.environ.get("QDRANT_URL") or store_cfg.get("url") or None
            api_key = os.environ.get("QDRANT_API_KEY") or None
            path = store_cfg.get("path", "qdrant_data")
            store = PersonVectorStore(path=path, url=url, api_key=api_key,
                                      **({} if dim is None else {"dim": dim}))
            backend = url if url else f"LOCAL '{path}'"
            print(f"[live] gallery store ready at {backend} "
                  f"(existing points: {store.count()}).")
            return store
        except Exception as e:
            print(f"[live] could NOT open the gallery store ({e}); offline "
                  f"reconcile disabled -> falling back to live-annotated output.")
            return None

    def run(self):
        cap = capability_report(self._g("run", "device", "auto"),
                                self._g("capture", "nvdec", "auto"))
        device = cap["device"]
        print(f"[live] Starting LivePipeline on {len(self.sources)} source(s): "
              f"{', '.join(n for n, _ in self.sources)} | device={device}")

        # ---- shared model (one extractor; per-camera detectors/embedders) ----
        # Built BEFORE the store because the store's vector size is taken from
        # the model that actually loaded (extractor.embedding_dim), not from a
        # constant -- so swapping reid.model for a different-width backbone
        # cannot quietly create a mis-sized collection.
        reid_cfg = self.cfg.get("reid", {}) or {}
        det_cfg = self.cfg.get("detector", {}) or {}
        trk_cfg = self.cfg.get("tracker", {}) or {}
        extractor = ReIDExtractor(
            weights=reid_cfg["weights"], device=device,
            # #39/#56. The tap is recorded in the run banner because every
            # threshold in the config is specific to it.
            tap=reid_cfg.get("tap", "post_relu"),
            max_batch=int(reid_cfg.get("max_batch", 32)),
            # Which backbone -- same reason it is in the banner as the tap.
            model=reid_cfg.get("model"))
        print(f"[live] ReID model: {extractor.describe()}")
        print(f"[live] ReID tap: {reid_cfg.get('tap', 'post_relu')} "
              f"-- thresholds are model- and tap-specific; never compare score "
              f"logs across either.")

        # ---- offline-reconcile flow: build the gallery store (v5 correct-answer-
        # at-the-end). The live RTSP feed is NEVER recorded; we persist per-track
        # embeddings during the run and re-render from the captured processed
        # frames after Ctrl-C. Needs the store; if it can't be built we fall back
        # to the classic immediate live-annotated output (paced writer).
        if self.reconcile_enabled:
            self.store = self._build_store(dim=extractor.embedding_dim)
            if self.store is None:
                # #59: reconcile silently switching itself off produces a
                # live-annotated video that LOOKS like a normal result -- provisional
                # per-camera ids, no cross-camera identity, and nothing saying so.
                # An operator cannot tell that from a successful run.
                self.reconcile_enabled = False
                print("=" * 72)
                print("[live] WARNING: the gallery store is unavailable, so the "
                      "OFFLINE RECONCILE IS DISABLED.")
                print("[live] Output videos will carry PROVISIONAL per-camera ids "
                      "with NO cross-camera identity.")
                print("[live] That is almost certainly not what you want. Start "
                      "Qdrant (docker compose up -d) and re-run.")
                print("=" * 72)
        if self.reconcile_enabled:
            print(f"[live] offline reconcile ENABLED (run_id={self.run_id}): live "
                  f"reids are provisional; the CORRECT cross-camera ids are settled "
                  f"on Ctrl-C and re-rendered into output_<cam>.mp4.")
            print_stop_hint("live")
        else:
            print("[live] offline reconcile OFF: writing live-annotated output "
                  "immediately (reids are the online engine's, not reconciled).")
            print("[stop] To stop: press Ctrl-C once.")

        # LIVE-only pose toggle: the pose ensemble is a SECOND model per frame.
        # Skipping it ~halves detection cost -> far fewer dropped frames -> stabler
        # ByteTrack -> less identity fragmentation. Scoped to the live path only via
        # live.inference.pose_ensemble; the file-batch path (main.py) still reads
        # detector.pose_ensemble unchanged (regression gate intact).
        # #66: defaults FALSE. It used to default True, so DELETING one config
        # line silently enabled a second detection model per frame -- measured
        # to CREATE duplicate boxes (0 overlapping pairs without it, 2 with)
        # and to mint synthetic track ids whose "primary" flips between people.
        live_pose = self._g("inference", "pose_ensemble", False)
        pose_cfg = det_cfg.get("pose_ensemble") if live_pose else None
        if not live_pose:
            print("[live] pose ensemble DISABLED for the live path "
                  "(throughput: 1 detection model/frame instead of 2).")

        detectors, embedders = {}, {}
        for name, _ in self.sources:
            # Per-camera overrides merged over the global detector block. pose_cfg
            # stays as resolved above: the live pose toggle is a THROUGHPUT
            # decision for the whole path, not a per-camera one.
            cam_det_cfg = resolve_detector_cfg(det_cfg, name)
            detectors[name] = PersonDetector(
                model_path=cam_det_cfg["model"],
                confidence_threshold=cam_det_cfg["confidence_threshold"],
                person_class_id=cam_det_cfg["person_class_id"],
                tracker_config=trk_cfg.get("config", "bytetrack.yaml"),
                pose_ensemble=pose_cfg,
                iou=cam_det_cfg.get("iou", 0.7),
            )
            embedders[name] = TrackEmbedder(
                extractor,
                interval=reid_cfg.get("interval", 10),
                ttl=reid_cfg.get("ttl", 300),
                quality=reid_cfg.get("quality"),
                max_embeddings_per_track=reid_cfg.get("max_embeddings_per_track", 0),
                warmup_embeddings=reid_cfg.get("warmup_embeddings", 3),
                warmup_spacing=reid_cfg.get("warmup_spacing", 3),
                # #47: when set, every camera embeds at the same rate in TIME
                # instead of the same number of FRAMES.
                interval_sec=reid_cfg.get("interval_sec", 0.0),
            )

        # #65: check free space BEFORE a run, not after it fails. Four cameras
        # write ~5.9 MB/s of transient clip, and with keep_frames they are never
        # deleted -- so a long run on a small volume silently fills the disk.
        if self.reconcile_enabled:
            try:
                import shutil as _shutil
                free_gb = _shutil.disk_usage(".").free / 1e9
                need = 5.9e6 * 3600 / 1e9        # one hour, four cameras
                if free_gb < 5:
                    print(f"[live] WARNING: only {free_gb:.1f} GB free. Clips are "
                          f"written at ~5.9 MB/s across four cameras (~{need:.0f} "
                          f"GB/hour) and keep_frames="
                          f"{self._keep_frames}. Free space or set "
                          f"live.reconcile.max_clip_gb.")
                else:
                    print(f"[live] disk: {free_gb:.1f} GB free "
                          f"(~{need:.0f} GB/hour of clips at four cameras).")
            except Exception:                                   # noqa: BLE001
                pass

        # ---- WARM-UP before opening any source (v5 §12) ----------------------
        self._warmup(detectors, extractor)

        # ---- build queues + per-camera stages --------------------------------
        bs = int(self._g("inference", "max_batch_size", 8)) if device.startswith("cuda") else 1
        out_cfg = self.live_cfg.get("output", {}) or {}
        max_writer_q = int(out_cfg.get("max_writer_queue", 16))
        fps = float(out_cfg.get("fps_default", 20))

        slots, render_queues, writer_queues = {}, {}, {}
        inference_queue = DropOldestQueue(int(self._g("inference", "max_inference_queue", 2)))
        identity_queue = DropOldestQueue(int(self._g("identity", "max_queue", 64)))

        for name, url in self.sources:
            slot = NewestSlot()
            slots[name] = slot
            backend = make_decode_backend(
                url, capabilities=cap, stop_event=self.stop_event,
                reconnect_attempts=int(self._g("capture", "reconnect_attempts", 5)),
                reconnect_backoff=float(self._g("capture", "reconnect_base_delay", 1)),
            )
            capt = CaptureThread(
                name, backend, slot, self.stop_event,
                reconnect_base_delay=float(self._g("capture", "reconnect_base_delay", 1)),
                reconnect_max_delay=float(self._g("capture", "reconnect_max_delay", 30)),
                reconnect_attempts=int(self._g("capture", "reconnect_attempts", 5)),
                device=device,
                # Shifts this camera's MEDIA timeline so several concurrently
                # recorded files line up (frame.py, "two clocks"). Ignored for a
                # live stream, whose `ts` is already the event time. 0 is correct
                # when every recording was started together.
                time_offset_sec=float(
                    (self._g("capture", "file_time_offsets", {}) or {}).get(name, 0.0)),
            )
            self.captures.append(capt)

            render_q = DropOldestQueue(max_writer_q)
            render_queues[name] = render_q
            if self.reconcile_enabled:
                # Capture mode: the render stage writes CLEAN processed frames to a
                # transient clip + records box geometry; the final labelled video is
                # produced after the offline reconcile. No paced writer in this flow.
                clip_path = f"._live_src_{name}.mp4"
                self._clip_paths[name] = clip_path
                renderer = RenderStage(name, render_q, None, self.stop_event,
                                       capture_mode=True, clip_path=clip_path,
                                       clip_fps=fps, run_id=self.run_id)
                renderer.clip_bytes_cap = self._clip_cap_bytes
                self.renderers.append(renderer)
                self.threads.append((f"render-{name}", renderer))
            else:
                writer_q = DropOldestQueue(max_writer_q)
                writer_queues[name] = writer_q
                renderer = RenderStage(name, render_q, writer_q, self.stop_event)
                self.renderers.append(renderer)
                writer = WriterStage(
                    name, writer_q, self.stop_event,
                    out_path=f"output_{name}.mp4", fps=fps,
                    codec=out_cfg.get("codec", "h264"),
                    offline_after_sec=float(out_cfg.get("offline_overlay_after_sec", 3)),
                    reconnect_ref=(lambda c=capt: c.reconnects),
                )
                self.writers.append(writer)
                # register in start order: consumers first (writer, render), captures last
                self.threads.append((f"writer-{name}", writer))
                self.threads.append((f"render-{name}", renderer))

        scheduler = BatchScheduler(
            slots, inference_queue, self.stop_event,
            max_batch_size=bs,
            t_batch_ms=float(self._g("inference", "t_batch_ms", 15)),
            # #48: an absolute 100ms is 2.5 frame periods at 25 fps but only 1.5 at
            # 15 fps, so the same number is a different rule per camera -- the slow
            # camera's frames are called stale sooner in relative terms, which is
            # backwards. `max_frame_staleness_periods` expresses it in frame periods
            # and is converted per camera from the nominal rate.
            max_frame_staleness_ms=float(self._g("capture", "max_frame_staleness_ms", 100)),
            staleness_periods=float(self._g("capture",
                                            "max_frame_staleness_periods", 0) or 0),
        )
        inference = InferenceStage(detectors, embedders, inference_queue,
                                   identity_queue, self.stop_event,
                                   max_workers=int(self._g("inference", "max_workers", 0)))
        limits_cfg = self.live_cfg.get("limits", {}) or {}
        identity = IdentityStage(
            identity_queue, render_queues, self.stop_event,
            min_evidence_obs=int(self._g("identity", "min_evidence_obs", 3)),
            same_camera_threshold=float(self._g("identity", "same_camera_threshold", 0.90)),
            cross_camera_threshold=float(self._g("identity", "cross_camera_threshold", 0.63)),
            accept_margin=float(self._g("identity", "accept_margin", 0.03)),
            bank_size=int(self._g("identity", "bank_size", 20)),
            active_ttl_sec=float(limits_cfg.get("active_ttl", 300)),
            max_active_identities=int(limits_cfg.get("max_active_identities", 200)),
            max_per_lane=int(self._g("identity", "max_queue", 64)),
            # Cross-camera physical-impossibility veto, built from the
            # live.topology config block (fail-open if absent/disabled/empty).
            topology=self._build_topology(),
            # Offline-reconcile persistence (None store => no persistence).
            store=self.store if self.reconcile_enabled else None,
            run_id=self.run_id,
            sample_stride=self._sample_stride,
            # Records each observation's floor position INTO the payload, so the
            # offline reconcile consumes recorded geometry instead of recomputing it.
            geometry=self._build_geometry([n for n, _ in self.sources]),
        )
        self.identity_stage = identity
        # shared consumers start before captures
        self.threads.append(("identity", identity))
        self.threads.append(("inference", inference))
        self.threads.append(("scheduler", scheduler))
        for capt in self.captures:
            self.threads.append((capt.name, capt))

        # ---- stash refs for the metrics reporter ----------------------------
        self._m = {
            "slots": slots, "inference_queue": inference_queue,
            "identity_queue": identity_queue, "render_queues": render_queues,
            "writer_queues": writer_queues, "scheduler": scheduler,
            "inference": inference, "identity": identity, "max_batch": bs,
            "embedders": embedders,
        }

        # ---- start (consumers already ordered before captures) --------------
        # #61: the start loop used to sit OUTSIDE the try/finally, so an exception
        # while starting the Nth thread left the first N-1 running with nothing to
        # join them -- identity never drained and the run's observations were lost.
        started = []
        try:
            for _, t in self.threads:
                t.start()
                started.append(t)
        except BaseException:                                   # noqa: BLE001
            print(f"[live] FAILED while starting threads ({len(started)} of "
                  f"{len(self.threads)} up) -- stopping them cleanly.")
            self.stop_event.set()
            for t in started:
                try:
                    t.join(timeout=5)
                except Exception:                               # noqa: BLE001
                    pass
            raise

        # ---- supervise until stop -------------------------------------------
        max_dur = float(self._g("run", "max_duration_sec", 0) or 0)
        log_interval = float((self.live_cfg.get("metrics", {}) or {}).get("log_interval_sec", 10) or 0)
        self._t_start = time.monotonic()
        last_log = self._t_start
        try:
            while not self.stop_event.is_set():
                self._track_peaks()
                if all(c.finished for c in self.captures):
                    print("[live] all sources ended -> draining and finalizing...")
                    time.sleep(1.0)     # let in-flight frames flush through
                    break
                if max_dur > 0 and (time.monotonic() - self._t_start) >= max_dur:
                    print(f"[live] max_duration_sec={max_dur:g} reached -> stopping.")
                    break
                now = time.monotonic()
                if log_interval > 0 and (now - last_log) >= log_interval:
                    self._report(now - self._t_start, final=False)
                    last_log = now
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[live] Ctrl-C -> stopping and finalizing outputs. Do NOT "
                  "press Ctrl-C again: the reconciled cross-camera ids are "
                  "decided during this step.")
        finally:
            self._shutdown()

    def _warmup(self, detectors, extractor):
        """Run a dummy frame through detection + ReID so the first real frames
        don't pay model-init latency. Guarded; failure is non-fatal."""
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            for det in detectors.values():
                det.track(dummy)
            # Warm up at the backend's OWN input size, not a hardcoded 256x128.
            # Functionally either works (preprocess resizes anyway), but the point
            # of a warm-up is to compile the kernels the real frames will hit, and
            # those are shaped by input_size -- 384x128 for FastReID SBS.
            h, w = extractor.input_size
            extractor.extract_batch([np.zeros((h, w, 3), dtype=np.uint8)])
            print("[live] warm-up complete (detector + ReID initialised).")
        except Exception as e:
            print(f"[live] warm-up skipped ({e}).")

    def _shutdown(self):
        """Race-free shutdown: stop, join in pipeline order, writers finalize
        their MP4s in their own finally-blocks, then the offline reconcile +
        re-render produce the deliverable videos.

        The whole phase runs under InterruptGuard: a second Ctrl-C here would
        kill the writers mid-flush and skip the reconcile, so extra presses are
        warned about instead of raising (3 in a row still force-quit).

        InterruptGuard covers SIGNALS. It does NOT cover a dead stdout, which is a
        separate way to lose the same work -- see _QuietOnBrokenPipe. Both are
        needed: the guard stops Ctrl-C aborting finalization, the stream wrapper
        stops a failed `print` doing it."""
        prev_out, prev_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = (_QuietOnBrokenPipe(prev_out),
                                  _QuietOnBrokenPipe(prev_err))
        try:
            with InterruptGuard("finalizing outputs (flushing videos, then "
                                "reconciling ids + re-rendering)"):
                self._shutdown_inner()
        finally:
            sys.stdout, sys.stderr = prev_out, prev_err

    def _shutdown_inner(self):
        """The actual shutdown sequence (always called under InterruptGuard)."""
        self.stop_event.set()
        # Join in the order data flows so downstream flushes what upstream sent.
        order = ["scheduler", "inference", "identity"]
        by_name = {}
        for nm, t in self.threads:
            by_name.setdefault(nm, t)
        # captures first (stop producing), then core, then render, then writers
        for capt in self.captures:
            capt.join(timeout=5)
        for nm in order:
            t = by_name.get(nm)
            if t:
                t.join(timeout=5)
        for nm, t in self.threads:
            if nm.startswith("render-"):
                t.join(timeout=5)
        for w in self.writers:
            w.join(timeout=10)     # writer flushes queue + releases the MP4
        # #55 second half: a stage that died left an empty output with no
        # explanation anywhere. Name it here, where the operator is already looking.
        dead = [nm for nm, t in self.threads
                if getattr(t, "failed", None) is not None]
        if dead:
            print("=" * 72)
            print(f"[live] WARNING: {len(dead)} stage(s) DIED during this run: "
                  f"{sorted(set(dead))}")
            print("[live] Their output is missing or incomplete -- the traceback is "
                  "above, earlier in this log.")
            print("=" * 72)
        # Flush and close the geometry sidecar HERE, inside the guarded phase, for
        # the same reason the writers finalize here: an append-only log must not
        # lose its tail to a second Ctrl-C. Identity has already joined, so no
        # further records can arrive. The observation payloads in Qdrant -- which is
        # what reconcile actually reads -- were written as the run went, so a lost
        # sidecar tail costs analysis data, never identities.
        geo = getattr(self.identity_stage, "geometry", None) if self.identity_stage else None
        if geo is not None:
            try:
                geo.close()
                for line in geo.summary():
                    print(f"[live] {line}")
            except Exception as e:                                 # noqa: BLE001
                print(f"[live] geometry finalize failed ({type(e).__name__}: {e}); "
                      f"continuing -- the ids matter.")

        # Reporting is DIAGNOSTIC and must never be able to skip what follows it.
        # Guarded because it sits between the joins and the reconcile: any failure
        # here -- a dead stdout, a KeyError in the metrics dicts, a stage object
        # that never got built -- used to abandon the run's ids.
        if self._t_start is not None:
            try:
                self._report(time.monotonic() - self._t_start, final=True)
            except Exception as e:                                 # noqa: BLE001
                print(f"[live] final metrics report failed ({type(e).__name__}: {e}); "
                      f"continuing to reconcile -- the ids matter, the report does not.")
        # Offline reconcile runs AFTER every stage has joined: identity has flushed
        # all observations to the store, and each render stage has finalized its
        # temp clip -- so the whole-gallery view and the re-render source are both
        # complete and quiescent.
        if self.reconcile_enabled:
            self._finalize_offline()
        print("[live] shutdown complete.")

    def _finalize_offline(self):
        """The correct-answer-at-the-end step: rebuild cross-camera identities from
        the persisted observations with the PROVEN offline reconcile, then
        re-render output_<cam>.mp4 from the captured processed frames so the same
        person carries one id/colour everywhere. Reuses the file path's
        reconcile_tracklets / render_final_videos / build_gid_map / print_run_summary
        unchanged (regression gate intact)."""
        print("\n[live] Please wait a moment while we render the final outputs "
              "(offline reconciliation)...")
        stored = self.identity_stage.stored if self.identity_stage else 0
        print(f"[live] persisted {stored} observation(s) for run {self.run_id}.")
        if stored == 0:
            print("[live] no observations were persisted -> nothing to reconcile; "
                  "skipping final render.")
            self._cleanup_clips()
            return
        try:
            from identity.reconcile import (describe_reconcile_kwargs,
                                            reconcile_tracklets,
                                            resolve_reconcile_kwargs)
            # reuse the file path's render + summary helpers (unchanged).
            # #60: this used to be a bare `from main import ...`, which only
            # resolves when the process was started from the project root. Run the
            # pipeline from anywhere else and finalization raised ImportError AFTER
            # the cameras had stopped -- the run's whole output, lost to a path.
            # Put the project root on sys.path first, derived from this file.
            _root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from main import render_final_videos, print_run_summary
        except Exception as e:
            print(f"[live] could not load the offline reconcile helpers ({e}); "
                  f"leaving the captured clips in place.")
            return

        # EVERY merge setting comes from identity.reconcile.* through the ONE
        # resolver, so this path, main.py's and both offline calibration tools
        # cannot drift -- they had (see resolve_reconcile_kwargs).
        recon_kwargs = resolve_reconcile_kwargs(self.cfg)
        try:
            print("[live] reconciling identities across cameras...")
            print(f"[live] reconcile settings: "
                  f"{describe_reconcile_kwargs(recon_kwargs)}")
            reid_quality = {}
            reconcile_tracklets(
                self.store,
                run_id=self.run_id,
                # #34: per-tracklet fit/margin for the on-screen label.
                quality_out=reid_quality,
                **recon_kwargs,
                **self._decision_log_kwargs(),
            )
        except Exception as e:
            print(f"[live] reconcile failed ({e}); rendering with unreconciled ids.")

        # Re-render each camera from its captured processed-frame clip. A minimal
        # cfg keeps render_final_videos from re-resizing (box geometry was captured
        # at the clip's resolution) and sets the output fps.
        _out = (self.live_cfg.get("output", {}) or {})
        fps = float(_out.get("fps_default", 20))
        out_cfg_codec = _out.get("codec", "mp4v")
        render_cfg = {"source": {"resize_width": 0},
                      "display": {"output_fps": fps,
                                  # #63: the configured codec now reaches the
                                  # product path instead of being ignored.
                                  "codec": out_cfg_codec}}
        jobs = [(name, self._clip_paths[name]) for name, _ in self.sources
                if name in self._clip_paths]
        shared = {"annotations": {r.cam: r.annotations for r in self.renderers}}
        try:
            print("[live] rendering final annotated videos with reconciled ids...")
            # Each camera at ITS OWN measured rate (#45/#46), so the four videos
            # share one time scale and can be compared by eye. Falls back to the
            # configured default for any camera whose rate could not be measured.
            fps_by_camera = {}
            for r in self.renderers:
                measured = r.measured_fps()
                if measured and measured > 0:
                    fps_by_camera[r.cam] = measured
            if fps_by_camera:
                print("[live] output fps per camera: "
                      + ", ".join(f"{c}={v:.1f}"
                                  for c, v in sorted(fps_by_camera.items()))
                      + f"  (was a single global {fps:.1f} for all)")
            render_final_videos(jobs, render_cfg, shared, self.store, self.run_id,
                                fps_by_camera=fps_by_camera,
                                quality=reid_quality)
            print_run_summary(self.store,
                              [(name, path) for name, path in jobs],
                              {"display": {"save_annotated": True}},
                              run_id=self.run_id)
            render_ok = True
        except Exception as e:
            # #58: the cleanup used to run in a `finally`, so a failed render
            # DELETED the only copy of the footage it failed on -- destroying the
            # one artefact needed to diagnose or retry it.
            render_ok = False
            print(f"[live] final render/summary error: {e}")
            print(f"[live] KEEPING the processed-frame clips because the render "
                  f"failed -- they are the only copy. Re-render with: "
                  f"python tests/calibration/rerender_from_clips.py {self.run_id}")
        if render_ok:
            self._cleanup_clips()

    def _decision_log_kwargs(self):
        """Build the reconcile decision-log arguments from identity.reconcile.*.

        Guarded and fail-soft: if anything here is misconfigured we reconcile
        WITHOUT diagnostics rather than losing the run's ids. Diagnostics are worth
        a lot, but never worth the deliverable.
        """
        try:
            from identity.decision_log import DecisionLog
            margin_cfg = self._recon_cfg.get("top2_margin") or {}
            path = self._recon_cfg.get("decision_log")
            log = None
            if path:
                path = str(path).replace("<run_id>", self.run_id)
                log = DecisionLog(path=path, run_id=self.run_id)
            return {
                "decision_log": log,
                "top2_margin_threshold": margin_cfg.get("threshold"),
                "top2_margin_basis": margin_cfg.get("basis", "eligible"),
            }
        except Exception as e:                                    # noqa: BLE001
            print(f"[live] decision log disabled ({e}); reconciling without it.")
            return {}

    def _cleanup_clips(self):
        """Delete the transient processed-frame clips (unless keep_frames).

        The box-geometry sidecar travels WITH its clip: either both survive or
        both go. A clip without its geometry cannot be re-rendered, and geometry
        without its clip has nothing to draw on, so keeping one alone is only a
        confusing leftover.
        """
        pairs = [(p, os.path.splitext(p)[0] + ".annotations.json")
                 for p in self._clip_paths.values()]
        if self._keep_frames:
            for clip, annos in pairs:
                if os.path.exists(clip):
                    print(f"[live] keeping processed-frame clip: {clip}"
                          + (f" (+ {annos})" if os.path.exists(annos) else ""))
            print("[live] re-render these at other reconcile settings with: "
                  "python tests/calibration/rerender_from_clips.py "
                  f"{self.run_id}")
            return
        for clip, annos in pairs:
            for p in (clip, annos):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError as e:
                    print(f"[live] could not remove {p}: {e}")

    # ---- metrics -----------------------------------------------------------
    def _push_measured_fps(self, elapsed):
        """#47: tell each camera's embedder the rate that camera is really running
        at, so `reid.interval_sec` converts to the right frame count FOR THAT
        CAMERA. Without this the config key exists but has nothing to convert with,
        and the frame-count interval silently stays in force."""
        embedders = (self._m or {}).get("embedders") or {}
        if elapsed <= 0:
            return
        for capt in self.captures:
            emb = embedders.get(capt.cam)
            if emb is None or not hasattr(emb, "set_fps"):
                emb = _NoFps()
            fps = capt.frame_index / elapsed
            if fps > 0:
                emb.set_fps(fps)
                # #48: the scheduler needs it too, so staleness can be expressed in
                # each camera's own frame periods rather than one absolute ms bound.
                sched = (self._m or {}).get("scheduler")
                if sched is not None and hasattr(sched, "set_camera_fps"):
                    sched.set_camera_fps(capt.cam, fps)

    def _track_peaks(self):
        m = self._m
        if not m:
            return
        self._peaks["inference_q"] = max(self._peaks["inference_q"], len(m["inference_queue"]))
        self._peaks["identity_q"] = max(self._peaks["identity_q"], len(m["identity_queue"]))
        wq = max((len(q) for q in m["writer_queues"].values()), default=0)
        self._peaks["writer_q"] = max(self._peaks["writer_q"], wq)

    def _report(self, elapsed, final=False):
        """Print a metrics snapshot. The numbers that answer 'is it keeping up?':
        read-fps vs written-fps (the gap is dropped frames), the drop counters at
        each shedding point, and the scheduler's batch UTILISATION (how full each
        inference batch is -- low util on GPU = headroom Stage 2 batching would
        use). Identity counters show whether the engine is minting/reacquiring/
        linking sensibly."""
        m = self._m
        if not m or elapsed <= 0:
            return
        self._track_peaks()
        self._push_measured_fps(elapsed)
        sched = m["scheduler"]
        avg_batch = sched.dispatched_frames / max(1, sched.dispatched_batches)
        util = 100.0 * avg_batch / max(1, m["max_batch"])
        st = m["identity"].stats()
        head = "final SUMMARY" if final else f"t=+{elapsed:.0f}s"
        print(f"[live:metrics] {head}  (elapsed={elapsed:.1f}s, max_batch={m['max_batch']})")
        for capt in self.captures:
            cam = capt.cam
            rd = capt.frame_index
            rendered = next((r.rendered for r in self.renderers if r.cam == cam), 0)
            written = next((w.frames_written for w in self.writers if w.cam == cam), 0)
            slot = m["slots"].get(cam)
            rq = m["render_queues"].get(cam)
            wq = m["writer_queues"].get(cam)
            print(f"    {cam}: read={rd} ({rd / elapsed:.1f}fps)  rendered={rendered}  "
                  f"written={written} ({written / elapsed:.1f}fps)  "
                  f"slot_drop={slot.dropped if slot else 0}  "
                  f"rq_drop={rq.dropped if rq else 0}  wq_drop={wq.dropped if wq else 0}  "
                  f"reconnects={capt.reconnects}")
        print(f"    scheduler: batches={sched.dispatched_batches} frames={sched.dispatched_frames} "
              f"avg_batch={avg_batch:.2f} util={util:.0f}% stale_skipped={sched.stale_skipped}")
        print(f"    drops/peaks: infer_q={m['inference_queue'].dropped}/{self._peaks['inference_q']} "
              f"identity_in={m['identity_queue'].dropped}/{self._peaks['identity_q']} "
              f"identity_fair_drop={st['fair_dropped']} writer_q_peak={self._peaks['writer_q']} "
              f"| inference_done={m['inference'].frames_done}")
        print(f"    identity: minted={st['minted']} reacquired={st['reacquired']} "
              f"linked={st['linked']} active={st['active_identities']} "
              f"resolved_frames={st['frames_done']} stored={st.get('stored', 0)}")
        print(f"    x-camera: attempts={st['xcam_attempts']} linked={st['linked']} "
              f"rejected[thresh={st['xcam_rej_threshold']} margin={st['xcam_rej_margin']} "
              f"recip={st['xcam_rej_reciprocal']} topology={st['xcam_rej_topology']}] "
              f"max_subthreshold_score={st['xcam_max_subthreshold']:.3f}")
        print(f"    same-cam reacquire: attempts={st['recam_attempts']} "
              f"ok={st['reacquired']} rejected_below_thr={st['recam_rej_below']} "
              f"max_rejected_score={st['recam_max_rej']:.3f}  "
              f"(high rejected_below_thr => fragmentation; same_camera_threshold too strict)")
        print(f"    coactive_vetoes={st['coactive_vetoes']}  "
              f"(co-present same-camera false-merges prevented)   "
              f"topology_pruned={st['topology_pruned']}  "
              f"(impossible cross-camera candidates removed)")
        from live.identity_engine import HIST_LABELS
        lab = " ".join(HIST_LABELS)
        rh = " ".join(f"{n:>4}" for n in st['recam_hist'])
        xh = " ".join(f"{n:>4}" for n in st['xcam_hist'])
        print(f"    score histograms (best candidate per attempt)   bins: {lab}")
        print(f"      same-cam reacquire: {rh}   (pick same_camera_threshold below the same-person cluster)")
        print(f"      cross-camera:       {xh}   (pick cross_camera_threshold below the true-match cluster)")
