from __future__ import annotations

import threading
import time
from collections import defaultdict

from live.identity_engine import IdentityEngine
from live.persistent_identity import PersistentIdentityEngine
from live.priority import CameraFairQueue
from live.topology import FailOpenTopology


class IdentityStage(threading.Thread):
    """Serial live identity stage using the proven V6 matcher plus durable state."""

    def __init__(self, identity_queue, render_queues, stop_event,
                 min_evidence_obs=3, same_camera_threshold=0.90,
                 cross_camera_threshold=0.63, accept_margin=0.03,
                 bank_size=20, active_ttl_sec=300.0, max_active_identities=200,
                 topology=None, max_per_lane=64, sweep_interval_sec=2.0,
                 metrics=None, store=None, run_id=None, sample_stride=1,
                 geometry=None, identity_state_path=None, identity_model_id=None):
        super().__init__(name="identity", daemon=True)
        self.in_q = identity_queue
        self.render_queues = render_queues
        self.stop_event = stop_event
        self.metrics = metrics
        self.sweep_interval = float(sweep_interval_sec)

        self._store = store
        self._run_id = run_id
        self._sample_stride = max(1, int(sample_stride))
        self._sample_count = defaultdict(int)
        self.stored = 0
        self._store_failed = False
        self.geometry = geometry

        # Persistent identity state is deliberately separate from the bounded
        # active-track cache. REID_IDENTITY_DB can point at a deployment-specific
        # durable path; REID_MODEL_ID prevents incompatible embedding spaces from
        # being mixed into one identity gallery.
        import os
        state_path = identity_state_path or os.environ.get(
            "REID_IDENTITY_DB", "identity_state/reid_live.sqlite3"
        )
        model_id = identity_model_id or os.environ.get("REID_MODEL_ID", "live")
        self.engine = PersistentIdentityEngine(
            min_evidence_obs=min_evidence_obs,
            same_camera_threshold=same_camera_threshold,
            cross_camera_threshold=cross_camera_threshold,
            accept_margin=accept_margin,
            bank_size=bank_size,
            active_ttl_sec=active_ttl_sec,
            max_active_identities=max_active_identities,
            topology=topology or FailOpenTopology(),
            state_path=state_path,
            model_id=model_id,
        )
        self.fair = CameraFairQueue(max_per_lane=max_per_lane)
        self.frames_done = 0
        self.failed = None

    def run(self):
        try:
            last_sweep = time.monotonic()
            while not self.stop_event.is_set():
                frame = self.in_q.get(timeout=0.1)
                if frame is not None:
                    self.fair.push(frame.cam, frame)
                    while True:
                        f = self.in_q.get_nowait()
                        if f is None:
                            break
                        self.fair.push(f.cam, f)

                while not self.stop_event.is_set():
                    f = self.fair.pop()
                    if f is None:
                        break
                    self._resolve_frame(f)
                    rq = self.render_queues.get(f.cam)
                    if rq is not None:
                        rq.put(f)
                    self.frames_done += 1

                now = time.monotonic()
                if now - last_sweep >= self.sweep_interval:
                    self.engine.sweep(time.time())
                    last_sweep = now
        except BaseException as e:
            import traceback
            self.failed = e
            print(
                f"[IdentityStage] FATAL: this stage has DIED ({type(e).__name__}: {e})."
            )
            traceback.print_exc()
            raise

    def _resolve_frame(self, frame):
        fresh = frame.meta.get("fresh_track_ids", set())
        store_vecs, store_payloads = [], []
        for det in frame.detections or []:
            if det.track_id is None:
                det.reid_id = None
                det.global_id = None
                det.reid_score = None
                continue

            has_fresh_emb = det.track_id in fresh and det.embedding is not None
            reid = self.engine.assign(
                frame.cam,
                det.track_id,
                det.embedding,
                det.crop_quality,
                frame.ts,
                has_fresh_emb,
            )
            det.reid_id = reid
            det.global_id = reid if reid is not None and reid > 0 else None
            det.reid_score = self.engine.score_for(frame.cam, det.track_id)

            if self._store is not None and not self._store_failed and has_fresh_emb:
                key = (frame.cam, det.track_id)
                self._sample_count[key] += 1
                if self._sample_count[key] % self._sample_stride == 0:
                    payload = self._observation_payload(frame, det)
                    self._record_geometry(frame, det, payload)
                    store_vecs.append(det.embedding)
                    store_payloads.append(payload)

        if store_vecs:
            try:
                self._store.add_many(store_vecs, store_payloads)
                self.stored += len(store_vecs)
            except Exception as e:
                self._store_failed = True
                print(f"[identity] store persistence DISABLED after error: {e}")

    def _observation_payload(self, frame, det):
        payload = {
            "camera": frame.cam,
            "track_id": int(det.track_id),
            "frame": int(frame.frame_index),
            "run_id": self._run_id,
            "ts": float(frame.event_ts()),
        }
        bbox = [getattr(det, a, None) for a in ("x1", "y1", "x2", "y2")]
        if all(v is not None for v in bbox):
            payload["bbox"] = [float(v) for v in bbox]
        conf = getattr(det, "confidence", None)
        if conf is not None:
            payload["confidence"] = float(conf)
        cq = getattr(det, "crop_quality", None)
        if isinstance(cq, dict):
            payload["crop_quality"] = cq
        elif cq is not None:
            payload["crop_quality"] = float(cq)
        return payload

    def _frame_size(self, frame):
        img = getattr(frame, "image", None)
        shape = getattr(img, "shape", None)
        if shape is None or len(shape) != 3:
            return None
        if shape[2] <= 4:
            return (int(shape[1]), int(shape[0]))
        if shape[0] <= 4:
            return (int(shape[2]), int(shape[1]))
        return None

    def _record_geometry(self, frame, det, payload):
        if self.geometry is None or not self.geometry.enabled:
            return
        bbox = payload.get("bbox")
        if bbox is None:
            return
        self.geometry.annotate(
            payload,
            frame.cam,
            bbox,
            frame.event_ts(),
            frame_index=frame.frame_index,
            track_id=det.track_id,
            frame_size=self._frame_size(frame),
        )

    @property
    def minted(self):
        return self.engine.minted

    def stats(self):
        e = self.engine
        geo = self.geometry.stats() if self.geometry is not None else {}
        return {
            **geo,
            "minted": e.minted,
            "reacquired": e.reacquired,
            "linked": e.linked,
            "active_identities": len(e.store.gids()),
            "persistent_identities": len(e.registry.gids()),
            "persistent_loaded": int(getattr(e, "persistent_loaded", 0)),
            "next_gid": int(e.registry.next_gid),
            "fair_dropped": self.fair.dropped,
            "frames_done": self.frames_done,
            "stored": self.stored,
            "xcam_attempts": e.xcam_attempts,
            "xcam_rej_threshold": e.xcam_rej_threshold,
            "xcam_rej_margin": e.xcam_rej_margin,
            "xcam_rej_reciprocal": e.xcam_rej_reciprocal,
            "xcam_rej_topology": e.xcam_rej_topology,
            "xcam_max_subthreshold": e.xcam_max_subthreshold,
            "recam_attempts": e.recam_attempts,
            "recam_rej_below": e.recam_rej_below,
            "recam_max_rej": e.recam_max_rej,
            "coactive_vetoes": e.coactive_vetoes,
            "topology_pruned": e.topology_pruned,
            "recam_hist": list(e.recam_hist),
            "xcam_hist": list(e.xcam_hist),
        }
