"""
scheduler.py  --  STAGE: batch scheduler (Stage 1 = trivial; Stage 2 = fair).

Reads each active camera's NEWEST slot, keeps only frames that are still FRESH
(age from the wall-clock `ts` <= max_frame_staleness_ms -- so a frame that sat
in a slow decode is judged stale and skipped, v5 §5), and dispatches a batch to
the inference stage when it reaches `max_batch_size` OR `t_batch_ms` elapses.

Stage 1 keeps the policy trivial (grab fresh frames, dispatch; bs is usually 1
on CPU). It ALREADY emits a list ("batch"). It never waits for an individual
camera beyond one cycle -- a stalled camera simply contributes nothing that cycle
(no head-of-line blocking).

FAIRNESS: this scheduler holds NO rotation state, and deliberately so -- it is
fair by construction. `slot.get()` CLEARS the slot it reads, so a camera that was
just served has an empty slot on the next pass and the following camera wins even
at max_batch_size=1 (measured: 33.3/33.3/33.4% over 3 continuously-fresh cameras).
An earlier version of this docstring claimed a "round-robin, starvation bound"
that was never implemented, and config.yaml carried a matching `max_skip_cycles`
knob that nothing read; both are gone. Explicit round-robin DOES exist further
down the pipeline, in priority.py::CameraFairQueue, where the identity stage
drains one shared multi-camera queue and the ordering is not self-correcting.
"""

import threading
import time


class BatchScheduler(threading.Thread):
    def __init__(self, slots, inference_queue, stop_event,
                 max_batch_size=1, t_batch_ms=15, max_frame_staleness_ms=100,
                 cycle_sleep_ms=2, staleness_periods=0.0):
        super().__init__(name="scheduler", daemon=True)
        self.slots = slots                      # {cam: NewestSlot}
        self.inference_queue = inference_queue  # DropOldestQueue of batches (list[Frame])
        self.stop_event = stop_event
        self.max_batch_size = max(1, int(max_batch_size))
        # #48: a single absolute staleness bound is a DIFFERENT rule per camera --
        # 100ms is 2.5 frame periods at 25 fps but only 1.5 at 15 fps, so the slow
        # camera's frames are judged stale sooner in relative terms, which is
        # backwards (it has fewer frames to spare). When staleness_periods > 0 the
        # bound is computed per camera from its observed rate instead, so every
        # camera gets the same allowance measured in ITS OWN frames.
        self.staleness_periods = max(0.0, float(staleness_periods or 0.0))
        self._cam_fps = {}          # {cam: measured fps}, fed by set_camera_fps
        self.t_batch = float(t_batch_ms) / 1000.0
        self.max_staleness = float(max_frame_staleness_ms) / 1000.0
        self.cycle_sleep = float(cycle_sleep_ms) / 1000.0
        self.dispatched_batches = 0
        self.dispatched_frames = 0
        self.stale_skipped = 0

    def set_camera_fps(self, cam, fps):
        """#48: tell the scheduler a camera's measured rate so the staleness bound
        can be expressed in that camera's own frame periods."""
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            return
        if fps > 0:
            self._cam_fps[cam] = fps

    def _stale_ms(self, cam):
        """Staleness bound for THIS camera, in ms.

        Falls back to the absolute bound whenever periods are not configured or the
        camera's rate is not known yet -- so behaviour is unchanged until both are
        true, and a camera that never reports a rate is never treated differently.
        """
        if self.staleness_periods > 0:
            fps = self._cam_fps.get(cam)
            if fps:
                return 1000.0 * self.staleness_periods / fps
        return self.max_staleness * 1000.0

    failed = None

    def run(self):
        # #55: no worker had an exception guard. An unhandled error killed the
        # THREAD while the pipeline kept running -- a dead InferenceStage just
        # produces no detections, and a dead RenderStage writes nothing, with no
        # error anywhere. Record it loudly and set `failed` so shutdown can say
        # which stage died instead of leaving an empty output to explain.
        _FATAL_GUARD = True
        try:
            while not self.stop_event.is_set():
                batch = []
                deadline = time.monotonic() + self.t_batch
                # Accumulate fresh frames until the batch is full or the timer fires.
                while (len(batch) < self.max_batch_size
                       and time.monotonic() < deadline
                       and not self.stop_event.is_set()):
                    got_any = False
                    for cam, slot in self.slots.items():
                        frame = slot.get()
                        if frame is None:
                            continue
                        got_any = True
                        if frame.age_ms(time.time()) > self._stale_ms(cam):
                            self.stale_skipped += 1     # too old -> skip, don't process late
                            continue
                        batch.append(frame)
                        if len(batch) >= self.max_batch_size:
                            break
                    if not got_any:
                        time.sleep(self.cycle_sleep)     # nothing ready; yield briefly

                if batch:
                    self.inference_queue.put(batch)
                    self.dispatched_batches += 1
                    self.dispatched_frames += len(batch)
        except BaseException as e:                              # noqa: BLE001
            import traceback
            self.failed = e
            print(f"[BatchScheduler] FATAL: this stage has DIED ({type(e).__name__}: {e}). "
                  f"The run will continue but this stage produces nothing from now on.")
            traceback.print_exc()
            raise
