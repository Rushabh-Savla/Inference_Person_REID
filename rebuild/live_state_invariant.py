from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict

import cv2
import yaml

from rebuild.batch_state_invariant_joint_attributes import (
    BatchPipelineStateInvariantJointAttributes,
)


class Camera(threading.Thread):
    def __init__(self, name: str, source: str, out: Path, show: bool):
        super().__init__(daemon=True)
        self.name = name
        self.source = source
        self.out = out
        self.show = show
        self.stop_flag = threading.Event()
        self.error = None
        self.frames = 0
        self.fps = 20.0
        self.writer = None
        self.path = self.out / f"{self.name}.mkv"

    def stop(self):
        self.stop_flag.set()

    def run(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = RuntimeError(f"Cannot open RTSP source: {self.source}")
            return

        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            self.error = RuntimeError(f"Invalid stream dimensions for {self.name}")
            cap.release()
            return

        # Matroska is intentionally used for the live capture stage so a clean
        # or Ctrl-C stop does not depend on MP4 finalisation.
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
        if not self.writer.isOpened():
            # Broad OpenCV builds may not expose FFV1; fall back to MPEG-4 in MKV.
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
        if not self.writer.isOpened():
            self.error = RuntimeError(f"Cannot open recording writer for {self.name}")
            cap.release()
            return

        try:
            print(f"[live-state] {self.name}: recording {width}x{height} @ {self.fps:.2f} fps -> {self.path}")
            while not self.stop_flag.is_set():
                ok, image = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                self.writer.write(image)
                self.frames += 1
                if self.show:
                    view = image.copy()
                    cv2.putText(
                        view,
                        self.name,
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(self.name, view)
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.stop_flag.set()
                        break
        except Exception as exc:
            self.error = exc
        finally:
            cap.release()
            if self.writer is not None:
                self.writer.release()
            print(f"[live-state] {self.name}: captured {self.frames} frames")


def run_live_state(cfg_path: str, sources: Dict[str, str], output_dir: str | None, show: bool):
    if not sources:
        raise SystemExit("No live sources supplied")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(output_dir) if output_dir else Path("recordings")
    session = root / f"live_state_{stamp}"
    session.mkdir(parents=True, exist_ok=True)
    final = session / "rebuild_outputs"

    workers = [Camera(name, source, session, show) for name, source in sources.items()]
    print("[live-state] Safe055/V6 live mode: RECORD FIRST -> EXACT VIDEO PIPELINE RECONCILE")
    print("[live-state] The old FastReID/GlobalIdentityV4 live engine is not used.")
    print(f"[live-state] session: {session}")

    for worker in workers:
        worker.start()

    try:
        while any(worker.is_alive() for worker in workers):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[live-state] Ctrl-C received once: stopping capture, then running Safe055/V6 reconciliation.")
    finally:
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=10)
        if show:
            cv2.destroyAllWindows()

    errors = [worker.error for worker in workers if worker.error]
    if errors:
        raise RuntimeError("Live capture failure: " + "; ".join(map(str, errors)))

    paths = [str(worker.path) for worker in workers if worker.path.exists() and worker.frames > 0]
    if len(paths) != len(workers):
        raise RuntimeError("One or more live cameras produced no recording")

    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg.setdefault("input", {})["output_dir"] = str(final)

    with NamedTemporaryFile("w", suffix=".yaml", prefix="live_state_", delete=False) as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        temp = handle.name

    try:
        print("[live-state] PASS 1/2/3: running the exact Safe055/V6 joint video pipeline on the captured cameras")
        BatchPipelineStateInvariantJointAttributes(temp).run(paths)
    finally:
        Path(temp).unlink(missing_ok=True)

    print(f"[live-state] final Safe055/V6 outputs: {final}")
