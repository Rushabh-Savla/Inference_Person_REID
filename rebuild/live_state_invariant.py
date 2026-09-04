from __future__ import annotations

import json
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict

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
        self.process = None
        self.path = self.out / f"{self.name}.mp4"
        self.log = self.out / f"{self.name}.ffmpeg.log"

    def stop(self):
        self.stop_flag.set()

    @staticmethod
    def _probe(path: Path):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed for {path.name}: {result.stderr.strip()}")
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams", [])
        video = next((item for item in streams if item.get("codec_name")), None)
        duration = float((data.get("format") or {}).get("duration") or 0.0)
        if video is None or duration <= 0.0:
            raise RuntimeError(f"recording is not a valid MP4: {path}")
        return video, duration

    def _command(self, copy: bool, target: Path):
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-y",
            "-rtsp_transport",
            "tcp",
            "-rw_timeout",
            "5000000",
            "-i",
            self.source,
            "-map",
            "0:v:0",
            "-an",
        ]
        if copy:
            command += ["-c:v", "copy"]
        else:
            command += [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
            ]
        command += ["-movflags", "+faststart", str(target)]
        return command

    def _start(self, copy: bool, target: Path):
        logfile = self.log.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            self._command(copy, target),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=logfile,
        )
        logfile.close()

    def _interrupt(self):
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def run(self):
        if shutil.which("ffmpeg") is None:
            self.error = RuntimeError("ffmpeg is not installed or not in PATH")
            return
        if shutil.which("ffprobe") is None:
            self.error = RuntimeError("ffprobe is not installed or not in PATH")
            return

        self.path.unlink(missing_ok=True)
        copy_path = self.out / f".{self.name}.copy.mp4"
        copy_path.unlink(missing_ok=True)

        try:
            print(f"[live-state] {self.name}: starting MP4 capture -> {self.path}")
            self._start(True, copy_path)

            while not self.stop_flag.is_set():
                returncode = self.process.poll()
                if returncode is not None:
                    if copy_path.exists() and copy_path.stat().st_size > 0:
                        try:
                            self._probe(copy_path)
                            copy_path.replace(self.path)
                            break
                        except Exception:
                            pass

                    print(
                        f"[live-state] {self.name}: stream-copy capture exited with "
                        f"code {returncode}; retrying with H.264 MP4 encoding"
                    )
                    copy_path.unlink(missing_ok=True)
                    self._start(False, self.path)
                    while not self.stop_flag.is_set() and self.process.poll() is None:
                        time.sleep(0.5)
                    break
                time.sleep(0.5)

            if self.stop_flag.is_set():
                self._interrupt()

            if self.process is not None and self.process.poll() is not None:
                code = self.process.returncode
                if code not in (0, 255):
                    text = ""
                    try:
                        text = self.log.read_text(encoding="utf-8", errors="replace").strip()
                    except Exception:
                        pass
                    if not self.path.exists() or self.path.stat().st_size == 0:
                        self.error = RuntimeError(
                            f"{self.name}: ffmpeg exited with code {code}"
                            + (f": {text[-800:]}" if text else "")
                        )
                        return

            if not self.path.exists() or self.path.stat().st_size == 0:
                self.error = RuntimeError(f"{self.name}: no MP4 recording was produced")
                return

            video, duration = self._probe(self.path)
            rate = video.get("avg_frame_rate", "0/1")
            try:
                num, den = rate.split("/", 1)
                fps = float(num) / float(den) if float(den) else 0.0
            except Exception:
                fps = 0.0
            self.frames = int(max(1.0, duration * max(fps, 1.0)))
            print(
                f"[live-state] {self.name}: finalized MP4 "
                f"{video.get('width')}x{video.get('height')} @ {fps:.2f} fps "
                f"duration={duration:.2f}s -> {self.path}"
            )
        except Exception as exc:
            self.error = exc
            try:
                self._interrupt()
            except Exception:
                pass
        finally:
            copy_path.unlink(missing_ok=True)


def _manifest(session: Path, sources: Dict[str, str]):
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "mode": "safe055_v6_live_record_then_reconcile",
        "created_utc": now,
        "cameras": {
            name: {"source": source, "recording": str(session / f"{name}.mp4")}
            for name, source in sources.items()
        },
    }
    (session / "session.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def run_live_state(cfg_path: str, sources: Dict[str, str], output_dir: str | None, show: bool):
    if not sources:
        raise SystemExit("No live sources supplied")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("Live MP4 capture requires ffmpeg and ffprobe in PATH")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(output_dir) if output_dir else Path("recordings")
    session = root / f"live_state_{stamp}"
    session.mkdir(parents=True, exist_ok=True)
    final = session / "rebuild_outputs"
    _manifest(session, sources)

    workers = [Camera(name, source, session, show) for name, source in sources.items()]

    print("[live-state] Safe055/V6 LIVE")
    print("[live-state] RECORD ALL CAMERAS TO MP4 -> EXACT SAFE055/V6 JOINT RECONCILIATION")
    print("[live-state] Old FastReID/GlobalIdentityV4 live engine: DISABLED")
    print(f"[live-state] session: {session}")
    if show:
        print("[live-state] --show is headless-safe and is not used for capture; no Qt/OpenCV GUI is opened.")

    for worker in workers:
        worker.start()

    try:
        while any(worker.is_alive() for worker in workers):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[live-state] Ctrl-C received: finalizing all MP4 recordings before reconciliation.")
    finally:
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=25)

    errors = [worker.error for worker in workers if worker.error]
    if errors:
        raise RuntimeError("Live capture failure: " + "; ".join(map(str, errors)))

    paths = [str(worker.path) for worker in workers if worker.path.exists() and worker.path.stat().st_size > 0]
    if len(paths) != len(workers):
        raise RuntimeError("One or more live cameras produced no usable MP4")

    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg.setdefault("input", {})["output_dir"] = str(final)

    with NamedTemporaryFile(
        "w",
        suffix=".yaml",
        prefix="live_state_",
        delete=False,
    ) as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        temp = handle.name

    try:
        print("[live-state] PASS 1/2/3: exact Safe055/V6 joint video pipeline")
        BatchPipelineStateInvariantJointAttributes(temp).run(paths)
    finally:
        Path(temp).unlink(missing_ok=True)

    outputs = sorted(final.glob("*.mp4"))
    if not outputs:
        raise RuntimeError(f"Safe055/V6 completed but produced no MP4 outputs under {final}")

    print("[live-state] FINAL MP4 OUTPUTS:")
    for path in outputs:
        print(f"  {path}")
    print(f"[live-state] final Safe055/V6 outputs: {final}")
