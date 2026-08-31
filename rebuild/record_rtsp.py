#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass
class Camera:
    name: str
    url: str
    proc: subprocess.Popen | None = None
    path: Path | None = None
    started: float | None = None


def parse(value: str) -> Camera:
    text = str(value)
    if "=" not in text:
        raise ValueError(f"Camera must be NAME=RTSP_URL: {text}")
    name, url = text.split("=", 1)
    name = name.strip()
    url = url.strip()
    if not name or not url:
        raise ValueError(f"Invalid camera source: {text}")
    return Camera(name, url)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    ap = argparse.ArgumentParser(description="Synchronized multi-camera RTSP recorder")
    ap.add_argument("--camera", action="append", required=True, help="NAME=RTSP_URL; repeat for each camera")
    ap.add_argument("--output", default="recordings", help="Output root directory")
    ap.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 means record until Ctrl+C")
    ap.add_argument("--probe", type=float, default=3.0, help="Seconds to wait for output growth")
    args = ap.parse_args()

    try:
        cams: List[Camera] = [parse(x) for x in args.camera]
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    names = [x.name for x in cams]
    if len(set(names)) != len(names):
        print("ERROR: duplicate camera names", file=sys.stderr)
        return 2

    root = Path(args.output).expanduser().resolve()
    session = root / f"session_{stamp()}"
    video = session / "videos"
    video.mkdir(parents=True, exist_ok=True)

    manifest = {
        "session": session.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cameras": [{"name": c.name} for c in cams],
        "start_time": None,
        "duration_requested_sec": args.duration,
    }
    (session / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 72)
    print("                 8-CAMERA RTSP RECORDER")
    print("=" * 72)
    print(f"Session: {session}")
    print("Starting all camera recorders...")

    for cam in cams:
        part = video / f"{cam.name}.mp4.part"
        cam.path = part
        log = session / f"{cam.name}.ffmpeg.log"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-i", cam.url,
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            "-movflags", "+faststart",
        ]
        if args.duration > 0:
            cmd += ["-t", str(args.duration)]
        cmd += [str(part)]
        handle = log.open("w", encoding="utf-8")
        try:
            cam.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=handle)
        except FileNotFoundError:
            handle.close()
            print("ERROR: ffmpeg is not installed or is not on PATH.", file=sys.stderr)
            for item in cams:
                if item.proc and item.proc.poll() is None:
                    item.proc.terminate()
            return 3
        cam.started = time.monotonic()
        handle.close()

    deadline = time.monotonic() + max(1.0, args.probe)
    ready = set()
    print("\nWaiting for every camera to produce video data...\n")

    while time.monotonic() < deadline and len(ready) < len(cams):
        for cam in cams:
            if cam.name in ready or cam.proc is None or cam.path is None:
                continue
            if cam.proc.poll() is not None:
                continue
            if cam.path.exists() and cam.path.stat().st_size > 0:
                ready.add(cam.name)
        text = "  ".join(f"{c.name}: {'RECORDING' if c.name in ready else 'CONNECTING'}" for c in cams)
        print("\r" + text + " " * 8, end="", flush=True)
        time.sleep(0.25)

    print()
    failed = [cam for cam in cams if cam.name not in ready or cam.proc is None or cam.proc.poll() is not None]
    if failed:
        print("\nFAILED TO START:")
        for cam in failed:
            print(f"  {cam.name}: FAILED — check {session / (cam.name + '.ffmpeg.log')}")
        print("\nStopping all cameras so this session is not partial.")
        stop(cams)
        return 4

    manifest["start_time"] = datetime.now(timezone.utc).isoformat()
    (session / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("                    8/8 CAMERAS RECORDING")
    print("=" * 72)
    print("Press Ctrl+C to stop all cameras together.")
    if args.duration > 0:
        print(f"Automatic stop after {args.duration:.0f} seconds.")

    started = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - started
            states = []
            alive = True
            for cam in cams:
                ok = cam.proc is not None and cam.proc.poll() is None
                alive = alive and ok
                size = cam.path.stat().st_size if cam.path and cam.path.exists() else 0
                states.append(f"{cam.name}: {'REC' if ok else 'STOP'} {size / 1048576:.1f}MB")
            print("\r" + "  ".join(states) + f"   |   {elapsed:7.1f}s" + " " * 12, end="", flush=True)
            if not alive:
                print("\n\nA recorder process stopped unexpectedly. Stopping the remaining cameras.", file=sys.stderr)
                break
            if args.duration > 0 and elapsed >= args.duration:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\nStopping all cameras...")

    stop(cams)

    complete = []
    for cam in cams:
        if cam.path and cam.path.exists():
            final = cam.path.with_suffix("")
            cam.path.replace(final)
            complete.append({"camera": cam.name, "video": str(final.relative_to(session)), "bytes": final.stat().st_size})

    manifest["stop_time"] = datetime.now(timezone.utc).isoformat()
    manifest["videos"] = complete
    (session / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n\n" + "=" * 72)
    print("                    RECORDING COMPLETE")
    print("=" * 72)
    for item in complete:
        print(f"{item['camera']}: {item['video']} ({item['bytes'] / 1048576:.1f} MB)")
    print(f"\nSession folder: {session}")
    print("Manifest:        " + str(session / "manifest.json"))
    return 0


def stop(cams: List[Camera]) -> None:
    for cam in cams:
        if cam.proc is None or cam.proc.poll() is not None:
            continue
        try:
            if cam.proc.stdin:
                cam.proc.stdin.write(b"q\n")
                cam.proc.stdin.flush()
        except Exception:
            pass
    end = time.monotonic() + 8.0
    for cam in cams:
        if cam.proc is None or cam.proc.poll() is not None:
            continue
        wait = max(0.1, end - time.monotonic())
        try:
            cam.proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            cam.proc.send_signal(signal.SIGINT)
            try:
                cam.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                cam.proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
