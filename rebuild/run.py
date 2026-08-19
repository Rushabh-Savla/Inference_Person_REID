from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rebuild.batch import BatchPipeline  # noqa: E402
from rebuild.live import run_live  # noqa: E402


def parse_source(values):
    sources = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("Live sources must use CAMERA=SOURCE, e.g. cam_a=rtsp://...")
        camera, source = value.split("=", 1)
        if not camera or not source:
            raise SystemExit(f"Invalid live source: {value}")
        sources[camera] = source
    return sources


def main():
    parser = argparse.ArgumentParser(
        description="Clean same-camera + cross-camera person ReID system"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    batch = sub.add_parser("batch", help="recorded videos: detect -> tracklets -> reconcile -> render")
    batch.add_argument("--config", default="rebuild/config.yaml")
    batch.add_argument("--videos", nargs="*", default=[])

    live = sub.add_parser("live", help="live/RTSP sources using the same core pipeline")
    live.add_argument("--config", default="rebuild/config.yaml")
    live.add_argument("--sources", nargs="+", required=True)
    live.add_argument("--output-dir", default=None)
    live.add_argument("--show", action="store_true")

    args = parser.parse_args()

    if args.mode == "batch":
        BatchPipeline(args.config).run(args.videos)
    else:
        run_live(
            args.config,
            parse_source(args.sources),
            args.output_dir,
            args.show,
        )


if __name__ == "__main__":
    main()
