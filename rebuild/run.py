from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rebuild.batch_v2 import BatchPipelineV2  # noqa: E402
from rebuild.live_v2 import run_live_v2  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Same-camera + cross-camera person ReID with a shared multi-view gallery")
    sub = parser.add_subparsers(dest="mode", required=True)

    batch = sub.add_parser("batch", help="recorded videos: detect -> track -> continuously collect features -> global gallery -> render")
    batch.add_argument("--config", default="rebuild/config.yaml")
    batch.add_argument("--videos", nargs="*", default=[])

    live = sub.add_parser("live", help="live/RTSP sources using the same online global gallery")
    live.add_argument("--config", default="rebuild/config.yaml")
    live.add_argument("--sources", nargs="+", required=True)
    live.add_argument("--output-dir", default=None)
    live.add_argument("--show", action="store_true")

    args = parser.parse_args()
    if args.mode == "batch":
        BatchPipelineV2(args.config).run(args.videos)
    else:
        run_live_v2(args.config, parse_source(args.sources), args.output_dir, args.show)


if __name__ == "__main__":
    main()
