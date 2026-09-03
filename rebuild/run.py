from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from rebuild.batch_v2 import BatchPipelineV2  # noqa: E402
from rebuild.batch_v3 import BatchPipelineV3  # noqa: E402
from rebuild.batch_v4 import BatchPipelineV4  # noqa: E402
from rebuild.batch_v5 import BatchPipelineV5  # noqa: E402
from rebuild.batch_v6 import BatchPipelineV6  # noqa: E402
from rebuild.batch_v6_local_global import BatchPipelineV6LocalGlobal  # noqa: E402
from rebuild.batch_multimodel import BatchPipelineMultiModel  # noqa: E402
from rebuild import batch_state_invariant as state_pipeline  # noqa: E402
from rebuild.multimodel_state_invariant_fast import StateInvariantFinalResolverFast  # noqa: E402

# The final state-invariant entry point uses the optimized resolver while all
# extraction, tracking, model weights, thresholds and rendering remain unchanged.
state_pipeline.StateInvariantFinalResolver = StateInvariantFinalResolverFast
from rebuild.batch_state_invariant_safe_overlap import BatchPipelineStateInvariantSafeOverlap  # noqa: E402
from rebuild.live_v4 import run_live_v4  # noqa: E402
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
    parser = argparse.ArgumentParser(
        description="Same-camera + cross-camera person ReID with persistent track-state and global identity memory"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    batch = sub.add_parser("batch", help="V6 recorded-video pipeline")
    batch.add_argument("--config", default="rebuild/config_v6.yaml")
    batch.add_argument("--videos", nargs="*", default=[])

    final_batch = sub.add_parser(
        "batch_final",
        help="Final multimodel cross-camera pipeline: NVIDIA ResNet50 + NVIDIA Swin Base + SOLIDER",
    )
    final_batch.add_argument("--config", default="rebuild/config_final_multimodel.yaml")
    final_batch.add_argument("--videos", nargs="*", default=[])

    state_final = sub.add_parser(
        "batch_state_final",
        help="State-invariant final ReID: ResNet + NVIDIA Swin + SOLIDER across full/upper/torso/lower views",
    )
    state_final.add_argument("--config", default="rebuild/config_state_invariant.yaml")
    state_final.add_argument("--videos", nargs="*", default=[])

    local_global = sub.add_parser(
        "batch_local_global",
        help="V6: independent camera-local identity solving followed by global cross-camera reconciliation",
    )
    local_global.add_argument("--config", default="rebuild/config_v6_local_global.yaml")
    local_global.add_argument("--videos", nargs="*", default=[])

    old5 = sub.add_parser("batch_v5", help="V5 adaptive recorded-video pipeline")
    old5.add_argument("--config", default="rebuild/config_v5.yaml")
    old5.add_argument("--videos", nargs="*", default=[])

    old4 = sub.add_parser("batch_v4", help="V4 multimodal baseline pipeline")
    old4.add_argument("--config", default="rebuild/config_v4.yaml")
    old4.add_argument("--videos", nargs="*", default=[])

    old3 = sub.add_parser("batch_v3", help="V3 body baseline pipeline")
    old3.add_argument("--config", default="rebuild/config_v3.yaml")
    old3.add_argument("--videos", nargs="*", default=[])

    old = sub.add_parser("batch_v2", help="V2 baseline pipeline")
    old.add_argument("--config", default="rebuild/config.yaml")
    old.add_argument("--videos", nargs="*", default=[])

    live = sub.add_parser("live", help="V4 live/RTSP sources")
    live.add_argument("--config", default="rebuild/config_v4.yaml")
    live.add_argument("--sources", nargs="+", required=True)
    live.add_argument("--output-dir", default=None)
    live.add_argument("--show", action="store_true")

    oldlive = sub.add_parser("live_v2", help="V2 live/RTSP baseline")
    oldlive.add_argument("--config", default="rebuild/config.yaml")
    oldlive.add_argument("--sources", nargs="+", required=True)
    oldlive.add_argument("--output-dir", default=None)
    oldlive.add_argument("--show", action="store_true")

    args = parser.parse_args()
    if args.mode == "batch":
        BatchPipelineV6(args.config).run(args.videos)
    elif args.mode == "batch_final":
        BatchPipelineMultiModel(args.config).run(args.videos)
    elif args.mode == "batch_state_final":
        BatchPipelineStateInvariantSafeOverlap(args.config).run(args.videos)
    elif args.mode == "batch_local_global":
        BatchPipelineV6LocalGlobal(args.config).run(args.videos)
    elif args.mode == "batch_v5":
        BatchPipelineV5(args.config).run(args.videos)
    elif args.mode == "batch_v4":
        BatchPipelineV4(args.config).run(args.videos)
    elif args.mode == "batch_v3":
        BatchPipelineV3(args.config).run(args.videos)
    elif args.mode == "batch_v2":
        BatchPipelineV2(args.config).run(args.videos)
    elif args.mode == "live":
        run_live_v4(args.config, parse_source(args.sources), args.output_dir, args.show)
    else:
        run_live_v2(args.config, parse_source(args.sources), args.output_dir, args.show)


if __name__ == "__main__":
    main()
