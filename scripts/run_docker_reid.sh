#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VIDEO_ARGS=("$@")
if [[ "${#VIDEO_ARGS[@]}" -eq 0 ]]; then
  VIDEO_ARGS=(
    "input/_live_src_cam_219.mp4"
    "input/_live_src_cam_240_2.mp4"
    "input/_live_src_cam_240.mp4"
  )
fi

echo "[docker] starting persistent Qdrant"
docker compose up -d qdrant

echo "[docker] building DeepStream 8.0 ReID image"
docker compose build reid

echo "[docker] running strict runtime verification"
docker compose run --rm reid \
  python3 scripts/verify_reid_runtime.py \
  --video "${VIDEO_ARGS[0]}"

echo "[docker] running NvDCF multimodal ReID"
docker compose run --rm reid \
  python3 rebuild/run.py batch_state_final \
  --config rebuild/config_state_invariant.yaml \
  --videos "${VIDEO_ARGS[@]}"

echo "[docker] results: $ROOT/rebuild_outputs_nvdcf_strict_v4"
