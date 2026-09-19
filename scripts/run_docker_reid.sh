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
QDRANT_HOST_PORT="${QDRANT_HOST_PORT:-}"
if [[ -z "$QDRANT_HOST_PORT" ]]; then
  for port in 6335 6337 6339 6341 6343; do
    if ! (echo >/dev/tcp/127.0.0.1/$port) >/dev/null 2>&1; then
      QDRANT_HOST_PORT="$port"
      break
    fi
  done
fi
if [[ -z "$QDRANT_HOST_PORT" ]]; then
  echo "[docker] ERROR: no free Qdrant host port found"
  exit 1
fi
export QDRANT_HOST_PORT
export QDRANT_GRPC_HOST_PORT="$((QDRANT_HOST_PORT + 1))"
echo "[docker] Qdrant host port: ${QDRANT_HOST_PORT} -> container 6333"
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
