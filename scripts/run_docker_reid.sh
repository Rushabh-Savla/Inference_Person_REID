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

echo "[docker] selecting Qdrant endpoint"
ready() {
  curl -fsS --max-time 2 "$1/readyz" >/dev/null 2>&1
}
busy() {
  python3 - "$1" <<'PY'
import socket
import sys
s = socket.socket()
s.settimeout(0.4)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
raise SystemExit(0)
PY
}
if [[ -n "${QDRANT_URL:-}" ]]; then
  qdrant_url="$QDRANT_URL"
elif ready "http://127.0.0.1:6333"; then
  qdrant_url="http://127.0.0.1:6333"
else
  port="${QDRANT_PORT:-6333}"
  while busy "$port"; do
    port=$((port + 1))
  done
  export QDRANT_PORT="$port"
  qdrant_url="http://127.0.0.1:$port"
  export QDRANT_URL="$qdrant_url"
  echo "[docker] port 6333 is not a ready Qdrant; starting Qdrant on $qdrant_url"
  docker compose up -d qdrant
fi
export QDRANT_URL="$qdrant_url"

echo "[docker] waiting for Qdrant"
for attempt in $(seq 1 60); do
  if ready "$QDRANT_URL"; then
    break
  fi
  if [[ "$attempt" == "60" ]]; then
    echo "[docker] ERROR: Qdrant is not ready at $QDRANT_URL"
    exit 1
  fi
  sleep 1
done

echo "[docker] building DeepStream 8.0 ReID image"
docker compose build reid

echo "[docker] running strict runtime verification"
docker compose run --no-deps --rm -e QDRANT_URL="$QDRANT_URL" reid \
  python3 scripts/verify_reid_runtime.py \
  --video "${VIDEO_ARGS[0]}"

echo "[docker] running NvDCF multimodal ReID"
docker compose run --no-deps --rm -e QDRANT_URL="$QDRANT_URL" reid \
  python3 rebuild/run.py batch_state_final \
  --config rebuild/config_state_invariant.yaml \
  --videos "${VIDEO_ARGS[@]}"

echo "[docker] results: $ROOT/rebuild_outputs_nvdcf_strict_v4"
