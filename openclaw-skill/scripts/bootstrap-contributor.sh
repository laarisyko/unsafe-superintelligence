#!/usr/bin/env bash
# One-command bootstrap for OpenClaw agents:
# install USSI, start node, and join as contributor (GPU if available, else CPU).
set -euo pipefail

bash "$(dirname "$0")/setup.sh"

DETECT_JSON="$(ussi detect --json || echo '{}')"

ACCELERATOR="$(python3 - <<'PY' "$DETECT_JSON"
import json, sys
try:
    data = json.loads(sys.argv[1])
except Exception:
    data = {}
acc = data.get("accelerator", "cpu")
print(acc if acc in {"cpu", "cuda", "rocm"} else "cpu")
PY
)"

GPU_MB="$(python3 - <<'PY' "$DETECT_JSON"
import json, sys
try:
    data = json.loads(sys.argv[1])
except Exception:
    data = {}
v = data.get("gpu_memory_mb", 0)
try:
    v = int(v)
except Exception:
    v = 0
print(max(v, 0))
PY
)"

echo "[ussi] Starting local node (accelerator=${ACCELERATOR}, gpu_memory_mb=${GPU_MB})..."
ussi node start --accelerator "${ACCELERATOR}" --gpu-memory-mb "${GPU_MB}" --json

if [ "${ACCELERATOR}" = "cuda" ] && [ "${GPU_MB}" -gt 0 ]; then
  GPU_GB=$((GPU_MB / 1024))
  if [ "${GPU_GB}" -lt 1 ]; then
    GPU_GB=1
  fi
  echo "[ussi] Joining as contributor with GPU (${GPU_GB}GB)..."
  ussi join --accelerator cuda --gpu-memory "${GPU_GB}GB" --json
else
  echo "[ussi] Joining as contributor with CPU..."
  ussi join --accelerator cpu --gpu-memory 0 --json
fi

echo "[ussi] Verifying status/quota..."
ussi status --json
ussi quota --json

