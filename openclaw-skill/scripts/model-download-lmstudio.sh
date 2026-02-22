#!/usr/bin/env bash
# Download a USSI model directly through LM Studio (GGUF) when possible.
# Usage:
#   model-download-lmstudio.sh [--model MODEL_NAME_OR_REPO] [--quant Q4_K_M]
set -euo pipefail

MODEL=""
QUANT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|-m)
      MODEL="${2:-}"
      shift 2
      ;;
    --quant|-q)
      QUANT="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: model-download-lmstudio.sh [--model MODEL_NAME_OR_REPO] [--quant Q4_K_M]" >&2
      exit 1
      ;;
  esac
done

if ! command -v lms >/dev/null 2>&1; then
  echo "[ussi] ERROR: lms CLI not found." >&2
  echo "[ussi] Install LM Studio and bootstrap CLI: ~/.lmstudio/bin/lms bootstrap" >&2
  exit 1
fi

if [[ -z "${MODEL}" ]] && command -v ussi >/dev/null 2>&1; then
  MODELS_JSON="$(ussi models --json 2>/dev/null || true)"
  MODEL="$(python3 - <<'PY' "${MODELS_JSON}"
import json, sys
raw = sys.argv[1]
if not raw:
    print("")
    raise SystemExit(0)
try:
    data = json.loads(raw)
except Exception:
    print("")
    raise SystemExit(0)
entries = data if isinstance(data, list) else data.get("models", [])
if not isinstance(entries, list):
    entries = []
preferred = None
for e in entries:
    if isinstance(e, dict) and any(bool(e.get(k)) for k in ("is_current", "is_latest", "latest", "default")):
        preferred = e
        break
if preferred is None and entries and isinstance(entries[0], dict):
    preferred = entries[0]
if not isinstance(preferred, dict):
    print("")
    raise SystemExit(0)
for k in ("hf_repo", "huggingface_repo", "repo", "model_name", "id", "model_id"):
    v = preferred.get(k)
    if isinstance(v, str) and v.strip():
        print(v.strip())
        break
else:
    print("")
PY
)"
fi

if [[ -z "${MODEL}" ]]; then
  echo "[ussi] ERROR: no model name/repo available for LM Studio download." >&2
  echo "[ussi] Provide one explicitly, e.g.:" >&2
  echo "       bash scripts/model-download-lmstudio.sh --model lmstudio-community/Qwen2.5-7B-Instruct-GGUF --quant Q4_K_M" >&2
  echo "[ussi] Or use artifact download + import fallback:" >&2
  echo "       bash scripts/model-download.sh --lmstudio-import" >&2
  exit 1
fi

REQUEST="${MODEL}"
if [[ -n "${QUANT}" ]]; then
  REQUEST="${REQUEST}@${QUANT}"
fi

echo "[ussi] Downloading GGUF model via LM Studio: ${REQUEST}"
lms get "${REQUEST}" --gguf

echo "[ussi] Download complete. Local LM Studio models:"
lms ls

