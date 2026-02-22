#!/usr/bin/env bash
# Download the current (latest) USSI model artifact for local execution.
# Defaults to GGUF so the artifact is compatible with LM Studio / llama.cpp.
# Usage:
#   model-download.sh [--model MODEL_ID] [--output-dir DIR] [--url DIRECT_URL]
#                     [--format gguf|mlx|auto] [--quant Q4_K_M]
#                     [--lmstudio-import]
set -euo pipefail

MODEL_ID=""
OUTPUT_DIR="${HOME}/.ussi/models"
DIRECT_URL=""
MODEL_FORMAT="gguf"
QUANT=""
LMSTUDIO_IMPORT="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|-m)
      MODEL_ID="${2:-}"
      shift 2
      ;;
    --output-dir|-o)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --url|-u)
      DIRECT_URL="${2:-}"
      shift 2
      ;;
    --format|-f)
      MODEL_FORMAT="${2:-gguf}"
      shift 2
      ;;
    --quant|-q)
      QUANT="${2:-}"
      shift 2
      ;;
    --lmstudio-import)
      LMSTUDIO_IMPORT="1"
      shift 1
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: model-download.sh [--model MODEL_ID] [--output-dir DIR] [--url DIRECT_URL] [--format gguf|mlx|auto] [--quant Q4_K_M] [--lmstudio-import]" >&2
      exit 1
      ;;
  esac
done

if [[ "${MODEL_FORMAT}" != "gguf" && "${MODEL_FORMAT}" != "mlx" && "${MODEL_FORMAT}" != "auto" ]]; then
  echo "[ussi] ERROR: --format must be one of: gguf, mlx, auto" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

URL="${DIRECT_URL}"
SELECTED_MODEL="${MODEL_ID}"

if [[ -z "${URL}" ]]; then
  if ! command -v ussi >/dev/null 2>&1; then
    echo "[ussi] ERROR: ussi CLI not found. Install first with scripts/setup.sh" >&2
    exit 1
  fi

  MODELS_JSON="$(ussi models --json 2>/dev/null || true)"
  if [[ -z "${MODELS_JSON}" ]]; then
    echo "[ussi] ERROR: unable to query model registry via 'ussi models --json'." >&2
    echo "[ussi] Provide a direct artifact URL with --url." >&2
    exit 1
  fi

  PARSED="$(python3 - <<'PY' "${MODELS_JSON}" "${MODEL_ID}" "${MODEL_FORMAT}" "${QUANT}"
import json, sys

def candidate_urls(entry):
    urls = []
    for k in ("download_url", "artifact_url", "url", "weights_url", "checkpoint_url"):
        v = entry.get(k)
        if isinstance(v, str) and v:
            urls.append(v)

    for k in ("files", "artifacts", "variants"):
        arr = entry.get(k)
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, str):
                    urls.append(item)
                elif isinstance(item, dict):
                    for kk in ("url", "download_url", "artifact_url"):
                        vv = item.get(kk)
                        if isinstance(vv, str) and vv:
                            urls.append(vv)
    return urls

def score_url(url, wanted_format, wanted_quant):
    u = url.lower()
    score = 0
    if wanted_format == "gguf":
        if ".gguf" in u:
            score += 100
        else:
            score -= 50
    elif wanted_format == "mlx":
        if "mlx" in u:
            score += 100
        else:
            score -= 20
    else:
        if ".gguf" in u:
            score += 20

    if wanted_quant:
        q = wanted_quant.lower()
        if q in u:
            score += 30
        else:
            score -= 5

    if "q4_k_m" in u:
        score += 5
    return score

def best_url(entry, wanted_format, wanted_quant):
    urls = candidate_urls(entry)
    if not urls:
        return ""
    ranked = sorted(urls, key=lambda u: score_url(u, wanted_format, wanted_quant), reverse=True)
    return ranked[0]

raw = sys.argv[1]
wanted = sys.argv[2].strip()
wanted_format = sys.argv[3].strip().lower()
wanted_quant = sys.argv[4].strip()

try:
    data = json.loads(raw)
except Exception:
    print("|")
    sys.exit(0)

entries = data if isinstance(data, list) else data.get("models", [])
if not isinstance(entries, list):
    entries = []

chosen = None

if wanted:
    for e in entries:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("model_id", e.get("id", e.get("name", ""))))
        if eid == wanted:
            chosen = e
            break

if chosen is None:
    for e in entries:
        if isinstance(e, dict) and any(bool(e.get(k)) for k in ("is_current", "is_latest", "latest", "default")):
            chosen = e
            break

if chosen is None and entries:
    first = entries[0]
    if isinstance(first, dict):
        chosen = first

if chosen is None:
    print("|")
    sys.exit(0)

mid = str(chosen.get("model_id", chosen.get("id", chosen.get("name", ""))))
url = best_url(chosen, wanted_format, wanted_quant)
print(f"{mid}|{url}")
PY
)"

  SELECTED_MODEL="${PARSED%%|*}"
  URL="${PARSED#*|}"
fi

if [[ -z "${URL}" ]]; then
  echo "[ussi] ERROR: no download URL found for model '${SELECTED_MODEL:-<auto>}'." >&2
  echo "[ussi] Pass a direct URL: model-download.sh --url https://.../model.gguf" >&2
  exit 1
fi

if [[ "${MODEL_FORMAT}" = "gguf" ]]; then
  URL_LC="${URL,,}"
  case "${URL_LC}" in
    *.gguf|*.zip|*.tgz|*.tar.gz)
      ;;
    *)
      echo "[ussi] ERROR: selected artifact is not GGUF or a supported archive: ${URL}" >&2
      echo "[ussi] For LM Studio compatibility, use a GGUF URL or archive containing GGUF." >&2
      exit 1
      ;;
  esac
fi

if [[ -z "${SELECTED_MODEL}" ]]; then
  SELECTED_MODEL="current"
fi

FILE_NAME="$(basename "${URL}")"
DEST_FILE="${OUTPUT_DIR}/${FILE_NAME}"
MODEL_DIR="${OUTPUT_DIR}/${SELECTED_MODEL}"
mkdir -p "${MODEL_DIR}"

echo "[ussi] Downloading model artifact..."
if command -v curl >/dev/null 2>&1; then
  curl -fL "${URL}" -o "${DEST_FILE}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${DEST_FILE}" "${URL}"
else
  echo "[ussi] ERROR: neither curl nor wget is available." >&2
  exit 1
fi

EXTRACTED_PATH="${DEST_FILE}"
case "${DEST_FILE}" in
  *.tar.gz|*.tgz)
    tar -xzf "${DEST_FILE}" -C "${MODEL_DIR}"
    EXTRACTED_PATH="${MODEL_DIR}"
    ;;
  *.zip)
    if command -v unzip >/dev/null 2>&1; then
      unzip -o "${DEST_FILE}" -d "${MODEL_DIR}" >/dev/null
      EXTRACTED_PATH="${MODEL_DIR}"
    else
      echo "[ussi] WARNING: unzip not found; keeping zip as-is at ${DEST_FILE}" >&2
    fi
    ;;
esac

if [[ "${MODEL_FORMAT}" = "gguf" ]]; then
  FOUND_GGUF=""
  if [[ "${EXTRACTED_PATH}" == *.gguf ]]; then
    FOUND_GGUF="${EXTRACTED_PATH}"
  elif [[ -d "${EXTRACTED_PATH}" ]]; then
    FOUND_GGUF="$(find "${EXTRACTED_PATH}" -type f -name '*.gguf' | head -n 1 || true)"
  fi

  if [[ -z "${FOUND_GGUF}" ]]; then
    echo "[ussi] ERROR: no .gguf file found in downloaded artifact." >&2
    echo "[ussi] Use --url with a GGUF file (or archive containing GGUF)." >&2
    exit 1
  fi
fi

python3 - <<'PY' "${SELECTED_MODEL}" "${URL}" "${DEST_FILE}" "${EXTRACTED_PATH}"
import json, sys
print(json.dumps({
  "status": "downloaded",
  "model_id": sys.argv[1],
  "url": sys.argv[2],
  "artifact_path": sys.argv[3],
  "local_model_path": sys.argv[4],
}, indent=2))
PY

if [[ "${LMSTUDIO_IMPORT}" = "1" ]]; then
  GGUF_PATH=""
  if [[ "${EXTRACTED_PATH}" == *.gguf ]]; then
    GGUF_PATH="${EXTRACTED_PATH}"
  elif [[ -d "${EXTRACTED_PATH}" ]]; then
    GGUF_PATH="$(find "${EXTRACTED_PATH}" -type f -name '*.gguf' | head -n 1 || true)"
  fi

  if [[ -z "${GGUF_PATH}" ]]; then
    echo "[ussi] ERROR: --lmstudio-import requested, but no .gguf file was found." >&2
    exit 1
  fi

  if command -v lms >/dev/null 2>&1; then
    echo "[ussi] Importing GGUF into LM Studio via lms..."
    lms import "${GGUF_PATH}" -y --copy
  else
    echo "[ussi] WARNING: lms CLI not found; cannot auto-import." >&2
    echo "[ussi] Install LM Studio CLI and run: lms import \"${GGUF_PATH}\" --copy" >&2
  fi
fi
