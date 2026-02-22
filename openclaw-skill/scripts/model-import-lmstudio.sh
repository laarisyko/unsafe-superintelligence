#!/usr/bin/env bash
# Import a GGUF model artifact into LM Studio.
# Usage:
#   model-import-lmstudio.sh --model-file /path/to/model.gguf [--copy|--hard-link|--symbolic-link] [--user-repo author/repo]
set -euo pipefail

MODEL_FILE=""
MODE="--copy"
USER_REPO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-file|-m)
      MODEL_FILE="${2:-}"
      shift 2
      ;;
    --copy|--hard-link|--symbolic-link)
      MODE="$1"
      shift 1
      ;;
    --user-repo)
      USER_REPO="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: model-import-lmstudio.sh --model-file /path/to/model.gguf [--copy|--hard-link|--symbolic-link] [--user-repo author/repo]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_FILE}" ]]; then
  echo "Usage: model-import-lmstudio.sh --model-file /path/to/model.gguf [--copy|--hard-link|--symbolic-link] [--user-repo author/repo]" >&2
  exit 1
fi

if [[ "${MODEL_FILE,,}" != *.gguf ]]; then
  echo "[ussi] ERROR: LM Studio import expects a .gguf file: ${MODEL_FILE}" >&2
  exit 1
fi

if ! command -v lms >/dev/null 2>&1; then
  echo "[ussi] ERROR: lms CLI not found." >&2
  echo "[ussi] Install LM Studio and bootstrap CLI (run once):" >&2
  echo "       ~/.lmstudio/bin/lms bootstrap" >&2
  exit 1
fi

CMD=(lms import "${MODEL_FILE}" -y "${MODE}")
if [[ -n "${USER_REPO}" ]]; then
  CMD+=(--user-repo "${USER_REPO}")
fi

echo "[ussi] Importing model into LM Studio..."
"${CMD[@]}"

echo "[ussi] Import complete. Confirm with:"
echo "       lms ls"

