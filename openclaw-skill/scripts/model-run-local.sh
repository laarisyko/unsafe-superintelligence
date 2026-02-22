#!/usr/bin/env bash
# Run local inference using a downloaded USSI model.
# Backends:
#   - transformers (directory or HF repo ID)
#   - llama.cpp (GGUF file + llama-cli)
# Usage:
#   model-run-local.sh --model-path PATH_OR_REPO --prompt "Hello"
set -euo pipefail

MODEL_PATH=""
PROMPT=""
MAX_TOKENS="128"
BACKEND="auto"  # auto|transformers|llama-cpp

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path|-m)
      MODEL_PATH="${2:-}"
      shift 2
      ;;
    --prompt|-p)
      PROMPT="${2:-}"
      shift 2
      ;;
    --max-tokens|-n)
      MAX_TOKENS="${2:-128}"
      shift 2
      ;;
    --backend|-b)
      BACKEND="${2:-auto}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: model-run-local.sh --model-path PATH_OR_REPO --prompt TEXT [--max-tokens N] [--backend auto|transformers|llama-cpp]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" || -z "${PROMPT}" ]]; then
  echo "Usage: model-run-local.sh --model-path PATH_OR_REPO --prompt TEXT [--max-tokens N] [--backend auto|transformers|llama-cpp]" >&2
  exit 1
fi

choose_backend() {
  local path="$1"
  local selected="${BACKEND}"
  if [[ "${selected}" != "auto" ]]; then
    echo "${selected}"
    return
  fi

  if [[ "${path}" == *.gguf ]] && command -v llama-cli >/dev/null 2>&1; then
    echo "llama-cpp"
    return
  fi

  if python3 - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("transformers")
importlib.import_module("torch")
PY
  then
    echo "transformers"
    return
  fi

  echo "unknown"
}

SELECTED_BACKEND="$(choose_backend "${MODEL_PATH}")"

if [[ "${SELECTED_BACKEND}" == "llama-cpp" ]]; then
  exec llama-cli -m "${MODEL_PATH}" -p "${PROMPT}" -n "${MAX_TOKENS}"
fi

if [[ "${SELECTED_BACKEND}" == "transformers" ]]; then
  python3 - <<'PY' "${MODEL_PATH}" "${PROMPT}" "${MAX_TOKENS}"
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1]
prompt = sys.argv[2]
max_tokens = int(sys.argv[3])

tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
if torch.cuda.is_available():
    model = model.cuda()

inputs = tok(prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

text = tok.decode(out[0], skip_special_tokens=True)
print(text)
PY
  exit 0
fi

echo "[ussi] ERROR: no supported local runtime found." >&2
echo "[ussi] Install one of:" >&2
echo "  - llama.cpp (llama-cli) and use a .gguf file" >&2
echo "  - python packages: torch + transformers" >&2
exit 1

