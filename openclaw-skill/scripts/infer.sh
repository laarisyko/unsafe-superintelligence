#!/usr/bin/env bash
# Run inference on a model via the USSI network.
# Usage: infer.sh --model llama-7b --prompt "Your prompt"
set -euo pipefail
ussi infer "$@" --json
