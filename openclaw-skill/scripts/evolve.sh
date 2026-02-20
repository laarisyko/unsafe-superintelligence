#!/usr/bin/env bash
# Propose an architecture mutation.
# Usage: evolve.sh --model llama-7b --mutation add_layer --position 3
set -euo pipefail
sssi evolve "$@" --json
