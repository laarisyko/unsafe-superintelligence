#!/usr/bin/env bash
# Join a decentralized training round.
# Usage: train.sh --model llama-7b --rounds 5
set -euo pipefail
ussi train "$@" --json
