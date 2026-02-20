#!/usr/bin/env bash
# Start the USSI P2P node.
set -euo pipefail

BOOTSTRAP="${1:-}"
EXTRA_ARGS=""
if [ -n "$BOOTSTRAP" ]; then
    EXTRA_ARGS="--bootstrap $BOOTSTRAP"
fi

ussi node start $EXTRA_ARGS --json
