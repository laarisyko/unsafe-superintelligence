#!/usr/bin/env bash
# Start the USSI P2P node.
set -euo pipefail

ussi node start --openclaw "$@" --json
