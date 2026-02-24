#!/usr/bin/env bash
# One-command bootstrap for OpenClaw agents:
# install USSI, discover bootstrap peers, start node, and join as contributor.
set -euo pipefail

bash "$(dirname "$0")/setup.sh"

echo "[ussi] Running OpenClaw bootstrap..."
ussi openclaw bootstrap "$@" --json

echo "[ussi] Verifying status/quota..."
ussi status --json
ussi quota --json
