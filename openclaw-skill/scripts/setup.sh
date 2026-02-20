#!/usr/bin/env bash
# Setup script: install SSSI SDK and start a P2P node.
set -euo pipefail

echo "[sssi] Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
    echo "[sssi] ERROR: python3 not found. Please install Python 3.10+." >&2
    exit 1
fi

if ! command -v pip &>/dev/null && ! command -v pip3 &>/dev/null; then
    echo "[sssi] ERROR: pip not found." >&2
    exit 1
fi

PIP_CMD="pip"
command -v pip3 &>/dev/null && PIP_CMD="pip3"

echo "[sssi] Installing supersafesuperintelligence..."
$PIP_CMD install --quiet supersafesuperintelligence

echo "[sssi] Verifying installation..."
sssi --help >/dev/null 2>&1 || { echo "[sssi] ERROR: sssi CLI not found after install." >&2; exit 1; }

echo "[sssi] Detecting compute resources..."
sssi detect --json

if command -v docker &>/dev/null; then
    echo "[sssi] Docker available. You can start a node with: sssi node start"
else
    echo "[sssi] WARNING: Docker not found. Install Docker to run a P2P node, or build the Rust binary from source."
fi

echo "[sssi] Setup complete."
