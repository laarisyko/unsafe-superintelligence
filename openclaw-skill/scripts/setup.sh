#!/usr/bin/env bash
# Setup script: install USSI SDK and start a P2P node.
set -euo pipefail

echo "[ussi] Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
    echo "[ussi] ERROR: python3 not found. Please install Python 3.10+." >&2
    exit 1
fi

if ! command -v pip &>/dev/null && ! command -v pip3 &>/dev/null; then
    echo "[ussi] ERROR: pip not found." >&2
    exit 1
fi

PIP_CMD="pip"
command -v pip3 &>/dev/null && PIP_CMD="pip3"

echo "[ussi] Installing unsafesuperintelligence..."
$PIP_CMD install --quiet unsafesuperintelligence

echo "[ussi] Verifying installation..."
ussi --help >/dev/null 2>&1 || { echo "[ussi] ERROR: ussi CLI not found after install." >&2; exit 1; }

echo "[ussi] Detecting compute resources..."
ussi detect --json

if command -v docker &>/dev/null; then
    echo "[ussi] Docker available. You can start a node with: ussi node start"
else
    echo "[ussi] WARNING: Docker not found. Install Docker to run a P2P node, or build the Rust binary from source."
fi

echo "[ussi] Setup complete."
