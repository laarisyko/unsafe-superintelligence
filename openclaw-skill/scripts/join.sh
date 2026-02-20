#!/usr/bin/env bash
# Join the SSSI network and advertise compute capacity.
# Usage: join.sh [--gpu-memory 8GB] [--accelerator cuda]
set -euo pipefail
sssi join "$@" --json
