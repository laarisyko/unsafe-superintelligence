#!/usr/bin/env bash
# Join the USSI network and advertise compute capacity.
# Usage: join.sh [--gpu-memory 8GB] [--accelerator cuda]
set -euo pipefail
ussi join --openclaw "$@" --json
