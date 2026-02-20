#!/usr/bin/env bash
# Start the OpenAI-compatible API server.
# Usage: serve.sh [--port PORT] [--contribute] [--gpu-memory MEM] [--accelerator TYPE]
set -euo pipefail
exec sssi serve "$@"
