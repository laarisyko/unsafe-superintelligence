#!/usr/bin/env bash
# Vote on an architecture proposal.
# Usage: vote.sh --proposal arch-abc123 --decision approve
set -euo pipefail
ussi vote "$@" --json
