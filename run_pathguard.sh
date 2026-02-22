#!/usr/bin/env bash
set -euo pipefail

# PathGuard HUD Only — Spatial Safety HUD with Corridor + GroundedDINO + Depth
# Does NOT require Cactus engine / Apple Silicon

source .venv/bin/activate 2>/dev/null || true

export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/cactus/python/src"

streamlit run "pages/1_🛡️_PathGuard_HUD.py"
