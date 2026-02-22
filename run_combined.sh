#!/usr/bin/env bash
set -euo pipefail

# PathGuard Combined App — Launches Both HUD + Narrator as Multipage Streamlit App
# Requires: Python venv with dependencies for whichever system you want to use

source .venv/bin/activate 2>/dev/null || true

# Add all required paths:
# - . (root)          : pathguard/, narrator/, integration/ packages
# - cactus/python/src : Cactus C++ FFI bindings (cactus_init, cactus_complete, etc.)
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/cactus/python/src"

streamlit run Home.py
