#!/usr/bin/env bash
set -euo pipefail

# Cactus Narrator Only — On-Device VLM Video Narrator
# Requires: macOS Apple Silicon + compiled Cactus engine (libcactus.dylib)

source .venv/bin/activate 2>/dev/null || true

# Inject the Cactus Python bindings AND the narrator package into PYTHONPATH
# - cactus/python/src: contains the cactus FFI module (cactus_init, cactus_complete, etc.)
# - . (root): contains narrator/ package with cactus_vl.py
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/cactus/python/src:$(pwd)"

streamlit run "pages/2_🌵_Cactus_Narrator.py"
