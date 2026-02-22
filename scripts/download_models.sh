#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Download & Convert Model Weights for the Cactus On-Device Video Narrator
# ============================================================================
#
# This script downloads the raw PyTorch weights from HuggingFace and converts
# them locally to INT8 quantized format for the Apple Neural Engine.
#
# IMPORTANT: Do NOT use `cactus download` from Homebrew — the pre-compiled
# zip files on HuggingFace are missing the `projector_layer_norm` tensor,
# which causes a silent crash during model initialization. This script
# forces a local reconversion from the raw float32 checkpoints.
#
# Usage:
#   ./scripts/download_models.sh [MODEL_ID]
#
# Examples:
#   ./scripts/download_models.sh                              # defaults to LFM2.5-VL-1.6B
#   ./scripts/download_models.sh LiquidAI/LFM2.5-VL-1.6B     # explicit
# ============================================================================

MODEL_VL="${1:-LiquidAI/LFM2.5-VL-1.6B}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================="
echo "  Cactus Model Weight Downloader"
echo "  Model: $MODEL_VL"
echo "  Precision: INT8 (local reconversion)"
echo "============================================="
echo ""

# Activate venv
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "ERROR: Virtual environment not found at $REPO_ROOT/.venv"
    echo "Run setup.sh first, or create it manually: python3 -m venv .venv"
    exit 1
fi

# Set PYTHONPATH so the cactus CLI can find its own modules
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/cactus/python/src:$REPO_ROOT/cactus/python"

# Run the conversion using the Python module directly (not the broken pip wrapper)
cd "$REPO_ROOT/cactus/python"
python -m src.cli download "$MODEL_VL" --precision INT8 --reconvert

echo ""
echo "Weights saved to: $REPO_ROOT/cactus/weights/"

# Copy weights to the root weights/ directory so app.py can find them
mkdir -p "$REPO_ROOT/weights"

# Find the model directory name (lowercase, hyphens replaced)
MODEL_DIR_NAME=$(echo "$MODEL_VL" | tr '/' '-' | tr '[:upper:]' '[:lower:]' | sed 's/^[^-]*-//')
SRC_DIR="$REPO_ROOT/cactus/weights/$MODEL_DIR_NAME"
DST_DIR="$REPO_ROOT/weights/$MODEL_DIR_NAME"

if [ -d "$SRC_DIR" ]; then
    echo "Symlinking $SRC_DIR -> $DST_DIR"
    ln -sfn "$SRC_DIR" "$DST_DIR"
    echo "Done! Weights are accessible at: weights/$MODEL_DIR_NAME"
else
    echo "WARNING: Expected weight directory not found at $SRC_DIR"
    echo "You may need to manually copy or symlink the weights."
    ls "$REPO_ROOT/cactus/weights/"
fi
