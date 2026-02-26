#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Cactus On-Device Video Narrator — Full Environment Bootstrap
# ============================================================================
#
# This script sets up EVERYTHING from scratch:
#   1. Creates a Python virtual environment
#   2. Installs all pip dependencies
#   3. Compiles the Cactus C++ engine (libcactus.dylib) for Apple Neural Engine
#   4. Downloads and converts the LiquidAI VLM weights to INT8
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# After setup completes, launch the app with:
#   ./run_combined.sh
# ============================================================================

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "============================================="
echo "  🌵 Cactus On-Device Video Narrator Setup"
echo "============================================="
echo ""

# ------------------------------------------------------------------
# Step 1: Verify System Prerequisites
# ------------------------------------------------------------------
echo "📋 Step 1/4: Checking system prerequisites..."

MISSING=()

if ! command -v python3 &> /dev/null; then
    MISSING+=("python3")
fi

if ! command -v cmake &> /dev/null; then
    MISSING+=("cmake")
fi

if ! command -v make &> /dev/null; then
    MISSING+=("make")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "❌ ERROR: Missing required tools: ${MISSING[*]}"
    echo ""
    echo "Install them with:"
    echo "  xcode-select --install"
    echo "  brew install cmake python3"
    echo ""
    exit 1
fi

echo "  ✅ python3, cmake, make — all found."
echo ""

# ------------------------------------------------------------------
# Step 2: Create Virtual Environment & Install Dependencies
# ------------------------------------------------------------------
echo "📦 Step 2/4: Setting up Python virtual environment..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created .venv/"
else
    echo "  .venv/ already exists, reusing."
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements_pathguard.txt -q
pip install -r requirements_narrator.txt -q

echo "  ✅ All Python dependencies installed."
echo ""

# ------------------------------------------------------------------
# Step 3: Compile the Cactus C++ Engine
# ------------------------------------------------------------------
echo "🔨 Step 3/4: Compiling Cactus C++ engine for Apple Neural Engine..."

if [ -f "cactus/cactus/build/libcactus.dylib" ]; then
    echo "  libcactus.dylib already exists. Skipping build."
    echo "  (Delete cactus/cactus/build/ and re-run to force rebuild)"
else
    cd cactus/cactus
    chmod +x build.sh
    ./build.sh
    cd "$REPO_ROOT"
fi

echo "  ✅ libcactus.dylib compiled successfully."
echo ""

# ------------------------------------------------------------------
# Step 4: Download & Convert Model Weights
# ------------------------------------------------------------------
echo "🧠 Step 4/4: Downloading and quantizing LiquidAI/LFM2.5-VL-1.6B to INT8..."

WEIGHT_DIR="weights/lfm2.5-vl-1.6b"

if [ -d "$WEIGHT_DIR" ] && [ "$(ls -A "$WEIGHT_DIR" 2>/dev/null)" ]; then
    echo "  Weight directory $WEIGHT_DIR already populated. Skipping download."
    echo "  (Delete $WEIGHT_DIR and re-run to force re-download)"
else
    chmod +x scripts/download_models.sh
    ./scripts/download_models.sh LiquidAI/LFM2.5-VL-1.6B
fi

echo "  ✅ Model weights ready."
echo ""

# ------------------------------------------------------------------
# Step 5: Create .env if missing
# ------------------------------------------------------------------
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "⚠️  Created .env from .env.example."
        echo "   Please edit .env and add your GEMINI_API_KEY."
    else
        echo 'GEMINI_API_KEY="your_vertex_ai_key_here"' > .env
        echo "⚠️  Created .env with placeholder key."
        echo "   Please edit .env and add your GEMINI_API_KEY."
    fi
else
    echo "✅ .env already exists."
fi

echo ""
echo "============================================="
echo "  🎉 Setup Complete!"
echo "============================================="
echo ""
echo "  Launch the application with:"
echo "    ./run_combined.sh"
echo ""
echo "  The app will open at: http://localhost:8501"
echo "============================================="
