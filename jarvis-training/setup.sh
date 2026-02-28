#!/usr/bin/env bash
# ============================================================================
# ONE-CLICK SETUP — Jarvis Labs A6000
# Downloads datasets, prepares data, and validates the environment.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh                    # Full setup (all datasets)
#   ./setup.sh --skip-massive     # Skip 50 GB Massive dataset
# ============================================================================

set -euo pipefail

echo ""
echo "================================================================="
echo "  SKIN DISEASE TRAINING — JARVIS LABS SETUP"
echo "================================================================="
echo ""

# ── 1. Check GPU ──
echo "[1/5] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  WARNING: nvidia-smi not found. Ensure you selected a GPU instance."
fi
echo ""

# ── 2. Check Kaggle API ──
echo "[2/5] Checking Kaggle API..."
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo ""
    echo "  ERROR: Kaggle API not configured!"
    echo ""
    echo "  Steps to fix:"
    echo "    1. Go to https://www.kaggle.com/settings"
    echo "    2. Click 'Create New Token' under API"
    echo "    3. Upload the downloaded kaggle.json to this instance"
    echo "    4. Run:"
    echo "       mkdir -p ~/.kaggle"
    echo "       mv kaggle.json ~/.kaggle/"
    echo "       chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "  Then re-run: ./setup.sh"
    exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"
echo "  Kaggle API: OK"
echo ""

# ── 3. Install dependencies ──
echo "[3/5] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  Dependencies: OK"
echo ""

# ── 4. Download datasets ──
echo "[4/5] Downloading datasets..."
EXTRA_ARGS=""
if [[ "${1:-}" == "--skip-massive" ]]; then
    EXTRA_ARGS="--skip-massive"
    echo "  (Skipping Massive Balanced dataset as requested)"
fi
python download_data.py $EXTRA_ARGS
echo ""

# ── 5. Prepare unified dataset ──
echo "[5/5] Preparing unified top-20 dataset..."
python prepare_data.py --top-n 20
echo ""

# ── Done ──
echo "================================================================="
echo "  SETUP COMPLETE!"
echo "================================================================="
echo ""
echo "  To start training:"
echo "    python train.py"
echo ""
echo "  To resume interrupted training:"
echo "    python train.py --resume"
echo ""
echo "  To evaluate after training:"
echo "    python evaluate.py"
echo "    python evaluate.py --tta    (with test-time augmentation)"
echo ""
echo "  Monitor with TensorBoard:"
echo "    tensorboard --logdir=runs --port=6006"
echo ""
echo "================================================================="
