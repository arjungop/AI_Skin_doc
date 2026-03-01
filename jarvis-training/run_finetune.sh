#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Smartphone Fine-tuning Runner
# Run this on Jarvis Labs (RTX 6000 Ada) — estimated time: ~35 minutes
#
# FIRST TIME SETUP — Kaggle credentials (one-time):
#   1. Go to https://www.kaggle.com/settings → API → "Create New Token"
#      → downloads kaggle.json
#   2. Run on this machine:
#        mkdir -p ~/.kaggle
#        mv /path/to/kaggle.json ~/.kaggle/kaggle.json
#        chmod 600 ~/.kaggle/kaggle.json
#   Then run this script — PAD-UFES-20 will download automatically (~1GB).
# ─────────────────────────────────────────────────────────────────────────────
set -e

cd "$(dirname "$0")"

echo "======================================="
echo "  Skin-Doc Smartphone Fine-tuning"
echo "======================================="
echo ""

# Install dependencies (includes kaggle CLI)
pip install -q -r requirements_finetune.txt

# Check kaggle credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo ""
  echo "ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json"
  echo ""
  echo "  1. Go to https://www.kaggle.com/settings"
  echo "  2. Click API -> 'Create New Token' -> downloads kaggle.json"
  echo "  3. Run:"
  echo "       mkdir -p ~/.kaggle"
  echo "       mv /path/to/kaggle.json ~/.kaggle/kaggle.json"
  echo "       chmod 600 ~/.kaggle/kaggle.json"
  echo "  4. Re-run this script."
  echo ""
  exit 1
fi

# Run fine-tuning
# Checkpoint is auto-downloaded from HuggingFace (arjg/skin-doc-model)
# if not already present in ../backend/ml/weights/best_model.pth
python finetune_smartphone.py \
    --epochs 15 \
    --batch-size 64 \
    --lr 5e-5 \
    --num-workers 8 \
    --unfreeze-stages 2 3

echo ""
echo "======================================="
echo "  Done! Copying to backend weights..."
echo "======================================="
cp checkpoints/finetuned_smartphone.pth ../backend/ml/weights/best_model.pth
echo "  Replaced ../backend/ml/weights/best_model.pth"
echo ""
echo "  Test it:"
echo "    python run_test.py <your_image.png>"
