#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Smartphone Fine-tuning Runner
# Run this on Jarvis Labs (RTX 6000 Ada) — estimated time: ~35 minutes
#
# BEFORE running, place PAD-UFES-20 dataset manually:
#   1. Download from: https://www.kaggle.com/datasets/andrewmvd/pad-ufes-20
#   2. Unzip into:    jarvis-training/finetune_data/pad_ufes_20/
#      So you have:   finetune_data/pad_ufes_20/metadata.csv
#                     finetune_data/pad_ufes_20/*.png
# ─────────────────────────────────────────────────────────────────────────────
set -e

cd "$(dirname "$0")"

echo "======================================="
echo "  Skin-Doc Smartphone Fine-tuning"
echo "======================================="
echo ""
echo "Checking for PAD-UFES-20 dataset..."
if [ ! -f "finetune_data/pad_ufes_20/metadata.csv" ]; then
  echo ""
  echo "ERROR: PAD-UFES-20 not found!"
  echo "  Download from: https://www.kaggle.com/datasets/andrewmvd/pad-ufes-20"
  echo "  Unzip into: $(pwd)/finetune_data/pad_ufes_20/"
  echo "  Required:   finetune_data/pad_ufes_20/metadata.csv"
  echo ""
  exit 1
fi
echo "  Found PAD-UFES-20 dataset."
echo ""

# Install dependencies
pip install -q -r requirements_finetune.txt

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
