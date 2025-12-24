#!/usr/bin/env bash
set -euo pipefail

# Train + evaluate the skin model and set env vars.
# Usage:
#   scripts/train_and_eval.sh \
#     --data dataset/melanoma_cancer_dataset \
#     --backbone resnet50 \
#     --epochs 12 \
#     --batch 32 \
#     --weights backend/ml/weights/skin_resnet50.pth \
#     --threshold 0.45

DATA_DIR="data/lesions_binary"
BACKBONE="resnet18"
EPOCHS=12
BATCH=32
WEIGHTS="backend/ml/weights/skin_resnet18.pth"
THRESHOLD="0.45"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_DIR="$2"; shift 2;;
    --backbone) BACKBONE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --weights) WEIGHTS="$2"; shift 2;;
    --threshold) THRESHOLD="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "[train_and_eval] Data: $DATA_DIR | Backbone: $BACKBONE | Epochs: $EPOCHS | Batch: $BATCH"
mkdir -p "$(dirname "$WEIGHTS")"

# Train
python3 scripts/train_skin.py \
  --data-dir "$DATA_DIR" \
  --split-subdirs \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --output "$WEIGHTS"

# Evaluate (prefer test split if present)
TEST_DIR="$DATA_DIR/test"
if [[ ! -d "$TEST_DIR" ]]; then
  echo "[train_and_eval] WARNING: $TEST_DIR not found; evaluating on full data at $DATA_DIR"
  TEST_DIR="$DATA_DIR"
fi
OUT_DIR="reports/$(basename "$WEIGHTS" .pth)"
python3 scripts/eval_skin.py \
  --weights "$WEIGHTS" \
  --data-dir "$TEST_DIR" \
  --backbone "$BACKBONE" \
  --out "$OUT_DIR"

# Write .env entries (idempotent-ish: append lines; user can dedupe later)
{
  echo "LESION_MODEL_WEIGHTS=$WEIGHTS"
  echo "LESION_FORCE_THRESHOLD=1"
  echo "LESION_MALIGNANT_THRESHOLD=$THRESHOLD"
} >> .env

echo "[train_and_eval] Done. Weights: $WEIGHTS"
echo "[train_and_eval] Metrics: $OUT_DIR/metrics.txt"
echo "[train_and_eval] Confusion matrix: $OUT_DIR/confusion_matrix.png"
echo "[train_and_eval] .env updated with model settings. Restart backend to apply."
