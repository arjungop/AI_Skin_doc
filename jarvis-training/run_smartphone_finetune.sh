#!/usr/bin/env bash
# ============================================================
# Fine-tune the model on smartphone/clinical photos
# Uses PAD-UFES-20 data already on disk at ../padufes/
#
# Run on Jarvis Labs (GPU): ~25-35 min
# Run on Mac (MPS):         ~3-4 hours
#
# Usage:
#   bash run_smartphone_finetune.sh          # 15 epochs (default)
#   EPOCHS=20 bash run_smartphone_finetune.sh # custom epochs
# ============================================================
set -e
cd "$(dirname "$0")"

PADUFES_SRC="../padufes"
FINETUNE_DATA="finetune_data/pad_ufes_20"
CHECKPOINT="../backend/ml/weights/best_model.pth"
OUTPUT="../backend/ml/weights/best_model.pth"   # overwrites in place (backup first)

# ── Backup current model ────────────────────────────────────
if [ -f "$OUTPUT" ]; then
  BACKUP="${OUTPUT%.pth}_dermoscopy_backup.pth"
  if [ ! -f "$BACKUP" ]; then
    echo "Backing up current model to $BACKUP"
    cp "$OUTPUT" "$BACKUP"
  else
    echo "Backup already exists: $BACKUP"
  fi
fi

# ── Link padufes data where the script expects it ──────────
mkdir -p finetune_data
if [ ! -d "$FINETUNE_DATA" ]; then
  echo "Linking PAD-UFES-20 data: $PADUFES_SRC -> $FINETUNE_DATA"
  ln -s "$(cd "$PADUFES_SRC" && pwd)" "$FINETUNE_DATA"
else
  echo "PAD-UFES-20 already linked at $FINETUNE_DATA"
fi

# ── Activate venv if present ─────────────────────────────────
if [ -f "../.venv/bin/activate" ]; then
  source ../.venv/bin/activate
fi

# ── Run fine-tuning ──────────────────────────────────────────
echo ""
echo "Starting smartphone domain fine-tuning..."
echo "Checkpoint in : $CHECKPOINT"
echo "Output        : $OUTPUT"
echo ""

EPOCHS="${EPOCHS:-15}"

# Auto-resume if a previous run was interrupted
RESUME_FLAG=""
if [ -f "$(dirname "$OUTPUT")/resume.pth" ] || [ -f "checkpoints/resume.pth" ]; then
  echo "Found resume.pth — resuming interrupted run (use: rm checkpoints/resume.pth to start fresh)"
  RESUME_FLAG="--resume"
fi

python finetune_smartphone.py \
  --checkpoint "$CHECKPOINT" \
  --data-dir "finetune_data" \
  --output-dir "$(dirname "$OUTPUT")" \
  --output-name "best_model.pth" \
  --epochs "$EPOCHS" \
  --batch-size 64 \
  $RESUME_FLAG

echo ""
echo "Done. Fine-tuned model saved to $OUTPUT"
echo "The backend will reload it automatically (uvicorn --reload)."
