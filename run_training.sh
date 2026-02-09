#!/bin/bash
# ============================================================================
# ConvNeXt Training Script for Skin Disease Classification
# Optimized for RTX 6000 Ada (48GB) / A100 (40GB)
# ============================================================================

# Activate conda environment
source /dist_home/suryansh/miniforge3/bin/activate skindoc

# Navigate to project directory (adjust if different)
cd /home/suryansh/Skin-Doc  # <-- UPDATE THIS PATH

echo "=============================================="
echo "🚀 ConvNeXt Skin Disease Training"
echo "=============================================="
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Choose your variant based on GPU memory:
#   - convnext_tiny:  ~8GB  (fast, lower accuracy)
#   - convnext_small: ~12GB (balanced)
#   - convnext_base:  ~20GB (recommended for 48GB GPU)
#   - convnext_large: ~35GB (highest accuracy, needs 40GB+)

VARIANT="convnext_base"   # Recommended for RTX 6000 Ada
BATCH_SIZE=32             # Can increase to 48-64 for RTX 6000 Ada
IMAGE_SIZE=384            # Standard medical imaging size
EPOCHS=50
LR=1e-4
PATIENCE=10

# ============================================================================
# RUN TRAINING
# ============================================================================
python scripts/train_convnext.py \
    --data_dir data/unified_train \
    --val_dir data/unified_val \
    --checkpoint_dir checkpoints \
    --variant $VARIANT \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --patience $PATIENCE \
    --dropout 0.5 \
    --grad_accum 2 \
    --num_workers 8 \
    --weighted_sampling

echo ""
echo "=============================================="
echo "✅ Training Complete!"
echo "End time: $(date)"
echo "=============================================="
