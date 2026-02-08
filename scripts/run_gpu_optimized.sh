#!/bin/bash
# GPU-optimized training runner
# Auto-detects A100 or A6000 and optimizes settings

set -e

# Get GPU info
get_gpu_info() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "unknown"
        return
    fi
    
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "$gpu_name"
}

GPU_NAME=$(get_gpu_info)

echo "GPU-optimized training"
echo "Detected GPU: $GPU_NAME"

# Optimize settings based on GPU
BACKBONE="${1:-convnext_large}"
EPOCHS="${3:-50}"

# Detect GPU and optimize settings
if [[ "$GPU_NAME" == *"A100"* ]]; then
    echo "✅ NVIDIA A100 detected - using optimal settings"
    BATCH_SIZE="${2:-32}"
    GRAD_ACCUM=2
    PRECISION="bfloat16"
    echo "Backbone: $BACKBONE"
    echo "Batch size: $BATCH_SIZE"
    echo "Grad Accum: $GRAD_ACCUM"
    echo "Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
    
elif [[ "$GPU_NAME" == *"RTX 6000 Ada"* ]] || [[ "$GPU_NAME" == *"6000 Ada"* ]]; then
    echo "✅ NVIDIA RTX 6000 Ada detected - using optimal settings"
    BATCH_SIZE="${2:-32}"
    GRAD_ACCUM=2
    PRECISION="float16"
    
elif [[ "$GPU_NAME" == *"A6000"* ]]; then
    echo "✅ NVIDIA A6000 detected - using optimal settings"
    BATCH_SIZE="${2:-24}"
    GRAD_ACCUM=3
    PRECISION="float16"
else
    BATCH_SIZE="${2:-16}"
    GRAD_ACCUM=4
    PRECISION="float16"
fi

echo
echo "Configuration: $BACKBONE, batch=$BATCH_SIZE, epochs=$EPOCHS, grad_accum=$GRAD_ACCUM"
echo

# Run the bulletproof trainer
python scripts/train_bulletproof.py \
    --backbone "$BACKBONE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --grad_accum "$GRAD_ACCUM" \
    --checkpoint_dir "checkpoints/$BACKBONE" \
    --use_weighted_sampling

echo "----------------------------------------------------------------"
echo "✅ Training session completed"
echo "----------------------------------------------------------------"
