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
BACKBONE="${1:-efficientnet_b4}"
EPOCHS="${3:-50}"

# Detect GPU and optimize settings
if [[ "$GPU_NAME" == *"A100"* ]]; then
    echo "✅ NVIDIA A100 detected - using optimal settings"
    BATCH_SIZE="${2:-64}"
    GRAD_ACCUM=2
    PRECISION="bfloat16"
    echo "Batch size: $BATCH_SIZE"
    echo "Mixed precision: $PRECISION"
    echo "Expected time: ~10 hours (Swin-Base)"
    
elif [[ "$GPU_NAME" == *"RTX 6000 Ada"* ]] || [[ "$GPU_NAME" == *"6000 Ada"* ]]; then
    echo "✅ NVIDIA RTX 6000 Ada detected - using optimal settings"
    BATCH_SIZE="${2:-56}"
    GRAD_ACCUM=2
    PRECISION="float16"
    echo "Batch size: $BATCH_SIZE (49GB VRAM)"
    echo "Mixed precision: $PRECISION"
    echo "Expected time: ~12 hours (Swin-Base)"
    
elif [[ "$GPU_NAME" == *"A6000"* ]]; then
    echo "✅ NVIDIA A6000 detected - using optimal settings"
    BATCH_SIZE="${2:-54}"
    GRAD_ACCUM=2
    PRECISION="float16"
    echo "Batch size: $BATCH_SIZE (48GB VRAM)"
    echo "Mixed precision: $PRECISION"
    echo "Expected time: ~14 hours (Swin-Base)"
    
elif [[ "$GPU_NAME" == *"RTX"* ]] || [[ "$GPU_NAME" == *"GeForce"* ]]; then
    echo "✅ NVIDIA $GPU_NAME detected - using adaptive settings"
    BATCH_SIZE="${2:-32}"
    GRAD_ACCUM=4
    PRECISION="float16"
    echo "Batch size: $BATCH_SIZE (consumer GPU optimized)"
    echo "Mixed precision: $PRECISION"
    echo "Note: Adjust batch size if OOM occurs"
    
elif [[ "$GPU_NAME" == *"Tesla"* ]] || [[ "$GPU_NAME" == *"V100"* ]]; then
    echo "✅ NVIDIA $GPU_NAME detected - using datacenter settings"
    BATCH_SIZE="${2:-48}"
    GRAD_ACCUM=3
    PRECISION="float16"
    echo "Batch size: $BATCH_SIZE"
    echo "Mixed precision: $PRECISION"
    
elif [[ "$GPU_NAME" != "unknown" ]]; then
    echo "✅ GPU detected: $GPU_NAME - using universal settings"
    BATCH_SIZE="${2:-24}"
    GRAD_ACCUM=4
    PRECISION="float16"
    echo "Batch size: $BATCH_SIZE (conservative for compatibility)"
    echo "Mixed precision: $PRECISION"
    echo "Note: Will auto-adjust if needed"
    
else
    echo "⚠️  No GPU detected - using CPU fallback settings"
    BATCH_SIZE="${2:-16}"
    GRAD_ACCUM=8
    PRECISION="float32"
    echo "Batch size: $BATCH_SIZE"
    echo "Warning: Training will be very slow on CPU"
fi

echo
echo "Configuration: $BACKBONE, batch=$BATCH_SIZE, epochs=$EPOCHS, grad_accum=$GRAD_ACCUM"
echo

# Source the main training script
export DETECTED_GPU_TYPE="$GPU_NAME"
export OPTIMIZED_BATCH_SIZE="$BATCH_SIZE"

bash scripts/run_complete_training.sh "$BACKBONE" "$BATCH_SIZE" "$EPOCHS"
