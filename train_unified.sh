#!/bin/bash
#SBATCH --job-name=skindoc-unified
#SBATCH --output=logs/unified_%j.log
#SBATCH --error=logs/unified_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=workq
#SBATCH --requeue

set -e
source /dist_home/suryansh/miniforge3/bin/activate skindoc
cd $SLURM_SUBMIT_DIR
mkdir -p logs checkpoints

nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

if [[ "$GPU_NAME" == *"A100"* ]]; then
    # A100: Larger batch, larger workers, bf16 enabled
    BS=64
    MODEL="convnext_large"
    WORKERS=24
    PRECISION="bf16"
else
    # A6000 or others: Smaller batch to be safe, fp16
    BS=32
    MODEL="convnext_base"
    WORKERS=16
    PRECISION="fp16"
fi

echo "=================================================="
echo "ðŸ”¥ UNIFIED TRAINING LAUNCHER"
echo "=================================================="
echo "GPU: $GPU_NAME"
echo "Model: $MODEL"
echo "Batch Size: $BS"
echo "Precision: $PRECISION"
echo "Workers: $WORKERS"
echo "=================================================="

export OMP_NUM_THREADS=$WORKERS
export MKL_NUM_THREADS=$WORKERS

# Launch python script with EXPLICIT arguments
python scripts/train_unified.py \
    --data_dir data/unified_train \
    --val_dir data/unified_val \
    --checkpoint_dir checkpoints \
    --backbone $MODEL \
    --batch_size $BS \
    --image_size 384 \
    --epochs 60 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --patience 15 \
    --grad_accum 2 \
    --num_workers $WORKERS \
    --weighted_sampling \
    --mixed_precision $PRECISION

echo "Done: $(date)"
