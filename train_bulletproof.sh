#!/bin/bash
#SBATCH --job-name=skindoc-train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
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
    BS=48; MODEL="convnext_large"; WORKERS=16
else
    BS=32; MODEL="convnext_base"; WORKERS=12
fi

echo "GPU: $GPU_NAME | Model: $MODEL | Batch: $BS"

export OMP_NUM_THREADS=$WORKERS
export MKL_NUM_THREADS=$WORKERS

python scripts/train_bulletproof.py \
    --data_dir data/unified_train \
    --val_dir data/unified_val \
    --checkpoint_dir checkpoints \
    --variant $MODEL \
    --batch_size $BS \
    --image_size 384 \
    --epochs 50 \
    --lr 1e-4 \
    --patience 10 \
    --dropout 0.5 \
    --grad_accum 2 \
    --num_workers $WORKERS \
    --weighted_sampling

echo "Done: $(date)"
