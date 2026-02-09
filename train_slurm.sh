#!/bin/bash
#SBATCH --job-name=skin-convnext
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu           # <-- UPDATE: Your GPU partition name
# #SBATCH --constraint=a100       # Uncomment to request A100 specifically

# ============================================================================
# SLURM Job Script for ConvNeXt Training
# ============================================================================

echo "=============================================="
echo "🚀 SLURM Job Started"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

# Activate environment
source /dist_home/suryansh/miniforge3/bin/activate skindoc

# Navigate to project
cd $SLURM_SUBMIT_DIR

# Create logs directory
mkdir -p logs checkpoints

# Show GPU info
nvidia-smi

echo ""
echo "=============================================="
echo "Starting Training..."
echo "=============================================="

# Detect GPU and set optimal batch size
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "GPU: $GPU_NAME ($GPU_MEM MB)"

# Auto-configure based on GPU
if [[ "$GPU_NAME" == *"A100"* ]]; then
    BATCH_SIZE=48
    VARIANT="convnext_large"
    echo "🔥 A100 detected - Using large model with batch_size=$BATCH_SIZE"
elif [[ "$GPU_NAME" == *"6000"* ]] || [[ "$GPU_NAME" == *"A6000"* ]]; then
    BATCH_SIZE=32
    VARIANT="convnext_base"
    echo "⚡ RTX 6000 detected - Using base model with batch_size=$BATCH_SIZE"
else
    BATCH_SIZE=24
    VARIANT="convnext_base"
    echo "🖥️ Generic GPU - Using conservative settings"
fi

# Run training
python scripts/train_convnext.py \
    --data_dir data/unified_train \
    --val_dir data/unified_val \
    --checkpoint_dir checkpoints \
    --variant $VARIANT \
    --batch_size $BATCH_SIZE \
    --image_size 384 \
    --epochs 50 \
    --lr 1e-4 \
    --patience 10 \
    --dropout 0.5 \
    --grad_accum 2 \
    --num_workers 8 \
    --weighted_sampling

echo ""
echo "=============================================="
echo "✅ Job Complete: $(date)"
echo "=============================================="
