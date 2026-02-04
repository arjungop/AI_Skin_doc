#!/bin/bash
# SLURM job submission for skin disease classification
# Usage: bash scripts/submit_slurm.sh [model] [batch_size] [epochs] [gpu_type]

BACKBONE="${1:-swin_b}"
BATCH_SIZE="${2:-64}"
EPOCHS="${3:-100}"
GPU_TYPE="${4:-auto}"
WALLTIME="48:00:00"

JOB_NAME="skindoc_${BACKBONE}_e${EPOCHS}"
SLURM_SCRIPT="/tmp/skindoc_job_$$.sh"

cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=${WALLTIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Optional: Request specific GPU type if specified
EOF

if [[ "$GPU_TYPE" == "a100" ]]; then
    echo "#SBATCH --constraint=a100" >> "$SLURM_SCRIPT"
elif [[ "$GPU_TYPE" == "a6000" ]]; then
    echo "#SBATCH --constraint=a6000" >> "$SLURM_SCRIPT"
fi

cat >> "$SLURM_SCRIPT" << 'EOF'

# Print job info
echo "SLURM job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo

# Load modules if needed (uncomment and adjust for your cluster)
# module load cuda/12.1
# module load cudnn/8.9
# module load python/3.11

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skindoc

# Verify environment
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo

# Create logs directory if it doesn't exist
mkdir -p logs

# Run validation before training
echo "Running pre-training validation..."
python scripts/validate_setup.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Validation failed! Exiting."
    exit 1
fi
echo

# Run GPU-optimized training
echo "Starting training..."
bash scripts/run_gpu_optimized.sh $BACKBONE $BATCH_SIZE $EPOCHS

# Print completion info
echo
echo "Job completed"
echo "Finished: $(date)"
echo "Total runtime: $SECONDS seconds"
EOF

# Replace placeholders
sed -i.bak "s/\$BACKBONE/$BACKBONE/g" "$SLURM_SCRIPT"
sed -i.bak "s/\$BATCH_SIZE/$BATCH_SIZE/g" "$SLURM_SCRIPT"
sed -i.bak "s/\$EPOCHS/$EPOCHS/g" "$SLURM_SCRIPT"
rm -f "${SLURM_SCRIPT}.bak"

# Display job configuration
echo "Submitting SLURM job..."
echo "Job name: $JOB_NAME"
echo "Model: $BACKBONE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Walltime: 48 hours"
echo "GPU type: $GPU_TYPE"
echo "Memory: 64GB, CPUs: 8"
echo

# Create logs directory
mkdir -p logs

# Submit job
echo "Submitting job to SLURM..."
sbatch "$SLURM_SCRIPT"

if [ $? -eq 0 ]; then
    echo
    echo "[SUCCESS] Job submitted successfully!"
    echo
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f logs/slurm_<job_id>.out"
    echo
    echo "Cancel with:"
    echo "  scancel <job_id>"
else
    echo
    echo "[ERROR] Job submission failed!"
    exit 1
fi

# Keep the SLURM script for reference
SAVED_SCRIPT="logs/slurm_script_${JOB_NAME}_$(date +%Y%m%d_%H%M%S).sh"
cp "$SLURM_SCRIPT" "$SAVED_SCRIPT"
echo "SLURM script saved to: $SAVED_SCRIPT"

# Cleanup
rm -f "$SLURM_SCRIPT"
