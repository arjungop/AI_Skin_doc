#!/bin/bash
#SBATCH --job-name=cvnxt_l_top20           # Job name
#SBATCH --output=logs/train_%j.log         # Standard output log (%j = job ID)
#SBATCH --error=logs/train_%j.err          # Standard error log
#SBATCH --partition=workq                  # GPU partition
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=16                 # CPU cores per task
#SBATCH --gres=gpu:rtx6000:1               # Request 1 RTX 6000 Ada GPU
#SBATCH --mem=96G                          # Increased memory for Large model
#SBATCH --time=72:00:00                    # Time limit (72 hours for safety)
#SBATCH --mail-type=END,FAIL               # Email on completion/failure only
#SBATCH --mail-user=suryansh@example.com   # Your email (change this!)

# ==================== Job Information ====================
echo "================================================================================"
echo "🚀 ConvNeXt Training Job Started"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "================================================================================"
echo ""

# ==================== Environment Setup ====================
# Change to working directory
cd /dist_home/suryansh/arjungop/AI_Skin_doc || exit 1

# Create logs directory
mkdir -p logs

# Load modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate conda environment
echo "Activating conda environment..."
eval "$(/dist_home/suryansh/miniforge3/bin/conda shell.bash hook)"
conda activate skindoc
echo "✅ Conda environment 'skindoc' activated"

# ==================== GPU Information ====================
echo "GPU Information:"
nvidia-smi
echo ""
echo "CUDA Version:"
nvcc --version || echo "nvcc not found"
echo ""

# ==================== Python Environment ====================
echo "Python Environment:"
which python3
python3 --version
echo ""

echo "PyTorch Version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# ==================== Install/Update Dependencies ====================
echo "Checking dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements_training.txt
echo "✅ Dependencies ready"
echo ""

# ==================== Training Configuration ====================
MODEL_NAME="convnext_large"
BATCH_SIZE=48
EPOCHS=50
LR=3e-3
IMG_SIZE=384
DATA_DIR="/dist_home/suryansh/arjungop/AI_Skin_doc/data"
SAVE_DIR="checkpoints/convnext_large_top20"

echo "Training Configuration:"
echo "  Model: $MODEL_NAME (ConvNeXt Large for maximum accuracy)"
echo "  Dataset: Top 20 classes from unified dataset"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Image Size: $IMG_SIZE"
echo "  Data Directory: $DATA_DIR"
echo "  Save Directory: $SAVE_DIR"
echo ""

# ==================== Start Training ====================
echo "================================================================================"
echo "🔥 Starting Training..."
echo "================================================================================"
echo ""

# Run training script
python3 train_convnext_production.py \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --img_size $IMG_SIZE \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --train_dir top20_train \
    --val_dir top20_val \
    --num_workers 16 \
    --warmup_epochs 5 \
    --seed 42 \
    --mixed_precision True

# Capture exit code
TRAIN_EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "End Time: $(date)"
echo "================================================================================"
echo ""

# ==================== Post-Training Actions ====================
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "📊 Generating training summary..."
    
    # Copy best model to a timestamped backup
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    if [ -f "$SAVE_DIR/best_model.pth" ]; then
        cp "$SAVE_DIR/best_model.pth" "$SAVE_DIR/best_model_$TIMESTAMP.pth"
        echo "✅ Best model backed up to: $SAVE_DIR/best_model_$TIMESTAMP.pth"
    fi
    
    # Show final metrics
    if [ -f "$SAVE_DIR/history.json" ]; then
        echo ""
        echo "Final Training Metrics:"
        python3 -c "
import json
with open('$SAVE_DIR/history.json', 'r') as f:
    h = json.load(f)
    print(f\"  Final Train Acc: {h['train_acc'][-1]:.2f}%\")
    print(f\"  Final Val Acc: {h['val_acc'][-1]:.2f}%\")
    print(f\"  Best Val Acc: {max(h['val_acc']):.2f}%\")
"
    fi
    
    echo ""
    echo "📝 To view TensorBoard logs, run:"
    echo "   tensorboard --logdir=runs/convnext_base_production"
    echo ""
fi

# ==================== Cleanup (Optional) ====================
# Clean up old checkpoints if needed
# find $SAVE_DIR -name "checkpoint_epoch_*.pth" -mtime +7 -delete

echo "Job finished at: $(date)"

exit $TRAIN_EXIT_CODE
