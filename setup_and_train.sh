#!/bin/bash
# Master setup script for ConvNeXt Large with Top 20 classes

set -e  # Exit on any error

echo "================================================================================"
echo "🚀 ConvNeXt Large - Top 20 Classes Setup & Training"
echo "================================================================================"
echo ""

# Change to working directory
cd /dist_home/suryansh/arjungop/AI_Skin_doc || exit 1

echo "📂 Current directory: $(pwd)"
echo ""

# Step 1: Prepare top 20 dataset
echo "================================================================================"
echo "📊 STEP 1: Preparing Top 20 Dataset from Unified"
echo "================================================================================"
echo ""

if [ ! -f "prepare_top20_dataset.py" ]; then
    echo "❌ Error: prepare_top20_dataset.py not found!"
    exit 1
fi

python3 prepare_top20_dataset.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to prepare top 20 dataset"
    exit 1
fi

echo ""
echo "✅ Top 20 dataset prepared successfully"
echo ""

# Step 2: Verify datasets exist
echo "================================================================================"
echo "🔍 STEP 2: Verifying Dataset Structure"
echo "================================================================================"
echo ""

for split in top20_train top20_val top20_test; do
    dir="data/$split"
    if [ -d "$dir" ]; then
        class_count=$(ls -d $dir/*/ 2>/dev/null | wc -l)
        echo "  ✓ $split: $class_count classes"
    else
        echo "  ❌ $split: NOT FOUND"
        exit 1
    fi
done

echo ""
echo "✅ All datasets verified"
echo ""

# Step 3: Check training script
echo "================================================================================"
echo "🔍 STEP 3: Verifying Training Script"
echo "================================================================================"
echo ""

if [ ! -f "train_convnext_production.py" ]; then
    echo "❌ Error: train_convnext_production.py not found!"
    exit 1
fi

# Check if script has correct configuration
grep -q "convnext_large" train_convnext_production.py && \
grep -q "top20_train" train_convnext_production.py && \
grep -q "batch_size = 48" train_convnext_production.py

if [ $? -eq 0 ]; then
    echo "  ✓ Training script configured correctly"
    echo "    - Model: ConvNeXt Large"
    echo "    - Dataset: top20_train/val"
    echo "    - Batch size: 48"
else
    echo "  ⚠️  Warning: Training script may not be configured correctly"
fi

echo ""

# Step 4: Activate conda environment
echo "================================================================================"
echo "🐍 STEP 4: Activating Conda Environment"
echo "================================================================================"
echo ""

eval "$(/dist_home/suryansh/miniforge3/bin/conda shell.bash hook)"
conda activate skindoc

if [ $? -eq 0 ]; then
    echo "  ✓ Conda environment 'skindoc' activated"
else
    echo "  ❌ Error: Failed to activate conda environment"
    exit 1
fi

# Verify PyTorch
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}'); print(f'  ✓ CUDA available: {torch.cuda.is_available()}')"

if [ $? -ne 0 ]; then
    echo "  ❌ Error: PyTorch not available"
    exit 1
fi

# Verify timm
python3 -c "import timm; print(f'  ✓ timm {timm.__version__}')"

if [ $? -ne 0 ]; then
    echo "  ❌ Error: timm not available"
    exit 1
fi

echo ""

# Step 5: Check GPU availability
echo "================================================================================"
echo "🎮 STEP 5: GPU Status"
echo "================================================================================"
echo ""

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
    echo ""
else
    echo "  ⚠️  nvidia-smi not available (will check when job starts)"
    echo ""
fi

# Step 6: Create necessary directories
echo "================================================================================"
echo "📁 STEP 6: Creating Output Directories"
echo "================================================================================"
echo ""

mkdir -p logs
mkdir -p checkpoints/convnext_large_top20
mkdir -p runs/convnext_large_top20

echo "  ✓ logs/"
echo "  ✓ checkpoints/convnext_large_top20/"
echo "  ✓ runs/convnext_large_top20/"
echo ""

# Step 7: Check for existing training jobs
echo "================================================================================"
echo "⏳ STEP 7: Checking Existing Jobs"
echo "================================================================================"
echo ""

if command -v squeue &> /dev/null; then
    existing_jobs=$(squeue -u $USER -o "%.18i %.9P %.30j %.8T" | grep -c "convnext\|cvnxt" || true)
    
    if [ $existing_jobs -gt 0 ]; then
        echo "  ⚠️  Found $existing_jobs existing ConvNeXt job(s):"
        squeue -u $USER | grep -E "JOBID|convnext|cvnxt" || true
        echo ""
        read -p "  Do you want to cancel them before submitting new job? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            squeue -u $USER -o "%.18i" | grep -v JOBID | xargs -r scancel
            echo "  ✓ Cancelled existing jobs"
        fi
    else
        echo "  ✓ No existing ConvNeXt jobs found"
    fi
else
    echo "  ℹ️  squeue not available"
fi

echo ""

# Step 8: Submit job
echo "================================================================================"
echo "🚀 STEP 8: Submitting Training Job"
echo "================================================================================"
echo ""

if [ ! -f "submit_training.slurm" ]; then
    echo "❌ Error: submit_training.slurm not found!"
    exit 1
fi

# Make sure script is executable
chmod +x submit_training.slurm

# Submit the job
job_output=$(sbatch submit_training.slurm 2>&1)
job_status=$?

if [ $job_status -eq 0 ]; then
    job_id=$(echo $job_output | grep -oP '\d+' | head -1)
    echo "✅ Job submitted successfully!"
    echo ""
    echo "  Job ID: $job_id"
    echo "  Job Name: cvnxt_l_top20"
    echo ""
    echo "📊 Monitor your job:"
    echo "  • Status:  squeue -j $job_id"
    echo "  • Live:    tail -f logs/train_${job_id}.log"
    echo "  • Errors:  tail -f logs/train_${job_id}.err"
    echo ""
    echo "📈 TensorBoard (after job starts):"
    echo "  tensorboard --logdir=runs/convnext_large_top20 --port=6006"
    echo ""
else
    echo "❌ Error: Failed to submit job"
    echo "$job_output"
    exit 1
fi

echo "================================================================================"
echo "✅ SETUP COMPLETE - Training Started!"
echo "================================================================================"
echo ""
echo "Configuration Summary:"
echo "  • Model: ConvNeXt Large (~198M parameters)"
echo "  • Dataset: Top 20 classes from unified (~110,000 images)"
echo "  • Batch Size: 48"
echo "  • Epochs: 50"
echo "  • Expected Time: 10-12 hours on RTX 6000 Ada"
echo "  • Expected Accuracy: 90-94%"
echo ""
echo "Output:"
echo "  • Checkpoints: checkpoints/convnext_large_top20/"
echo "  • Logs: logs/train_${job_id}.log"
echo "  • TensorBoard: runs/convnext_large_top20/"
echo ""
echo "================================================================================"
