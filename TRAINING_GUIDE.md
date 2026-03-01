# ConvNeXt Production Training - Quick Start Guide

## 📋 Overview

Production-ready training setup for ConvNeXt Base on skin disease classification (29 classes).

**Configuration:**
- Model: ConvNeXt Base (~89M parameters)
- Dataset: DermNet main (29 classes)
- GPU: RTX 6000 Ada (48GB VRAM)
- Batch Size: 64
- Epochs: 50
- Learning Rate: 4e-3

## 🚀 Quick Start

### 1. Submit SLURM Job

```bash
sbatch submit_training.slurm
```

### 2. Monitor Training

```bash
# Watch job status
squeue -u suryansh

# View live logs
tail -f logs/train_<JOB_ID>.log

# View errors (if any)
tail -f logs/train_<JOB_ID>.err
```

### 3. TensorBoard (Optional)

```bash
# On server or via SSH tunnel
tensorboard --logdir=runs/convnext_base_production --port=6006

# SSH tunnel from local machine:
ssh -L 6006:localhost:6006 suryansh@172.17.16.11
# Then open: http://localhost:6006
```

## 📂 Output Files

```
checkpoints/convnext_base_production/
├── best_model.pth              # Best model (highest val accuracy)
├── checkpoint_latest.pth       # Latest checkpoint (for resuming)
├── checkpoint_epoch_5.pth      # Periodic checkpoints
├── checkpoint_epoch_10.pth
├── ...
├── class_mapping.json          # Class labels and indices
├── config.json                 # Training configuration
└── history.json                # Training metrics history

logs/
├── train_<JOB_ID>.log         # Training stdout
└── train_<JOB_ID>.err         # Training stderr

runs/convnext_base_production/  # TensorBoard logs
```

## 🔧 Advanced Usage

### Resume Training

Training automatically resumes from `checkpoint_latest.pth` if it exists.

To force a fresh start:
```bash
python3 train_convnext_production.py --fresh_start
```

### Custom Configuration

```bash
python3 train_convnext_production.py \
    --model_name convnext_base \
    --batch_size 64 \
    --epochs 50 \
    --lr 4e-3 \
    --img_size 384 \
    --data_dir /path/to/data \
    --save_dir checkpoints/my_experiment
```

### Run Without SLURM (Direct Execution)

```bash
python3 train_convnext_production.py
```

## 📊 Model Evaluation

After training, evaluate on test set:

```bash
python3 scripts/eval_skin.py \
    --model_path checkpoints/convnext_base_production/best_model.pth \
    --test_dir data/main_test
```

## 🛠️ Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size:
   ```bash
   python3 train_convnext_production.py --batch_size 32
   ```

2. Enable gradient checkpointing (edit script):
   ```python
   gradient_checkpointing = True
   ```

### Job Killed/Timeout

Increase time limit in SLURM script:
```bash
#SBATCH --time=72:00:00  # 72 hours
```

### Slow Data Loading

Reduce num_workers:
```bash
python3 train_convnext_production.py --num_workers 8
```

## 📈 Expected Results

Based on ConvNeXt Base + DermNet main dataset:

- **Training Time:** ~6-8 hours for 50 epochs (RTX 6000 Ada)
- **Expected Val Accuracy:** 85-92%
- **Best Results:** Usually around epoch 30-40

## 🔄 Next Steps After Training

1. **Evaluate on test set** using `scripts/eval_skin.py`
2. **Deploy model** to backend (see `backend/ml/model.py`)
3. **Fine-tune** if needed with different hyperparameters
4. **Ensemble** with other models for better performance

## 📧 Notifications

Edit email in SLURM script:
```bash
#SBATCH --mail-user=your.email@example.com
```

## ⚙️ Files Overview

- `train_convnext_production.py` - Main training script
- `submit_training.slurm` - SLURM job submission script
- `requirements_training.txt` - Python dependencies

## 💡 Tips

- Check GPU utilization: `nvidia-smi -l 1`
- Monitor disk space: `df -h`
- Clean old checkpoints periodically
- Save TensorBoard logs for comparing experiments
- Use `--seed` for reproducibility

---

**Created:** 2026-02-15  
**Optimized for:** RTX 6000 Ada (48GB VRAM)  
**Dataset:** DermNet main (29 classes)
