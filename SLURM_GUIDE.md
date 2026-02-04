# SLURM Job Submission Guide - Swin Transformers

## Quick Start

### Submit with Auto-Detected GPU
```bash
bash scripts/submit_slurm.sh swin_b 64 100
```

### Submit with Specific GPU Type
```bash
# Request A100 specifically
bash scripts/submit_slurm.sh swin_b 64 100 a100

# Request A6000 specifically
bash scripts/submit_slurm.sh swin_s 64 50 a6000

# Swin Large model
bash scripts/submit_slurm.sh swin_large 64 100 a100
```

## Walltime Allocation

**Fixed 48-hour window for all Swin Transformer models** - provides ample time for training completion with large buffer.

### Available Swin Models

| Model | Parameters | Expected Training Time (100 epochs) |
|-------|-----------|-------------------------------------|
| swin_tiny (swin_t) | ~28M | ~8-12 hours (A100), ~12-16 hours (A6000) |
| swin_small (swin_s) | ~50M | ~10-14 hours (A100), ~14-18 hours (A6000) |
| swin_base (swin_b) | ~88M | ~12-16 hours (A100), ~16-20 hours (A6000) |
| swin_large (swin_l) | ~197M | ~18-24 hours (A100), ~24-30 hours (A6000) |

**Note**: 48-hour walltime ensures completion even with unexpected delays or slower hardware.

## Resource Allocation

Each job requests:
- **1 GPU** (A100 or A6000)
- **8 CPUs** (for data loading)
- **64GB RAM** (sufficient for all models)
- **Auto-calculated walltime** (based on model + buffer)

## Examples

### Fast Training (ResNet18, 50 epochs)
```bash
bash scripts/submit_slurm.sh resnet18 64 50
# Walltime: 4 hours
```
Swin Tiny (Fastest, 50 epochs)
```bash
bash scripts/submit_slurm.sh swin_t 64 50
# Walltime: 48 hours
```

### Swin Small (Balanced, 100 epochs)
```bash
bash scripts/submit_slurm.sh swin_s 64 100 a100
# Walltime: 48 hours
```

### Swin Base (Recommended, 100 epochs)
```bash
bash scripts/submit_slurm.sh swin_b 64 100 a100
# Walltime: 48 hours
```

### Swin Large (Best Accuracy, 100 epochs)
```bash
bash scripts/submit_slurm.sh swin_large 64 100 a100
# Walltime: 48 hours
# Note: May need reduced batch size (32-48) for memory
```bash
squeue -u $USER
```

### View Live Output
```bash
tail -f logs/slurm_<job_id>.out
```

### Check Job Details
```bash
scontrol show job <job_id>
```

### View GPU Usage
```bash
# SSH to the node running your job
ssh <node_name>
nvidia-smi
```

## Managing Jobs

### Cancel Job
```bash
scancel <job_id>
```

### Cancel All Your Jobs
```bash
scancel -u $USER
```

### Job Priority
```bash
# Check job priority
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %Q"
```

## Output Files

- `logs/slurm_<job_id>.out` - Standard output
- `logs/slurm_<job_id>.err` - Error output
- `logs/slurm_script_<name>_<timestamp>.sh` - Submitted SLURM script (saved for reference)

## Troubleshooting

### Job Pending Too Long
```bash
# Check why job is pending
squeue -u $USER --start
``Fixed 48-hour walltime should be sufficient for all Swin models
- If training exceeds 48 hours, consider reducing epochs or using smaller model
### Out of Memory
- Reduce batch size: `bash scripts/submit_slurm.sh model 32 epochs`
- Or request more memory by editing the script

### Time Limit Exceeded
- The walltime is automatically calculated with buffer
- For custom walltime, edit `scripts/submit_slurm.sh` line 63-65

### GPU Not Available
```bash
# Check available GPUs
sinfo -o "%20N %10c %10m %25f %10G"
```

## Cluster-Specific Settings

Edit `scripts/submit_slurm.sh` lines 99-101 to load modules for your cluster:

```bash
# Example for clusters requiring module loads
module load cuda/12.1
module load cudnn/8.9
module load python/3.11
```

## Advanced Usage

### Request Specific Partition
Edit line 66 in `sSwin model configurations
for model in swin_t swin_s
#SBATCH --partition=gpu-high-priority
```

### Multiple Jobs
```bash
# Submit multiple configurations
for model in resnet50 efficientnet_b4 swin_b; do
    bash scripts/submit_slurm.sh $model 64 100 a100
done
```

### Email Notifications
Add to the SLURM script:
```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@university.edu
```

## Pre-Submission Checklist

[OK] Datasets downloaded? (`bash scripts/server_setup.sh`)  
[OK] Conda environment created? (`conda activate skindoc`)  
[OK] Validation passed? (`python scripts/validate_setup.py`)  
[OK] Logs directory exists? (`mkdir -p logs`)  

## Recommended Workflow

1. **Setup** (one-time):
```bash
bash scripts/server_setup.sh
conda activate skindoc
python scripts/validate_setup.py
```

2. **Submit Job**:
```bash
bash scripts/submit_slurm.sh swin_b 64 100 a100
```

3. **Monitor**:
```bash
squeue -u $USER
tail -f logs/slurm_*.out
```

4. **Results**: Transformers):

| Model | Accuracy | Model Size | Training Time (A100) | Training Time (A6000) |
|-------|----------|------------|---------------------|---------------------|
| Swin-Tiny | 86-90% | ~110MB | 8-12 hours | 12-16 hours |
| Swin-Small | 88-92% | ~200MB | 10-14 hours | 14-18 hours |
| Swin-Base | 89-93% | ~350MB | 12-16 hours | 16-20 hours |
| Swin-Large | 90-94% | ~780MB | 18-24 hours | 24-30 hours |

**All jobs run within 48-hour walltime window**

**Outpu
## Expected Results

After successful training (100 epochs, Swin-Base):
- **Accuracy**: 88-93%
- **Model Size**: ~80MB
- **Training Time**: 10-14 hours (depending on GPU)
- **Checkpoint**: `backend/ml/weights/skin_<model>_final.pth`
