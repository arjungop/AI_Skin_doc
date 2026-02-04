# Deploy to University Server - Quick Guide

## Your folder is already created on the server [OK]

## Step 1: Configure Server Details

Edit [scripts/push_to_server.sh](scripts/push_to_server.sh) and set your server details:

```bash
SERVER_USER="your_username"        # Your university username
SERVER_HOST="your.server.edu"      # Server address
SERVER_PATH="~/Skin-Doc"           # Path to your existing folder
```

Or the script will prompt you interactively.

## Step 2: Push Code to Server

```bash
bash scripts/push_to_server.sh
```

This syncs:
- [OK] All scripts
- [OK] Backend code
- [OK] Training configurations
- [ERROR] NOT datasets (downloaded on server)
- [ERROR] NOT large model weights

## Step 3: SSH to Server

```bash
ssh your_username@your.server.edu
cd Skin-Doc
```

## Step 4: Run Setup (One-Time)

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run server setup (downloads datasets, creates conda env)
bash scripts/server_setup.sh
```

This will:
1. Create conda environment `skindoc`
2. Install PyTorch with CUDA 12.1
3. Download all 7 datasets (~60GB)
4. Verify all imports

**Expected time**: 30-60 minutes (depending on download speed)

## Step 5: Validate Setup

```bash
conda activate skindoc
python scripts/validate_setup.py
```

Should show:
- [OK] GPU detected (A100 or A6000)
- [OK] All packages installed
- [OK] CUDA working
- [OK] Datasets present

## Step 6: Submit Training Job

```bash
# Submit with SLURM (48-hour window)
bash scripts/submit_slurm.sh swin_b 64 100 a100
```

Options:
- `swin_b` - Model (swin_t, swin_s, swin_b, swin_large)
- `64` - Batch size
- `100` - Epochs
- `a100` - GPU type (a100, a6000, or auto)

## Step 7: Monitor Training

```bash
# Check queue
squeue -u $USER

# View live output
tail -f logs/slurm_<job_id>.out

# Check GPU usage (if job is running)
ssh <node_name>
nvidia-smi
```

## Quick Reference

### After First Setup

Once setup is complete, to run new training:

```bash
# Login
ssh your_username@your.server.edu
cd Skin-Doc

# Activate environment
conda activate skindoc

# Submit job
bash scripts/submit_slurm.sh swin_b 64 100
```

### Update Code from Local

If you make changes locally and want to sync:

```bash
# On your Mac
bash scripts/push_to_server.sh

# On server (if needed)
conda activate skindoc
python scripts/validate_setup.py
```

### Cancel Job

```bash
scancel <job_id>
```

### Re-run Training

```bash
# Different model
bash scripts/submit_slurm.sh swin_large 48 100 a100

# Different epochs
bash scripts/submit_slurm.sh swin_b 64 50 a100
```

## Expected Training Times (48h walltime for all)

| Model | Epochs | A100 Time | A6000 Time | Expected Accuracy |
|-------|--------|-----------|------------|------------------|
| swin_tiny | 100 | 8-12h | 12-16h | 86-90% |
| swin_small | 100 | 10-14h | 14-18h | 88-92% |
| **swin_base** | 100 | 12-16h | 16-20h | **89-93%** |
| swin_large | 100 | 18-24h | 24-30h | 90-94% |

## Troubleshooting

### Can't SSH to Server
```bash
# Check VPN connection (if required)
# Verify server address and username
ssh -v your_username@your.server.edu
```

### Setup Script Fails
```bash
# Check conda is installed
which conda

# Check internet access
ping google.com

# Check disk space
df -h
```

### Job Stuck in Queue
```bash
# Check queue position
squeue -u $USER --start

# Check available GPUs
sinfo -o "%20N %10c %10m %25f %10G"
```

### Out of Disk Space
```bash
# Check usage
du -sh ~/Skin-Doc/*

# Clean old logs
rm -rf logs/slurm_*.out

# Clean cache
rm -rf ~/.cache/torch
```

## Files Generated on Server

After training completes:
- `backend/ml/weights/skin_<model>_final.pth` - Trained model
- `logs/training_*.log` - Training metrics
- `logs/slurm_*.out` - Job output
- `logs/slurm_script_*.sh` - Submitted SLURM scripts

## Next Steps After Training

1. Download trained model:
```bash
# On your Mac
scp your_username@server.edu:~/Skin-Doc/backend/ml/weights/skin_swin_b_final.pth backend/ml/weights/
```

2. Test locally:
```bash
python scripts/inference.py --image test.jpg --model swin_b
```

3. Deploy to production (your app server)
