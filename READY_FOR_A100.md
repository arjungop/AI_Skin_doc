# 🚀 Skin-Doc A100 Training Guide (Massive 262k Dataset)

Everything is optimized for the **University Cluster** and the **Massive 262k dataset**.

## 🚀 Quick Start (Cluster)

### 1. Update Repo
```bash
git pull origin main
```

### 2. Prepare Data (Stratified Split)
This new script performs an **Intelligent Stratified Split**. It splits each of the 34 sub-classes into Train (80%), Val (10%), and Test (10%) to ensure even representation.
```bash
python scripts/prepare_unified_dataset_v2.py
```

### 3. Submit SLURM Job
This submits the **ConvNeXt-Large** training job. It will automatically use the `data/unified_train` and `data/unified_val` created in the previous step.
```bash
sbatch scripts/submit_slurm.sh
```

### 4. Monitoring
```bash
# Check queue status
squeue -u $USER

# Watch logs in real-time
tail -f logs/slurm_*.out
```

## 🛠️ Optimizations Applied

1. **ConvNeXt-Large Backbone**:
   - 1536 features, 384x384 resolution.
   - 10x more parameters than ResNet18 for deep feature extraction from dermoscopic images.

2. **Stratified Splitting**:
   - Correctly handles the 34 raw folders in the massive dataset.
   - Maintains class ratios across Train, Val, and Test folders.
   - Creates `data/unified_test` for pure blind evaluation after training.

3. **Bulletproof Trainer**:
   - **Auto-Resume**: Picks up from last epoch if preempted.
   - **BFloat16**: Native A100 precision.
   - **Weighted Sampling**: Balanced batches despite raw class imbalances (Infectious vs Cancer).

4. **HPC Requirements**:
   - Requesting: 1 GPU (A100), 12 CPUs, 80GB RAM.
   - Time limit: 24 hours (Auto-resume handles multi-day runs).

---
*Created by GitHub Copilot for the Skin-Doc Research Team.*
