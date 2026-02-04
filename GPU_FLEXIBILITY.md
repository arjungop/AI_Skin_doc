# GPU Flexibility Guide (A100 / A6000)

##  Your Setup Now Works With BOTH GPUs!

Your scripts automatically detect and optimize for either **A100** or **A6000** based on availability.

---

##  Quick Start (GPU Agnostic)

```bash
# This works on EITHER A100 or A6000
bash scripts/run_gpu_optimized.sh swin_b 64 100
```

The script will:
- [OK] Auto-detect GPU type (A100 or A6000)
- [OK] Optimize batch sizes automatically
- [OK] Set appropriate mixed precision
- [OK] Adjust settings for best performance

---

## 📊 Automatic Optimizations

### A100 Detected:
```
[OK] A100 detected - Using optimal settings
   Batch size: 64 (full)
   Mixed precision: bfloat16 (bf16)
   Expected time: ~10 hours (Swin-Base)
```

### A6000 Detected:
```
[OK] A6000 detected - Adjusting for workstation GPU
   Batch size: 54 (85% of 64)
   Mixed precision: float16 (fp16)
   Expected time: ~14 hours (Swin-Base)
```

---

##  Recommended Configurations

### For Either GPU:

**Fast Training (6-8 hours):**
```bash
bash scripts/run_gpu_optimized.sh efficientnet_b4 64 50
```

**Best Balance (10-14 hours):**
```bash
bash scripts/run_gpu_optimized.sh swin_b 64 100
```

**Maximum Accuracy (15-20 hours):**
```bash
bash scripts/run_gpu_optimized.sh swin_v2_b 48 150
```

---

## 📋 Performance Comparison

| Model | A100 Time | A6000 Time | Accuracy |
|-------|-----------|------------|----------|
| EfficientNet-B4 | ~6 hours | ~8 hours | 85-90% |
| ConvNeXt-Base | ~8 hours | ~11 hours | 87-92% |
| Swin-Base | ~10 hours | ~14 hours | 88-93% |
| Swin-Large | ~15 hours | ~20 hours | 90-95% |

---

## 💡 Queue Strategy

### Check Both Queues:

```bash
# Check A100 queue
squeue -p a100

# Check A6000 queue  
squeue -p a6000

# Use whichever is shorter!
```

### Submit to Available GPU:

```bash
# Example SLURM job
#SBATCH --gres=gpu:1  # Any GPU (auto-optimizes)
# OR
#SBATCH --gres=gpu:a100:1  # Specifically A100
# OR
#SBATCH --gres=gpu:a6000:1  # Specifically A6000
```

---

## [OK] What Changed

1. **Auto GPU Detection** - Scripts detect A100 vs A6000
2. **Batch Size Optimization** - A6000 uses 85% batch size
3. **Mixed Precision Selection** - bf16 for A100, fp16 for A6000
4. **Validation Updates** - Works with both GPUs
5. **New Script** - `run_gpu_optimized.sh` handles everything

---

##  Your Workflow

```bash
# 1. Setup (works on any node)
bash scripts/server_setup.sh

# 2. Validate (checks whatever GPU you have)
python scripts/validate_setup.py

# 3. Train (auto-optimizes for your GPU)
bash scripts/run_gpu_optimized.sh swin_b 64 100
```

---

## 💡 Pro Tip

**Use whichever queue is shorter!** Both GPUs work great:

- **A100:** Faster (10 hours) ⚡
- **A6000:** More memory (48GB), slightly slower (14 hours) 🧠

**The scripts optimize automatically - just pick the shorter queue!** 
