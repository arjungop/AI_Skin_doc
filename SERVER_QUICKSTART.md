# A100 Server Training - Quick Guide

## 🎯 Complete Training in 3 Steps

### 1️⃣ Run Server Setup (Downloads datasets, creates environment)
```bash
cd ~/Skin-Doc
bash scripts/server_setup.sh
```

**What it does:**
- Creates **separate conda environment** called `skindoc` (NEVER uses base)
- Installs PyTorch with CUDA 12.1 for A100
- Downloads all datasets from Kaggle (with retry logic)
- Validates everything is ready

**Before running:** Download `kaggle.json` from https://www.kaggle.com/settings

---

### 2️⃣ Validate Setup (CRITICAL - Run before queue submission!)
```bash
python scripts/validate_setup.py
```

**What it checks:**
- ✅ Not using base conda environment
- ✅ CUDA working and A100 detected
- ✅ All datasets downloaded
- ✅ Enough disk space
- ✅ Kaggle API working

**This catches issues BEFORE wasting queue time!**

---

### 3️⃣ Start Training
```bash
bash scripts/run_complete_training.sh efficientnet_b4 64 50
#                                    [backbone]  [batch] [epochs]
```

**What it does:**
- Activates correct conda environment
- Prepares unified dataset
- Trains model with logging
- Saves best checkpoint
- Handles interrupts gracefully

---

## 📊 Training Options

### Fast Training (~6 hours, 85-90% accuracy)
```bash
bash scripts/run_complete_training.sh efficientnet_b4 64 50
```

### High Accuracy (~10 hours, 88-93% accuracy)
```bash
bash scripts/run_complete_training.sh swin_b 48 100
```

### Custom Training
```bash
python scripts/train_a100.py \
  --backbone efficientnet_b4 \
  --batch_size 64 \
  --epochs 50 \
  --use_weighted_sampling
```

**Available backbones:**
- `efficientnet_b4` - Best balance (recommended)
- `convnext_large` - Higher accuracy
- `swin_b` - Highest accuracy potential
- `resnet50` - Fast baseline

---

## ⚠️ IMPORTANT WARNINGS

### ❌ NEVER use base conda environment
```bash
# WRONG
conda activate base

# CORRECT
conda activate skindoc
```

### ✅ ALWAYS validate before queue submission
```bash
python scripts/validate_setup.py
```

This saves hours of queue time if something is wrong!

---

## 🔍 Troubleshooting

### Dataset download fails
```bash
# Check Kaggle credentials
ls -la ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Test API
kaggle datasets list --page-size 1
```

### Out of memory during training
```bash
# Reduce batch size
bash scripts/run_complete_training.sh efficientnet_b4 32 50
```

### Training interrupted
Don't worry! Checkpoints are saved every epoch. Resume by running again - it will use existing data.

---

## 📂 Output Files

After training:
```
checkpoints/YYYYMMDD_HHMMSS_backbone/
├── best_model.pth          # Best model checkpoint
├── checkpoint_epoch_XX.pth # Latest checkpoint
├── config.json             # Training configuration
└── training.log            # Complete training log
```

---

## 💡 Tips

1. **Use validation script** before every training run
2. **Monitor training** with: `tail -f checkpoints/*/training.log`
3. **Start with smaller model** (efficientnet_b4) to test setup
4. **Use screen/tmux** for long training sessions
5. **Check disk space** regularly during training

---

## 🆘 Quick Commands

```bash
# Check environment
conda env list

# Activate environment
conda activate skindoc

# Check GPU
nvidia-smi

# Validate setup
python scripts/validate_setup.py

# Start training
bash scripts/run_complete_training.sh

# Monitor training
tail -f checkpoints/*/training.log
```

---

## 📞 Need Help?

1. Run validation: `python scripts/validate_setup.py`
2. Check the output - it tells you what's wrong
3. Fix the issue
4. Validate again
5. Start training

Good luck! 🚀
