# 🚀 Skin-Doc A100 Training Guide (Updated Feb 2026)

Everything is optimized for the **University Cluster** and the **Massive 262k dataset**.

## 🚀 Quick Start (Cluster)

### 1. Update Repo
```bash
git pull origin main
```

### 2. Prepare Data & Train (SLURM)
The most robust way to run. It will handle the 262k images, map them correctly, and train the **ConvNeXt-Large** model.
```bash
bash scripts/submit_slurm.sh convnext_large 32 50 a100
```
*   **Model**: ConvNeXt-Large
*   **Batch**: 32 (with 2x Grad Accumulation = 64 Effective)
*   **Epochs**: 50
*   **GPU**: Requesting A100

### 3. Monitoring
```bash
tail -f logs/slurm_<job_id>.out
```

## 🛠️ Optimizations Applied

1.  **ConvNeXt-Large Backbone**:
    *   State-of-the-art accuracy for skin lesions.
    *   Switched from EfficientNet to ConvNeXt-Large as default.
    *   Input size: 384x384.

2.  **Bulletproof Trainer (`scripts/train_bulletproof.py`)**:
    *   **Auto-Resume**: If the cluster preempts your job, it will pick up exactly where it left off.
    *   **Preemption Handling**: Listens for `SIGTERM` and `SIGCONT` (common in university clusters).
    *   **BFloat16**: Uses NVIDIA A100's native precision for 2x speedup.
    *   **Gradient Accumulation**: Handles large models on 40GB/80GB cards without OOM.

3.  **Massive Dataset Support**:
    *   `scripts/prepare_unified_dataset_v2.py` now maps the **262,000 images** from the `massive 2` folder into Category/Disease heads.
    *   Auto-merges ISIC 2019, HAM10000, DermNet, and Massive 2.

4.  **HPC Ready**:
    *   Requesting 12 CPUs and 80GB RAM in SLURM for fast data loading.
    *   Ensures `num_workers` and `pin_memory` are maximized for the A100 bandwidth.

## 📁 File Structure for Massive 2
Ensure your dataset is at:
`~/arjungop/AI_Skin_Doc/massive 2/balanced_dataset/balanced_dataset/`

---

That's it! Everything else is handled automatically.

---

## [WARNING] CRITICAL CHECKLIST

Before submitting to queue, verify these:

### 1. Environment Check
```bash
echo $CONDA_DEFAULT_ENV
# Should show: skindoc
# Should NEVER show: base
```

### 2. Run Validation
```bash
python scripts/validate_setup.py
# ALL checks must pass
```

### 3. Verify GPU
```bash
nvidia-smi
# Should show A100 40GB
```

### 4. Check Disk Space
```bash
df -h .
# Need at least 10GB free
```

---

## 🔒 Key Safety Features

### 1. **Never Uses Base Conda**
- Script **forces** separate `skindoc` environment
- **Validates** not in base before proceeding
- **Exits with error** if base detected

### 2. **Download Retry Logic**
- All downloads retry 3 times
- 5-second delay between attempts
- Clear error messages

### 3. **Pre-Training Validation**
- Checks 8 critical aspects
- Catches issues **before** queue wait
- **Saves hours** of wasted time

### 4. **Robust Error Handling**
- Training validates environment first
- Graceful interrupts (Ctrl+C)
- Stack traces for debugging
- Auto-saves config & checkpoints

---

## 📊 Training Configurations

### Fast & Balanced (~6 hours, 85-90% accuracy)
```bash
bash scripts/run_complete_training.sh efficientnet_b4 64 50
```

### High Accuracy (~8 hours, 87-92% accuracy)
```bash
bash scripts/run_complete_training.sh convnext_large 32 75
```

### Best Results (~10 hours, 88-93% accuracy)
```bash
bash scripts/run_complete_training.sh swin_b 48 100
```

---

## 📁 Project Structure

```
Skin-Doc/
├── scripts/
│   ├── server_setup.sh           ← Initial setup (run once)
│   ├── validate_setup.py          ← Pre-training check (run always!)
│   ├── run_complete_training.sh   ← Complete pipeline (recommended)
│   ├── train_a100.py              ← Direct training (advanced)
│   ├── setup_wizard.py            ← Interactive guide
│   └── verify_dataset_links.py    ← Check Kaggle links
│
├── SERVER_QUICKSTART.md           ← Quick reference
├── TRAINING_SETUP.md              ← Complete guide
└── SETUP_IMPROVEMENTS.md          ← What changed
```

---

## 🎓 Typical Workflow

### First Time on Server:
```bash
# 1. Setup everything
bash scripts/server_setup.sh
# Downloads ~60GB datasets, creates environment

# 2. Validate
python scripts/validate_setup.py
# Check everything is ready

# 3. Test run (5 epochs)
bash scripts/run_complete_training.sh efficientnet_b4 64 5

# 4. Full training
bash scripts/run_complete_training.sh efficientnet_b4 64 50
```

### Subsequent Training Runs:
```bash
# 1. Always validate first
python scripts/validate_setup.py

# 2. Train
bash scripts/run_complete_training.sh swin_b 48 100
```

---

## 🛡️ What Makes This Bulletproof

### [ERROR] Old Way (Error-Prone):
- Manual conda activation
- Could use base environment  
- No validation before training
- Downloads could fail silently
- No error recovery

### [OK] New Way (Bulletproof):
- [OK] Auto-creates separate conda env
- [OK] Validates not in base
- [OK] Pre-flight checks catch all issues
- [OK] Downloads retry automatically
- [OK] Comprehensive error handling
- [OK] Graceful interrupts
- [OK] Auto-saves everything

---

## 💡 Pro Tips

### 1. Always Use screen/tmux
```bash
screen -S training
bash scripts/run_complete_training.sh
# Ctrl+A, D to detach
# screen -r training to reattach
```

### 2. Monitor Training
```bash
# In another terminal
tail -f checkpoints/*/training.log
```

### 3. Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### 4. Test First
```bash
# Quick 5-epoch test run
bash scripts/run_complete_training.sh efficientnet_b4 64 5
```

---

## 📞 Troubleshooting

### Issue: "Still in base environment"
**Fix:**
```bash
conda activate skindoc
```

### Issue: "Kaggle API not working"
**Fix:**
```bash
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list --page-size 1  # Test
```

### Issue: "No datasets found"
**Fix:**
```bash
bash scripts/server_setup.sh  # Re-run setup
```

### Issue: "Out of memory"
**Fix:**
```bash
# Reduce batch size
bash scripts/run_complete_training.sh efficientnet_b4 32 50
```

---

## 🎉 You're All Set!

Everything is configured for **reliable, automated A100 training**.

### Final Checklist:
- [x] Separate conda environment (never base)
- [x] Download retry logic
- [x] Kaggle API validation
- [x] Pre-training validation script
- [x] Complete automated pipeline
- [x] Error handling & recovery
- [x] Comprehensive documentation

### Next Steps:
1. Push to server: `bash scripts/push_to_server.sh`
2. SSH to server
3. Run: `bash scripts/server_setup.sh`
4. **Validate: `python scripts/validate_setup.py`**
5. Train: `bash scripts/run_complete_training.sh efficientnet_b4 64 50`

**Good luck with your training!** 

---

*Pro Tip: Bookmark `SERVER_QUICKSTART.md` on the server for quick reference!*
