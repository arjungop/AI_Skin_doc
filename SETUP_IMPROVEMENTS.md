# A100 Training Setup - What Changed

## [OK] All Issues Fixed

Your setup is now **bulletproof** for university A100 training. Here's what was improved:

---

## 🔒 Critical Fixes

### 1. **Conda Environment - NEVER Uses Base**
**Problem:** Could accidentally use base conda environment  
**Solution:** 
- Force creates separate `skindoc` environment
- Validates NOT in base before proceeding
- Shows error and exits if in base

```bash
# In server_setup.sh:
if [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
    echo "ERROR: Still in base environment! Exiting."
    exit 1
fi
```

### 2. **Dataset Download Retry Logic**
**Problem:** Downloads could fail, wasting queue time  
**Solution:**
- All downloads retry 3 times
- 5-second delay between retries
- Clear success/failure messages

```bash
# Retries automatically
download_with_retry "andrewmvd/isic-2019" "isic_2019" "..."
```

### 3. **Kaggle API Validation**
**Problem:** Could start without working Kaggle API  
**Solution:**
- Checks credentials exist
- Verifies permissions (600)
- Tests API before proceeding
- Pauses if missing credentials

### 4. **Pre-Training Validation**
**Problem:** Issues discovered AFTER waiting in queue  
**Solution:** New `validate_setup.py` script checks:
- [OK] Conda environment (not base)
- [OK] CUDA and A100 GPU working
- [OK] All datasets downloaded
- [OK] Disk space sufficient
- [OK] Training scripts present
- [OK] Prepared data exists

**Run before EVERY queue submission!**

### 5. **Comprehensive Error Handling in Training**
**Problem:** Training could fail mid-run  
**Solution:**
- Environment validation before training starts
- Data directory validation
- GPU capability checks
- Graceful error handling with stack traces
- Config saved to checkpoint directory

---

##  New Automated Scripts

### 1. `server_setup.sh` - Enhanced
- Creates separate conda env (never base)
- Retry logic for all downloads
- Kaggle API validation
- Disk space checks

### 2. `run_complete_training.sh` - NEW
Complete pipeline in one command:
```bash
bash scripts/run_complete_training.sh efficientnet_b4 64 50
```

Does everything:
1. Activates correct conda environment
2. Verifies datasets exist
3. Prepares unified dataset (if needed)
4. Starts training with logging
5. Saves best model

### 3. `validate_setup.py` - NEW  
Pre-training validation:
```bash
python scripts/validate_setup.py
```

**Run this BEFORE queue submission** to catch all issues!

### 4. `setup_wizard.py` - NEW
Interactive setup guide:
```bash
python scripts/setup_wizard.py
```

Walks you through entire process step-by-step.

---

## 📚 New Documentation

### 1. `SERVER_QUICKSTART.md` - NEW
Quick reference for server usage with:
- 3-step setup process
- Common troubleshooting
- Important warnings
- Command reference

### 2. `TRAINING_SETUP.md` - Updated
Now includes:
- Automated workflow (recommended)
- Validation instructions
- Critical warnings about conda base
- Complete command examples

---

##  Recommended Workflow

### First Time Setup:
```bash
# 1. Setup server (downloads ~60GB)
bash scripts/server_setup.sh

# 2. Validate everything
python scripts/validate_setup.py

# 3. Start training
bash scripts/run_complete_training.sh efficientnet_b4 64 50
```

### Before Every Training Run:
```bash
# Always validate first!
python scripts/validate_setup.py

# Then train
bash scripts/run_complete_training.sh [backbone] [batch] [epochs]
```

---

## 📊 What Each Script Does

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `server_setup.sh` | Initial server setup | Once, first time |
| `validate_setup.py` | Pre-training checks | Before EVERY training |
| `run_complete_training.sh` | Complete pipeline | Every training run |
| `setup_wizard.py` | Interactive guide | First time (optional) |
| `train_a100.py` | Direct training | Advanced/manual control |

---

## [WARNING] Critical Reminders

### ALWAYS:
1. [OK] Run `validate_setup.py` before queue submission
2. [OK] Use `conda activate skindoc` (NEVER base)
3. [OK] Use automated scripts (less error-prone)
4. [OK] Check disk space (need 10GB+)

### NEVER:
1. [ERROR] Use base conda environment
2. [ERROR] Skip validation before training
3. [ERROR] Assume datasets downloaded correctly
4. [ERROR] Start training without checking GPU

---

## 🔍 Validation Checklist

Before submitting to queue:

```bash
python scripts/validate_setup.py
```

This checks:
- [ ] Not in base conda environment
- [ ] CUDA available with A100
- [ ] All required datasets downloaded
- [ ] Unified dataset prepared
- [ ] Sufficient disk space (10GB+)
- [ ] Kaggle API working
- [ ] Training scripts present

If ALL checks pass → Safe to train!

---

## 💡 Pro Tips

1. **Use screen/tmux** for long training:
   ```bash
   screen -S training
   bash scripts/run_complete_training.sh
   # Ctrl+A, D to detach
   ```

2. **Monitor training** in another terminal:
   ```bash
   tail -f checkpoints/*/training.log
   ```

3. **Check GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Test with small run first**:
   ```bash
   bash scripts/run_complete_training.sh efficientnet_b4 64 5
   # Only 5 epochs to test setup
   ```

---

## 🎉 You're Ready!

Everything is now configured for **reliable, bulletproof A100 training**.

**Next Steps:**
1. Push code to server: `bash scripts/push_to_server.sh`
2. SSH to server
3. Run: `bash scripts/server_setup.sh`
4. Validate: `python scripts/validate_setup.py`
5. Train: `bash scripts/run_complete_training.sh efficientnet_b4 64 50`

Good luck with your training! 
