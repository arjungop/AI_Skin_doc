# Skin Disease Classification - A100 Training Setup

## � QUICK START (Recommended)

**Complete setup in 3 commands:**

```bash
# 1. On your university A100 server
bash scripts/server_setup.sh

# 2. Validate everything is ready (IMPORTANT - saves queue time!)
python scripts/validate_setup.py

# 3. Start training (runs data prep + training automatically)
bash scripts/run_complete_training.sh efficientnet_b4 64 50
```

That's it! The scripts handle everything automatically.

---

## 📋 What the Scripts Do

### `server_setup.sh` - Initial Server Setup
- [OK] Creates **separate** conda environment (NEVER uses base)
- [OK] Installs PyTorch with CUDA 12.1 for A100
- [OK] Configures Kaggle API
- [OK] Downloads all datasets with retry logic
- [OK] Validates downloads

### `validate_setup.py` - Pre-Training Checks
**RUN THIS BEFORE TRAINING** to catch issues early!
- [OK] Verifies conda environment (not base)
- [OK] Checks CUDA and A100 GPU
- [OK] Validates datasets are downloaded
- [OK] Checks disk space
- [OK] Tests Kaggle API

### `run_complete_training.sh` - Complete Pipeline
- [OK] Activates correct conda environment
- [OK] Prepares unified dataset
- [OK] Starts training with logging
- [OK] Handles interrupts gracefully

---

## 📦 Dataset Download Instructions

### Step 1: Setup Kaggle API

```bash
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token" (downloads kaggle.json)
# 3. Setup the credentials:

mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Install Dependencies

```bash
cd ~/Skin-Doc
pip install kaggle gdown tqdm requests
```

### Step 3: Download Datasets

**AUTOMATIC (Recommended):**
```bash
bash scripts/server_setup.sh
```

**OR MANUAL:**

```bash
# ISIC 2019 (~25K images, 8 classes) - ESSENTIAL
kaggle datasets download -d andrewmvd/isic-2019 -p datasets/isic_data/isic_2019 --unzip

# HAM10000 (~10K images, 7 classes) - ESSENTIAL  
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p datasets/ham10000 --unzip

# DermNet (~19K images, 23 classes) - ESSENTIAL
kaggle datasets download -d shubhamgoel27/dermnet -p datasets/dermnet_main --unzip

# Fitzpatrick17k (~17K images, diverse skin tones) - RECOMMENDED
kaggle datasets download -d mmaximillian/fitzpatrick17k-images -p datasets/data --unzip

# PAD-UFES-20 (~2.3K images, diverse skin tones) - RECOMMENDED
kaggle datasets download -d mahdavi1202/pad-ufes-20 -p datasets/pad_ufes_20 --unzip

# MASSIVE DATASET (262K images, 34 classes) - OPTIONAL (50GB+)
kaggle datasets download -d kylegraupe/skin-disease-balanced-dataset -p datasets/massive_skin_disease --unzip
```

### Step 4: Prepare Unified Dataset

```bash
python scripts/prepare_unified_dataset.py
```

---

##  A100 Training Commands

### Option 1: Automated (Recommended)

```bash
# Validate setup first (IMPORTANT!)
python scripts/validate_setup.py

# Run complete pipeline
bash scripts/run_complete_training.sh [backbone] [batch_size] [epochs]

# Examples:
bash scripts/run_complete_training.sh efficientnet_b4 64 50
bash scripts/run_complete_training.sh swin_b 48 100
```

### Option 2: Manual Control

```bash
# 1. Prepare data
python scripts/prepare_unified_dataset_v2.py

# 2. Validate before training (saves queue time!)
python scripts/validate_setup.py

# 3. Train
python scripts/train_a100.py \
  --backbone efficientnet_b4 \
  --batch_size 64 \
  --image_size 384 \
  --epochs 50 \
  --use_weighted_sampling
```

### Recommended Configurations for A100 40GB

**Fast & Balanced (6 hours):**
```bash
python scripts/train_a100.py \
  --backbone efficientnet_b4 \
  --batch_size 64 \
  --image_size 384 \
  --epochs 50 \
  --use_weighted_sampling
```

**High Accuracy (8 hours):**
```bash
python scripts/train_a100.py \
  --backbone convnext_large \
  --batch_size 32 \
  --image_size 384 \
  --epochs 75 \
  --gradient_accumulation 4 \
  --use_weighted_sampling
```

**Best Results (10-12 hours):**
```bash
python scripts/train_a100.py \
  --backbone swin_b \
  --batch_size 48 \
  --image_size 384 \
  --epochs 100 \
  --patience 15 \
  --use_weighted_sampling
```

---

## [WARNING] CRITICAL REMINDERS

### Before Submitting to Queue:

1. **ALWAYS validate first:**
   ```bash
   python scripts/validate_setup.py
   ```

2. **NEVER use base conda environment:**
   ```bash
   # WRONG - Will cause issues
   conda activate base
   
   # CORRECT
   conda activate skindoc
   ```

3. **Check you have enough disk space:**
   - Minimum: 10GB for checkpoints
   - Recommended: 50GB+

4. **Use the automated script to avoid failures:**
   ```bash
   bash scripts/run_complete_training.sh
   ```

---

## 📊 Expected Results

| Backbone | Batch Size | Image Size | Expected Accuracy | Training Time |
|----------|-----------|------------|-------------------|---------------|
| EfficientNet-B4 | 64 | 384 | 85-90% | ~6 hours |
| ConvNeXt-Base | 48 | 384 | 87-92% | ~8 hours |
| Swin-Base | 48 | 384 | 88-93% | ~10 hours |
| Swin-Large | 32 | 448 | 90-95% | ~15 hours |

---

## 📁 Final Directory Structure

```
datasets/
├── isic_data/
│   └── isic_2019/           # ISIC 2019 Challenge
├── ham10000/                 # HAM10000 Dataset
├── dermnet_main/             # DermNet (23 classes)
├── data/
│   ├── fitzpatrick17k.csv
│   └── finalfitz17k/         # Fitzpatrick17k images
├── pad_ufes_20/              # PAD-UFES-20
├── massive_skin_disease/     # Optional: 262K images
└── downloads/                # Temporary downloads

data/
├── unified_train/            # Training set (after prepare_unified_dataset.py)
│   ├── melanoma/
│   ├── bcc/
│   ├── psoriasis/
│   └── ... (19 disease folders)
└── unified_val/              # Validation set

checkpoints/
├── best_model.pth            # Best model weights
└── checkpoint_epoch_N.pth    # Epoch checkpoints
```

---

##  Target Disease Coverage (19 Diseases)

| Category | Diseases | Dataset Sources |
|----------|----------|-----------------|
| **Cancer** | melanoma, bcc, scc, ak | ISIC, HAM10000, DermNet |
| **Benign** | nevus, seborrheic_keratosis, angioma, wart | ISIC, HAM10000, DermNet |
| **Inflammatory** | eczema, psoriasis, lichen_planus, urticaria | DermNet, Fitzpatrick |
| **Infectious** | impetigo, herpes, candida, scabies | DermNet, Fitzpatrick |
| **Pigmentary** | vitiligo, melasma, hyperpigmentation | DermNet, Fitzpatrick |

---

## [WARNING] Notes

1. **ISIC & HAM10000** are primarily dermoscopic images (skin cancer focused)
2. **DermNet & Fitzpatrick** provide clinical images for inflammatory/infectious diseases
3. **Massive dataset** is optional but highly recommended for best results
4. Run training on A100 with **mixed precision (bf16)** for optimal performance
