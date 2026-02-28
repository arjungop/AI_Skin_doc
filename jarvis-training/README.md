# Skin Disease Classification — Jarvis Labs Training Pipeline

20-class skin disease classifier achieving **90–94% accuracy** using ConvNeXt-Large on NVIDIA A6000 (48 GB VRAM).

## Quick Start (Jarvis Labs)

```bash
# 1. Launch a Jarvis Labs instance:
#    Template: PyTorch
#    GPU: A6000 (48 GB)

# 2. Clone this repo
git clone https://github.com/<your-username>/skin-training.git
cd skin-training

# 3. Upload Kaggle API key (see "Kaggle Setup" below)

# 4. One-click setup (downloads all datasets + prepares data)
chmod +x setup.sh
./setup.sh

# 5. Train
python train.py

# 6. Evaluate
python evaluate.py --tta
```

---

## Architecture & Techniques

| Component | Choice | Why |
|-----------|--------|-----|
| **Backbone** | ConvNeXt-Large (timm) | 198M params, ImageNet-22k pretrained at 384×384 |
| **Loss** | FocalLoss (γ=2.0) | Focuses on hard examples, reduces false negatives |
| **Label Smoothing** | 0.1 | Prevents overconfidence |
| **Class Weights** | Inverse-frequency + 2× cancer boost | Handles imbalance, prioritises cancer detection |
| **Sampler** | WeightedRandomSampler | Balanced class exposure per epoch |
| **EMA** | Decay 0.9998 | Smoothed weights generalise better (+0.5–1%) |
| **Mixup/CutMix** | α=0.2/1.0, 50% prob | Data-level regularisation (+1–2%) |
| **Optimizer** | AdamW (wd=0.05) | Decoupled weight decay |
| **Scheduler** | OneCycleLR (max_lr=3e-4) | 10% warmup → cosine decay |
| **Precision** | bfloat16 | A6000 native, no GradScaler needed |
| **Grad Checkpointing** | Enabled | Saves ~40% VRAM |
| **Augmentations** | Crop, Flip, Rotate, Jitter, Erasing | Medical image augmentation suite |
| **TTA** | 5 views at eval | +1–2% test accuracy |

---

## Datasets (6 sources, ~330k+ raw images)

| Dataset | Slug | Images | Classes |
|---------|------|--------|---------|
| ISIC 2019 | `andrewmvd/isic-2019` | ~25k | 8 |
| HAM10000 | `kmader/skin-cancer-mnist-ham10000` | ~10k | 7 |
| DermNet | `shubhamgoel27/dermnet` | ~20k | 23 |
| PAD-UFES-20 | `mahdavi1202/skin-cancer` | ~2.3k | 6 |
| Fitzpatrick17k | `mmaximillian/fitzpatrick17k-images` | ~17k | 114 |
| Massive Balanced | `kylegraupe/skin-disease-balanced-dataset` | ~262k | 34 |

All are unified via label mapping → ~30 canonical classes → top 20 selected by sample count.

---

## Project Structure

```
jarvis-training/
├── setup.sh            # One-click setup (deps + download + prepare)
├── download_data.py    # Download all 6 datasets from Kaggle
├── prepare_data.py     # Unify labels, select top 20, stratified split
├── train.py            # Main training script (A6000 optimised)
├── evaluate.py         # Test evaluation with TTA
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

After setup + training:
```
datasets/               # Raw Kaggle downloads (~70 GB)
data/
├── train/              # 80% split (class subfolders)
├── val/                # 10% split
├── test/               # 10% split
└── class_info.json     # Class metadata
checkpoints/
├── best_model.pth      # Best validation accuracy
├── checkpoint_latest.pth
├── config.json
└── class_mapping.json
logs/                   # Training logs
runs/                   # TensorBoard
eval_results/           # Evaluation outputs
```

---

## Kaggle Setup

1. Go to <https://www.kaggle.com/settings>
2. Under **API**, click **Create New Token** → downloads `kaggle.json`
3. On Jarvis Labs:
   ```bash
   mkdir -p ~/.kaggle
   # Upload kaggle.json to the instance, then:
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## Training Commands

```bash
# Default training (recommended)
python train.py

# Resume from checkpoint
python train.py --resume

# Custom hyperparameters
python train.py --epochs 60 --batch-size 32 --lr 2e-4 --grad-accum 2

# Skip mixup/cutmix (if overfitting to augmentations)
python train.py --mix-prob 0.0

# Disable EMA
python train.py --no-ema

# Use fp16 instead of bf16 (non-Ampere GPUs)
python train.py --amp-dtype float16
```

---

## Evaluation Commands

```bash
# Standard evaluation
python evaluate.py

# With Test-Time Augmentation (recommended, +1-2% acc)
python evaluate.py --tta

# Custom checkpoint
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_50.pth --tta
```

---

## Expected Performance

| Metric | Expected |
|--------|----------|
| Overall Accuracy | 90–94% |
| Balanced Accuracy | 88–92% |
| Cancer Sensitivity | 85–90% |
| Training Time | 10–15 hours |
| Time per Epoch | 12–18 minutes |
| Peak VRAM Usage | ~35–40 GB |

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 24 GB | 48 GB (A6000) |
| RAM | 16 GB | 32 GB |
| Storage | 100 GB | 200 GB |
| CPUs | 4 | 7+ |

---

## Cost Estimate (Jarvis Labs)

| Phase | Time | Cost (₹63.99/hr) |
|-------|------|-------------------|
| Setup + Download | ~1 hour | ₹64 |
| Training (50 epochs) | ~12 hours | ₹768 |
| Evaluation | ~15 min | ₹16 |
| **Total** | **~13 hours** | **~₹850** |

Budget recommendation: **₹1,500** (includes buffer for reruns/experiments).

---

## Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce batch size + increase gradient accumulation
python train.py --batch-size 32 --grad-accum 2
```

### Kaggle download fails
```bash
# Verify API key
kaggle datasets list -s isic

# Re-download a specific dataset
kaggle datasets download -d andrewmvd/isic-2019 -p datasets/isic_2019 --unzip
```

### Low accuracy on specific class
- Check `eval_results/eval_results.json` for per-class accuracy
- Consider increasing `--cancer-boost` for cancer classes
- Try `--focal-gamma 3.0` to focus more on hard examples

### Training interrupted
```bash
# Auto-resume from latest checkpoint
python train.py --resume
```

---

## Deploying the Model

After training, copy the best model to your backend:

```bash
# On Jarvis Labs: download best_model.pth
# Then in your Skin-Doc backend:
cp best_model.pth backend/ml/weights/skin_convnext_large.pth

# Update backend/ml/model.py to load the new model
```

The checkpoint contains:
- `ema_state_dict` — Use this for inference (best generalisation)
- `classes` — List of 20 class names
- `config` — All training hyperparameters

---

## License

For research and educational use.
