#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ISIC 2019 Fast Fine-tuning Runner
# Run on Jarvis Labs RTX A6000 (48 GB) — estimated total: ~40-45 min
#
#   Download ISIC 2019 images (~9 GB, public S3 — NO auth needed):  ~15 min
#   Extract:                                                          ~5 min
#   Training 8 epochs @ 224px, batch 128, AMP fp16:                 ~20 min
#   ──────────────────────────────────────────────────────────────────────────
#   Total:                                                           ~40 min
#
# Usage (on Jarvis, inside AI_Skin_doc/jarvis-training/):
#   bash run_isic2019.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

echo "======================================="
echo "  Skin-Doc ISIC 2019 Fast Fine-tuning"
echo "  Target: ~40 min on RTX A6000"
echo "======================================="
echo ""

START=$(date +%s)

# ── 1. Dependencies ──────────────────────────────────────────────────────────
echo ">>> Installing dependencies..."
pip install -q \
    timm>=1.0.0 \
    torch>=2.2.0 \
    torchvision>=0.17.0 \
    huggingface-hub>=0.22.0 \
    tqdm>=4.66.0 \
    Pillow>=10.0.0 \
    numpy>=1.24.0

# ── 2. Resolve checkpoint ────────────────────────────────────────────────────
CHECKPOINT=""
for candidate in \
    "../backend/ml/weights/best_model.pth" \
    "../backend/ml/weights/best_model_original.pth" \
    "checkpoints/best_model.pth"
do
    if [ -f "$candidate" ]; then
        CHECKPOINT="$candidate"
        echo ">>> Found checkpoint: $CHECKPOINT"
        break
    fi
done

if [ -z "$CHECKPOINT" ]; then
    echo ">>> Downloading checkpoint from HuggingFace (arjg/skin-doc-model)..."
    mkdir -p checkpoints
    python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('arjg/skin-doc-model', 'best_model.pth', local_dir='checkpoints')
print('Downloaded to:', p)
"
    CHECKPOINT="checkpoints/best_model.pth"
fi

# ── 3. GPU sanity check ──────────────────────────────────────────────────────
echo ">>> GPU info:"
python3 -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: No CUDA GPU found!')
    import sys; sys.exit(1)
g = torch.cuda.get_device_properties(0)
print(f'  {torch.cuda.get_device_name(0)}  ({g.total_memory/1e9:.0f} GB)')
"

# ── 4. Run fine-tuning ───────────────────────────────────────────────────────
echo ""
echo ">>> Starting ISIC 2019 fine-tuning..."
echo "    img_size=224 | batch=128 | epochs=8 | AMP fp16"
echo "    Freeze: stages 0-2 (body)  |  Unfreeze: stage 3 + head"
echo ""

python3 finetune_isic2019.py \
    --checkpoint  "$CHECKPOINT" \
    --epochs      8            \
    --batch-size  128          \
    --img-size    224          \
    --lr          3e-4         \
    --data-dir    finetune_data/isic2019 \
    --output-dir  checkpoints  \
    --output-name finetuned_isic2019.pth

# ── 5. Copy result to backend ─────────────────────────────────────────────────
RESULT="checkpoints/finetuned_isic2019.pth"
if [ -f "$RESULT" ]; then
    echo ""
    echo ">>> Copying to ../backend/ml/weights/best_model.pth"
    cp "$RESULT" "../backend/ml/weights/best_model.pth"
    echo "    Done!"
else
    echo "ERROR: Output checkpoint not found at $RESULT"
    exit 1
fi

END=$(date +%s)
ELAPSED=$(( END - START ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "======================================="
echo "  COMPLETE in ${MINS}m ${SECS}s"
echo ""
echo "  Model saved to:"
echo "    checkpoints/finetuned_isic2019.pth"
echo "    ../backend/ml/weights/best_model.pth"
echo ""
echo "  To copy to your Mac:"
echo "    scp jarvis:~/AI_Skin_doc/jarvis-training/checkpoints/finetuned_isic2019.pth \\"
echo "        ~/Skin-Doc/backend/ml/weights/best_model.pth"
echo "======================================="
