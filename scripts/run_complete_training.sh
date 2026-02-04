#!/bin/bash
# ---
# COMPLETE A100 TRAINING PIPELINE
# Runs all steps from data preparation to training in one command
# ---

set -e  # Exit on any error

echo
echo "     COMPLETE A100 TRAINING PIPELINE                          "
echo "     Prepares data and trains model automatically             "
echo

# Configuration - Set these variables
PROJECT_DIR="${HOME}/Skin-Doc"
BACKBONE="${1:-efficientnet_b4}"  # Default to efficientnet_b4
BATCH_SIZE="${2:-64}"              # Default to 64
EPOCHS="${3:-50}"                  # Default to 50

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# 1. Activate environment
# -----------------------------------------------------------------------------
echo
echo -e "${GREEN}---${NC}"

if command -v conda &> /dev/null; then
    # Source conda
    source "$(conda info --base)/etc/profile.d/conda.sh"
    
    # Activate skindoc environment
    conda activate skindoc
    
    # Verify NOT in base
    if [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
        echo -e "${RED}ERROR: In base environment! Use 'conda activate skindoc'${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Using conda environment: $CONDA_DEFAULT_ENV${NC}"
else
    # Use venv
    if [ -f "${PROJECT_DIR}/venv/bin/activate" ]; then
        source "${PROJECT_DIR}/venv/bin/activate"
        echo -e "${GREEN}✅ Using venv environment${NC}"
    else
        echo -e "${RED}ERROR: No environment found!${NC}"
        exit 1
    fi
fi

cd "${PROJECT_DIR}"

# -----------------------------------------------------------------------------
# 2. Verify datasets are downloaded
# -----------------------------------------------------------------------------
echo
echo -e "${GREEN}---${NC}"

DATASETS_DIR="${PROJECT_DIR}/datasets"
required_datasets=("isic_2019" "ham10000" "dermnet")

for dataset in "${required_datasets[@]}"; do
    if [ ! -d "${DATASETS_DIR}/${dataset}" ]; then
        echo -e "${RED}ERROR: Required dataset '${dataset}' not found!${NC}"
        echo "Please run: bash scripts/server_setup.sh"
        exit 1
    fi
done

echo -e "${GREEN}✅ All required datasets present${NC}"

# Count total images
total_images=$(find "${DATASETS_DIR}" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
echo "   Total raw images: ${total_images}"

# -----------------------------------------------------------------------------
# 3. Prepare unified dataset
# -----------------------------------------------------------------------------
echo
echo -e "${GREEN}---${NC}"

if [ -d "${PROJECT_DIR}/data/unified_train" ] && [ "$(ls -A ${PROJECT_DIR}/data/unified_train 2>/dev/null | wc -l)" -gt 0 ]; then
    echo -e "${YELLOW}Unified dataset already exists.${NC}"
    read -p "Recreate it? (y/N): " recreate
    if [[ "$recreate" =~ ^[Yy]$ ]]; then
        echo "Removing old unified dataset..."
        rm -rf "${PROJECT_DIR}/data/unified_train" "${PROJECT_DIR}/data/unified_val"
    else
        echo "Skipping data preparation..."
    fi
fi

if [ ! -d "${PROJECT_DIR}/data/unified_train" ] || [ "$(ls -A ${PROJECT_DIR}/data/unified_train 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "Running dataset preparation script..."
    python scripts/prepare_unified_dataset_v2.py
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Dataset preparation failed!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Dataset preparation complete${NC}"
fi

# Show dataset stats
train_count=$(find data/unified_train -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
val_count=$(find data/unified_val -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
echo "   Training images: ${train_count}"
echo "   Validation images: ${val_count}"

# -----------------------------------------------------------------------------
# 4. Create checkpoint directory
# -----------------------------------------------------------------------------
echo
echo -e "${GREEN}---${NC}"

CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/$(date +%Y%m%d_%H%M%S)_${BACKBONE}"
mkdir -p "${CHECKPOINT_DIR}"
echo -e "${GREEN}✅ Checkpoint directory: ${CHECKPOINT_DIR}${NC}"

# -----------------------------------------------------------------------------
# 5. Start training
# -----------------------------------------------------------------------------
echo "     STARTING TRAINING                                        "
echo "Configuration:"
echo "  Backbone: ${BACKBONE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo
echo "Press Ctrl+C to stop training gracefully"
echo

# Log file
LOG_FILE="${CHECKPOINT_DIR}/training.log"

# Run training with output to both console and log
python scripts/train_a100.py \
    --backbone "${BACKBONE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --use_weighted_sampling \
    2>&1 | tee "${LOG_FILE}"

# Check if training succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo
    echo
    echo "     TRAINING COMPLETE!                                       "
    echo
    echo
    echo -e "${GREEN}✅ Training completed successfully${NC}"
    echo "   Best model saved in: ${CHECKPOINT_DIR}"
    echo "   Training log: ${LOG_FILE}"
    echo
    
    # Show best checkpoint
    if [ -f "${CHECKPOINT_DIR}/best_model.pth" ]; then
        echo "   Best model: $(ls -lh ${CHECKPOINT_DIR}/best_model.pth | awk '{print $5}')"
    fi
else
    echo
    echo -e "${RED}❌ Training failed! Check logs: ${LOG_FILE}${NC}"
    exit 1
fi
