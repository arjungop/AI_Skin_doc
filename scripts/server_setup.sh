#!/bin/bash
# A100/A6000 server setup script
# Downloads datasets and prepares training environment

set -e

echo "Skin-Doc GPU server setup (A100/A6000)"
echo "Downloads datasets and prepares training environment"
echo

# Configuration - EDIT THIS
KAGGLE_API_TOKEN="KGAT_6baf5aa6ae1a7d6fccc09c8ab1e2c649"
PROJECT_DIR="${HOME}/Skin-Doc"

echo "Step 1: Setting up project directory"
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

# Create necessary directories
mkdir -p datasets/{isic_2019,ham10000,dermnet,fitzpatrick,pad_ufes,skin20,massive}
mkdir -p data/{unified_train,unified_val}
mkdir -p checkpoints
mkdir -p scripts

echo "Created directory structure at ${PROJECT_DIR}"

# -----------------------------------------------------------------------------
# 2. Setup Python environment
# -----------------------------------------------------------------------------
echo
echo "---"

# CRITICAL: Always use separate conda env, NEVER base
if command -v conda &> /dev/null; then
    echo "Using conda (creating separate 'skindoc' environment)..."
    
    # Remove old environment if exists
    conda env remove -n skindoc -y 2>/dev/null || true
    
    # Create fresh environment
    conda create -n skindoc python=3.10 -y
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate skindoc
    
    # Verify we're NOT in base
    if [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
        echo "ERROR: Still in base environment! Exiting."
        exit 1
    fi
    
    echo "✅ Created and activated conda environment: $CONDA_DEFAULT_ENV"
else
    echo "Using venv..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install dependencies with retries
echo "Installing dependencies..."
pip install --upgrade pip

# PyTorch with CUDA 12.1 support for A100
echo "Installing PyTorch for A100 (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
    echo "Retrying PyTorch installation..."
    sleep 5
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')" || {
    echo "ERROR: PyTorch installation failed!"
    exit 1
}

# Install training requirements
echo "Installing training dependencies..."
pip install -r requirements_training.txt || {
    echo "requirements_training.txt not found, installing manually..."
    pip install numpy pillow pandas scipy scikit-learn \
                kaggle gdown requests tqdm matplotlib seaborn \
                tensorboard albumentations jupyter ipykernel \
                python-dotenv pyyaml
}

# Verify critical imports
echo "Verifying installations..."
python -c "
import torch
import torchvision
import numpy as np
from PIL import Image
import tqdm
import kaggle
print('✅ All critical packages imported successfully')
print(f'   PyTorch: {torch.__version__}')
print(f'   Torchvision: {torchvision.__version__}')
print(f'   NumPy: {np.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
" || {
    echo "ERROR: Some packages failed to import!"
    exit 1
}

echo "Python environment ready"

# -----------------------------------------------------------------------------
# 3. Setup Kaggle API
# -----------------------------------------------------------------------------
echo
echo "---"

# Create .kaggle directory
mkdir -p ~/.kaggle

# Check if kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo
    echo "⚠️  Kaggle credentials not found!"
    echo "Please download kaggle.json from https://www.kaggle.com/settings"
    echo "Then place it at: ~/.kaggle/kaggle.json"
    echo
    read -p "Press Enter once kaggle.json is in place, or Ctrl+C to exit: "
fi

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json 2>/dev/null || true

# Verify Kaggle API works
if kaggle datasets list --page-size 1 &>/dev/null; then
    echo "✅ Kaggle API configured successfully"
else
    echo "❌ ERROR: Kaggle API not working. Please check credentials."
    exit 1
fi

# -----------------------------------------------------------------------------
# 4. Download datasets
# -----------------------------------------------------------------------------
echo
echo "---"

cd "${PROJECT_DIR}/datasets"

# Function to download with retry
download_with_retry() {
    local dataset=$1
    local dir=$2
    local check_path=$3
    local max_attempts=3
    
    if [ -d "$check_path" ]; then
        echo "  [OK] Already exists, skipping..."
        return 0
    fi
    
    for attempt in $(seq 1 $max_attempts); do
        echo "  Attempt $attempt/$max_attempts..."
        if kaggle datasets download -d "$dataset" -p "$dir" --unzip; then
            echo "  [OK] Downloaded successfully"
            return 0
        else
            echo "  [ERROR] Failed, retrying..."
            sleep 5
        fi
    done
    
    echo "  [WARNING] Failed after $max_attempts attempts"
    return 1
}

# ISIC 2019 (~9GB, essential for cancer detection)
echo
echo "[1/6] Downloading ISIC 2019..."
download_with_retry "andrewmvd/isic-2019" "isic_2019" "isic_2019/ISIC_2019_Training_Input"

# HAM10000 (~3GB, high-quality dermoscopic images)
echo
echo "[2/6] Downloading HAM10000..."
download_with_retry "kmader/skin-cancer-mnist-ham10000" "ham10000" "ham10000/HAM10000_images_part_1"

# DermNet (~1GB, 23 disease classes)
echo
echo "[3/6] Downloading DermNet..."
download_with_retry "shubhamgoel27/dermnet" "dermnet" "dermnet/train"

# Fitzpatrick17k (~5GB, diverse skin tones)
echo
echo "[4/6] Downloading Fitzpatrick17k..."
if [ ! -d "fitzpatrick" ] || [ -z "$(ls -A fitzpatrick 2>/dev/null)" ]; then
    kaggle datasets download -d mmaximillian/fitzpatrick17k-images -p fitzpatrick --unzip
else
    echo "  Already exists, skipping..."
fi

# PAD-UFES-20 (~0.5GB, diverse skin tones from Brazil)
echo
echo "[5/6] Downloading PAD-UFES-20..."
if [ ! -d "pad_ufes" ] || [ -z "$(ls -A pad_ufes 2>/dev/null)" ]; then
    kaggle datasets download -d mahdavi1202/pad-ufes-20 -p pad_ufes --unzip
else
    echo "  Already exists, skipping..."
fi

# 20 Skin Diseases Dataset (~320MB, good for missing classes)
echo
echo "[6/7] Downloading 20 Skin Diseases Dataset..."
if [ ! -d "skin20" ] || [ -z "$(ls -A skin20 2>/dev/null)" ]; then
    kaggle datasets download -d haroonalam16/20-skin-diseases-dataset -p skin20 --unzip
else
    echo "  Already exists, skipping..."
fi

# Massive Skin Disease Dataset (50GB+, 262K images, 34 classes - BEST COVERAGE)
echo
echo "[7/7] Downloading Massive Skin Disease Dataset (50GB - this will take a while)..."
if [ ! -d "massive" ] || [ -z "$(ls -A massive 2>/dev/null)" ]; then
    kaggle datasets download -d kylegraupe/skin-disease-balanced-dataset -p massive --unzip
else
    echo "  Already exists, skipping..."
fi

# -----------------------------------------------------------------------------
# 5. Show dataset summary
# -----------------------------------------------------------------------------
echo
echo "---"
cd "${PROJECT_DIR}/datasets"

echo
echo "Downloaded datasets:"
for dir in */; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
        printf "  %-20s %'d images\n" "$dir" "$count"
    fi
done

total=$(find . -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
echo
echo "  TOTAL: $total images"

# -----------------------------------------------------------------------------
# 6. Print next steps
# -----------------------------------------------------------------------------
echo "     SETUP COMPLETE!                                          "
echo "Next steps:"
echo
echo "1. Copy training scripts to server:"
echo "   scp scripts/*.py user@server:${PROJECT_DIR}/scripts/"
echo
echo "2. Copy unified dataset preparation script and run:"
echo "   python scripts/prepare_unified_dataset.py"
echo
echo "3. Start training on A100:"
echo "   python scripts/train_a100.py \\"
echo "     --backbone efficientnet_b4 \\"
echo "     --batch_size 64 \\"
echo "     --image_size 384 \\"
echo "     --epochs 50"
echo
echo "For best results with A100 40GB:"
echo "   python scripts/train_a100.py \\"
echo "     --backbone swin_b \\"
echo "     --batch_size 48 \\"
echo "     --image_size 384 \\"
echo "     --epochs 100"
