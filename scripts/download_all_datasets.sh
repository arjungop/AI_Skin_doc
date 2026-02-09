#!/bin/bash
# ============================================================================
# Download ALL Missing Datasets for Skin Disease Classification
# Run with: screen -S downloads -dm bash scripts/download_all_datasets.sh
# ============================================================================

set -e
cd "$(dirname "$0")/.."

# Activate conda
source /dist_home/suryansh/miniforge3/bin/activate skindoc

# Create datasets directory
mkdir -p datasets
cd datasets

echo "=========================================="
echo "🚀 Starting Dataset Downloads"
echo "Time: $(date)"
echo "=========================================="

# ============================================================================
# 1. PAD-UFES-20 (Smartphone skin lesion images from Brazil)
# ============================================================================
echo ""
echo "📥 [1/2] Downloading PAD-UFES-20..."
echo "=========================================="

# Check if kaggle is configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️  Kaggle not configured! Run:"
    echo "   mkdir -p ~/.kaggle"
    echo "   echo '{\"username\":\"YOUR_USERNAME\",\"key\":\"YOUR_KEY\"}' > ~/.kaggle/kaggle.json"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
else
    pip install kaggle --quiet
    
    # Download PAD-UFES-20
    if [ ! -d "pad_ufes_20" ]; then
        kaggle datasets download -d mahdavi1202/skin-cancer -p pad_ufes_20 --unzip
        echo "✅ PAD-UFES-20 downloaded!"
    else
        echo "✅ PAD-UFES-20 already exists"
    fi
fi

# ============================================================================
# 2. ISIC 2019 (if not already downloaded)
# ============================================================================
echo ""
echo "📥 [2/2] Checking ISIC 2019..."
echo "=========================================="

if [ ! -d "../isic_data" ] || [ $(find ../isic_data -name "*.jpg" 2>/dev/null | wc -l) -lt 1000 ]; then
    echo "Downloading ISIC 2019..."
    kaggle datasets download -d andrewmvd/isic-2019 -p isic_2019 --unzip
    echo "✅ ISIC 2019 downloaded!"
else
    echo "✅ ISIC data already exists"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "✅ Download Complete!"
echo "Time: $(date)"
echo "=========================================="
echo ""
echo "Datasets downloaded to: $(pwd)"
ls -la

echo ""
echo "📋 NEXT STEPS:"
echo "   1. rsync Fitzpatrick17k from local machine"
echo "   2. Run: python scripts/prepare_unified_v3.py"
