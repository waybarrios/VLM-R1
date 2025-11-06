#!/bin/bash
# Quick environment verification script
# Usage: bash verify_env.sh

echo "================================================================================"
echo "Environment Verification for VLM-R1 Training"
echo "================================================================================"
echo ""

# Activate environment
source /scratch/miniconda3/etc/profile.d/conda.sh
conda activate torch26

echo "📦 Checking critical packages..."
echo ""

# Check critical packages
CRITICAL_OK=true

check_exact() {
    PACKAGE=$1
    REQUIRED=$2
    INSTALLED=$(pip show $PACKAGE 2>/dev/null | grep "^Version:" | awk '{print $2}')

    if [ "$INSTALLED" == "$REQUIRED" ]; then
        echo "✅ $PACKAGE: $INSTALLED (required: $REQUIRED)"
    else
        echo "❌ $PACKAGE: $INSTALLED (required: $REQUIRED) - MISMATCH!"
        CRITICAL_OK=false
    fi
}

check_minimum() {
    PACKAGE=$1
    REQUIRED=$2
    INSTALLED=$(pip show $PACKAGE 2>/dev/null | grep "^Version:" | awk '{print $2}')

    if [ -n "$INSTALLED" ]; then
        echo "✅ $PACKAGE: $INSTALLED (required: $REQUIRED)"
    else
        echo "❌ $PACKAGE: NOT INSTALLED (required: $REQUIRED)"
        CRITICAL_OK=false
    fi
}

# Critical exact versions
echo "Critical packages (must match exactly):"
check_exact "transformers" "4.49.0"
check_exact "trl" "0.17.0"
check_exact "deepspeed" "0.15.4"
check_exact "liger_kernel" "0.5.2"

echo ""
echo "Required packages (minimum versions):"
check_minimum "torch" ">=2.5.1"
check_minimum "accelerate" ">=1.2.1"
check_minimum "datasets" ">=3.2.0"
check_minimum "wandb" ">=0.19.1"

echo ""
echo "================================================================================"

if [ "$CRITICAL_OK" = true ]; then
    echo "✅ Environment verified! Ready to train."
    echo ""
    echo "Start training with:"
    echo "  cd /gpudata3/Wayner/VLM-R1"
    echo "  bash train_vqa_multi_deepspeed.sh"
else
    echo "❌ Environment has issues! Please fix package versions."
    echo ""
    echo "Fix with:"
    echo "  conda activate torch26"
    echo "  pip install transformers==4.49.0 trl==0.17.0"
    echo "  cd /gpudata3/Wayner/VLM-R1/src/open-r1-multimodal"
    echo "  pip install -e ."
fi

echo "================================================================================"
