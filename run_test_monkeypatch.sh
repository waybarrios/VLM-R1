#!/bin/bash
# Test script for Qwen2.5-VL monkey patch validation

echo "=========================================="
echo "Qwen2.5-VL Monkey Patch Test"
echo "=========================================="
echo ""
echo "This test will:"
echo "1. Apply the monkey patch"
echo "2. Load Qwen2.5-VL-3B-Instruct"
echo "3. Test text-only generation"
echo "4. Test image + text generation"
echo "5. Test JSON format generation (VQA-style)"
echo ""
echo "GPU: Using CUDA device 0"
echo "=========================================="
echo ""

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Run the test
cd /gpudata3/Wayner/VLM-R1
python3 test_monkeypatch.py

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
