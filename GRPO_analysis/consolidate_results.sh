#!/bin/bash
# Consolidate GRPO analysis results into comparison tables

cd /gpudata3/Wayner/VLM-R1

echo "============================================================"
echo "GRPO ANALYSIS - RESULTS CONSOLIDATION"
echo "============================================================"
echo ""

python GRPO_analysis/consolidate_results.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Consolidation complete!"
    echo ""
    echo "View results:"
    echo "  cat GRPO_analysis/CONSOLIDATED_RESULTS.txt"
    echo "  cat GRPO_analysis/summary_table.csv"
else
    echo ""
    echo "❌ Consolidation failed!"
    exit 1
fi
