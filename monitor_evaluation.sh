#!/bin/bash
# Monitor evaluation progress

LOG_FILE="/gpudata3/Wayner/VLM-R1/final_table/evaluation_log.txt"
OUTPUT_DIR="/gpudata3/Wayner/VLM-R1/final_table"

echo "==================================================================="
echo "CRYSTAL EVALUATION - PROGRESS MONITOR"
echo "==================================================================="
echo ""

# Check if log exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Evaluation not started yet. Log file not found: $LOG_FILE"
    exit 1
fi

# Count total models
TOTAL_MODELS=14

# Extract current model being processed
CURRENT_MODEL=$(grep "Evaluating:" "$LOG_FILE" | tail -1 | awk -F': ' '{print $2}')
CURRENT_NUM=$(grep "Evaluating:" "$LOG_FILE" | wc -l)

echo "📊 Overall Progress: $CURRENT_NUM / $TOTAL_MODELS models"
echo "🔄 Currently processing: $CURRENT_MODEL"
echo ""

# Show success/failure counts
SUCCESS_COUNT=$(grep -c "✓ Success" "$LOG_FILE" 2>/dev/null || echo "0")
FAILED_COUNT=$(grep -c "❌ Error" "$LOG_FILE" 2>/dev/null || echo "0")

echo "✅ Completed successfully: $SUCCESS_COUNT"
echo "❌ Failed: $FAILED_COUNT"
echo ""

# Show completed models
echo "───────────────────────────────────────────────────────────────────"
echo "Completed Models:"
echo "───────────────────────────────────────────────────────────────────"
grep "✓ Success" "$LOG_FILE" | sed 's/.*Results saved for /  ✓ /' || echo "  (none yet)"
echo ""

# Check if still running
if ps aux | grep "[r]un_final_evaluation.sh" > /dev/null; then
    echo "🟢 Status: RUNNING"
    echo ""

    # Show last few lines of progress
    echo "───────────────────────────────────────────────────────────────────"
    echo "Recent Activity (last 15 lines):"
    echo "───────────────────────────────────────────────────────────────────"
    tail -15 "$LOG_FILE" | grep -v "Converting dataset" | grep -v "Traceback" | grep -v "ModuleNotFoundError"
else
    echo "🔴 Status: NOT RUNNING"
    echo ""

    # Check if completed
    if grep -q "FINAL EVALUATION COMPLETE" "$LOG_FILE"; then
        echo "✅ EVALUATION COMPLETED!"
        echo ""

        # Show summary
        grep -A 10 "FINAL EVALUATION COMPLETE" "$LOG_FILE"
    else
        echo "⚠️  Evaluation stopped prematurely. Check log for errors."
    fi
fi

echo ""
echo "==================================================================="
echo "For detailed log: tail -f $LOG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "==================================================================="
