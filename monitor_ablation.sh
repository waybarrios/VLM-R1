#!/bin/bash
# Monitor ablation experiments in real-time
# Shows progress of the 4 active experiments running on GPUs

LOG_DIR="ablation_results/logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "❌ No logs directory found. Are experiments running?"
    echo "   Expected: $LOG_DIR"
    exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "ABLATION EXPERIMENTS - LIVE MONITOR"
echo "════════════════════════════════════════════════════════════"
echo "Monitoring: $LOG_DIR"
echo "Press Ctrl+C to exit"
echo "════════════════════════════════════════════════════════════"
echo ""

while true; do
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "ABLATION EXPERIMENTS - LIVE MONITOR"
    echo "════════════════════════════════════════════════════════════"
    echo "Time: $(date '+%H:%M:%S')"
    echo ""

    # Get the 4 most recent log files (active experiments)
    RECENT_LOGS=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -4)

    if [ -z "$RECENT_LOGS" ]; then
        echo "⏳ Waiting for experiments to start..."
        sleep 2
        continue
    fi

    GPU_NUM=0
    for LOG_FILE in $RECENT_LOGS; do
        EXP_ID=$(basename "$LOG_FILE" .log | cut -d'_' -f2)
        GPU_ID=$(basename "$LOG_FILE" .log | cut -d'_' -f3 | sed 's/gpu//')

        echo "────────────────────────────────────────────────────────────"
        echo "GPU $GPU_ID | Experiment $EXP_ID"
        echo "────────────────────────────────────────────────────────────"

        # Get last progress line (tqdm output)
        PROGRESS_LINE=$(grep -E "Evaluating|Loading" "$LOG_FILE" 2>/dev/null | tail -1)

        if [ -z "$PROGRESS_LINE" ]; then
            # Check if completed
            if grep -q "✓ Results saved" "$LOG_FILE" 2>/dev/null; then
                RESULTS=$(tail -20 "$LOG_FILE" | grep -A 3 "SUMMARY TABLE" | tail -1)
                echo "✅ COMPLETED"
                if [ ! -z "$RESULTS" ]; then
                    echo "$RESULTS"
                fi
            else
                echo "⏳ Starting..."
            fi
        else
            # Show progress
            echo "$PROGRESS_LINE"
        fi

        echo ""
        GPU_NUM=$((GPU_NUM + 1))
    done

    # Count total experiments
    TOTAL_LOGS=$(ls "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    COMPLETED=$(grep -l "✓ Results saved" "$LOG_DIR"/*.log 2>/dev/null | wc -l)

    echo "════════════════════════════════════════════════════════════"
    echo "Overall Progress: $COMPLETED / $TOTAL_LOGS experiments completed"
    echo "════════════════════════════════════════════════════════════"

    # Update every 2 seconds
    sleep 2
done
