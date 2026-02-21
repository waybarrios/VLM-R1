#!/bin/bash
#
# Monitor inference progress for all checkpoints
#

OUTPUT_BASE="${1:-predictions_final}"

if [ ! -d "$OUTPUT_BASE" ]; then
    echo "Directory not found: $OUTPUT_BASE"
    echo "Usage: $0 [output_directory]"
    exit 1
fi

# Clear screen and show header
clear
echo "========================================"
echo "Inference Progress Monitor"
echo "========================================"
echo "Output directory: $OUTPUT_BASE"
echo "Press Ctrl+C to exit"
echo "========================================"
echo ""

while true; do
    # Move cursor to top (after header)
    tput cup 6 0

    # Count total checkpoints (only directories, not .log files)
    total_checkpoints=$(find "$OUTPUT_BASE" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | wc -l)

    if [ $total_checkpoints -eq 0 ]; then
        echo "No checkpoint directories found yet..."
        sleep 5
        continue
    fi

    # Table header
    printf "%-20s %10s %10s %10s %8s %12s\n" "Checkpoint" "Valid" "Total" "Progress" "Status" "ETA"
    printf "%-20s %10s %10s %10s %8s %12s\n" "----------" "-----" "-----" "--------" "------" "---"

    completed=0
    in_progress=0
    failed=0

    for checkpoint_dir in $(find "$OUTPUT_BASE" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V); do
        checkpoint_name=$(basename "$checkpoint_dir")
        stats_file="${checkpoint_dir}/inference_stats.json"

        if [ -f "$stats_file" ]; then
            # Completed - parse stats
            valid=$(python -c "import json; data=json.load(open('$stats_file')); print(data.get('valid_predictions', 0))" 2>/dev/null)
            total=$(python -c "import json; data=json.load(open('$stats_file')); print(data.get('processed_samples', 0))" 2>/dev/null)
            elapsed=$(python -c "import json; data=json.load(open('$stats_file')); print(int(data.get('elapsed_time_seconds', 0)/60))" 2>/dev/null)

            if [ -n "$valid" ] && [ -n "$total" ] && [ "$total" -gt 0 ]; then
                progress=$(echo "scale=1; $valid * 100 / $total" | bc 2>/dev/null)
                printf "%-20s %10s %10s %9s%% %8s %12s\n" "$checkpoint_name" "$valid" "$total" "$progress" "✓" "${elapsed}min"
                completed=$((completed + 1))
            else
                printf "%-20s %10s %10s %10s %8s %12s\n" "$checkpoint_name" "?" "?" "?" "?" "?"
                failed=$((failed + 1))
            fi
        else
            # In progress - count existing predictions
            num_predictions=$(ls "$checkpoint_dir"/*.json 2>/dev/null | grep -v "inference_summary.json" | grep -v "inference_stats.json" | wc -l)

            if [ $num_predictions -gt 0 ]; then
                # Try to estimate total from any completed checkpoint
                estimated_total="?"
                for other_dir in $(find "$OUTPUT_BASE" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null); do
                    other_stats="${other_dir}/inference_stats.json"
                    if [ -f "$other_stats" ]; then
                        estimated_total=$(python -c "import json; data=json.load(open('$other_stats')); print(data.get('processed_samples', '?'))" 2>/dev/null)
                        break
                    fi
                done

                # Calculate ETA based on log file age and progress
                log_file="${OUTPUT_BASE}/${checkpoint_name}.log"
                eta_str="..."
                if [ -f "$log_file" ] && [ "$estimated_total" != "?" ] && [ "$estimated_total" -gt 0 ]; then
                    # Get file age in seconds
                    current_time=$(date +%s)
                    log_start_time=$(stat -c %Y "$log_file" 2>/dev/null || stat -f %m "$log_file" 2>/dev/null)
                    if [ -n "$log_start_time" ]; then
                        elapsed_seconds=$((current_time - log_start_time))
                        if [ $num_predictions -gt 0 ] && [ $elapsed_seconds -gt 0 ]; then
                            # Calculate speed and ETA
                            speed=$(echo "scale=2; $num_predictions / $elapsed_seconds" | bc)
                            remaining=$((estimated_total - num_predictions))
                            eta_seconds=$(echo "scale=0; $remaining / $speed" | bc 2>/dev/null)
                            if [ -n "$eta_seconds" ] && [ "$eta_seconds" -gt 0 ]; then
                                eta_minutes=$((eta_seconds / 60))
                                if [ $eta_minutes -lt 60 ]; then
                                    eta_str="${eta_minutes}min"
                                else
                                    eta_hours=$((eta_minutes / 60))
                                    eta_str="${eta_hours}h"
                                fi
                            fi
                        fi
                    fi
                fi

                # Calculate progress if we have total
                if [ "$estimated_total" != "?" ] && [ "$estimated_total" -gt 0 ]; then
                    progress=$(echo "scale=1; $num_predictions * 100 / $estimated_total" | bc 2>/dev/null)
                    printf "%-20s %10s %10s %9s%% %8s %12s\n" "$checkpoint_name" "$num_predictions" "$estimated_total" "$progress" "⏳" "$eta_str"
                else
                    printf "%-20s %10s %10s %10s %8s %12s\n" "$checkpoint_name" "$num_predictions" "~2000" "..." "⏳" "..."
                fi
                in_progress=$((in_progress + 1))
            else
                printf "%-20s %10s %10s %10s %8s %12s\n" "$checkpoint_name" "0" "~2000" "0%" "⏸" "-"
            fi
        fi
    done

    # Summary
    echo ""
    echo "========================================"
    printf "Summary: %d completed | %d in progress | %d failed\n" $completed $in_progress $failed
    echo "========================================"
    echo ""
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"

    # Clear rest of screen
    tput ed

    # Update every 10 seconds
    sleep 10
done
