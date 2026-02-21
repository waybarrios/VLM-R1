#!/bin/bash
#
# Kill all running inference processes
#

PID_FILE="predictions_final/running_pids.txt"

if [ -f "$PID_FILE" ]; then
    echo "Killing processes from PID file..."
    cat "$PID_FILE"
    echo ""

    # Extract PIDs and kill them
    pids=$(cat "$PID_FILE" | grep -v '^#' | awk '{print $1}')

    if [ -n "$pids" ]; then
        echo "Killing PIDs: $pids"
        kill -9 $pids 2>/dev/null
        sleep 2
        echo "Done!"
    else
        echo "No PIDs found in file"
    fi
else
    echo "PID file not found: $PID_FILE"
    echo ""
    echo "Searching for inference processes manually..."
    pids=$(ps aux | grep "run_fast_inference.py" | grep -v grep | awk '{print $2}')

    if [ -n "$pids" ]; then
        echo "Found PIDs: $pids"
        echo -n "Kill these processes? [y/N] "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            kill -9 $pids 2>/dev/null
            echo "Killed!"
        else
            echo "Cancelled"
        fi
    else
        echo "No inference processes found"
    fi
fi

echo ""
echo "Remaining inference processes:"
ps aux | grep "run_fast_inference.py" | grep -v grep || echo "None"
