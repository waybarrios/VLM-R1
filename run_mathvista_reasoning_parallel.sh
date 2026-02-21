#!/bin/bash

# Run MathVista Testmini Reasoning on GPUs 1, 3, 4 in parallel
# This will use all 3 GPUs simultaneously for faster evaluation

echo "=========================================="
echo "Running MathVista Testmini Reasoning"
echo "Parallel execution on GPUs 1, 3, 4"
echo "=========================================="
echo ""
echo "Starting 3 parallel jobs..."
echo ""

# Make scripts executable
chmod +x run_mathvista_reasoning_gpu1.sh
chmod +x run_mathvista_reasoning_gpu3.sh
chmod +x run_mathvista_reasoning_gpu4.sh

# Run all 3 GPUs in parallel
./run_mathvista_reasoning_gpu1.sh > logs-mathvista-gpu1.log 2>&1 &
PID1=$!
echo "✓ GPU 1 started (PID: $PID1)"

./run_mathvista_reasoning_gpu3.sh > logs-mathvista-gpu3.log 2>&1 &
PID3=$!
echo "✓ GPU 3 started (PID: $PID3)"

./run_mathvista_reasoning_gpu4.sh > logs-mathvista-gpu4.log 2>&1 &
PID4=$!
echo "✓ GPU 4 started (PID: $PID4)"

echo ""
echo "All 3 jobs running in parallel!"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs-mathvista-gpu1.log"
echo "  tail -f logs-mathvista-gpu3.log"
echo "  tail -f logs-mathvista-gpu4.log"
echo ""
echo "Check GPU usage with:"
echo "  nvidia-smi"
echo ""

# Wait for all jobs to complete
echo "Waiting for all jobs to complete..."
wait $PID1
STATUS1=$?
echo "✓ GPU 1 finished (exit code: $STATUS1)"

wait $PID3
STATUS3=$?
echo "✓ GPU 3 finished (exit code: $STATUS3)"

wait $PID4
STATUS4=$?
echo "✓ GPU 4 finished (exit code: $STATUS4)"

echo ""
echo "=========================================="
echo "All MathVista evaluations complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - logs-mathvista-testmini-reasoning-ckpt1500-gpu1/"
echo "  - logs-mathvista-testmini-reasoning-ckpt1500-gpu3/"
echo "  - logs-mathvista-testmini-reasoning-ckpt1500-gpu4/"
echo ""
echo "Logs saved to:"
echo "  - logs-mathvista-gpu1.log"
echo "  - logs-mathvista-gpu3.log"
echo "  - logs-mathvista-gpu4.log"
echo ""

if [ $STATUS1 -eq 0 ] && [ $STATUS3 -eq 0 ] && [ $STATUS4 -eq 0 ]; then
    echo "✅ All jobs completed successfully!"
else
    echo "⚠️  Some jobs failed. Check the logs above."
    echo "   GPU 1: exit code $STATUS1"
    echo "   GPU 3: exit code $STATUS3"
    echo "   GPU 4: exit code $STATUS4"
fi
echo ""
