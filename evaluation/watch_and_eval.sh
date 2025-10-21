#!/bin/bash
#
# Watch for new checkpoints and automatically evaluate them
#
# Usage:
#   # Run with default settings (GPU 1)
#   CUDA_VISIBLE_DEVICES=1 bash watch_and_eval.sh
#
#   # Or customize GPU
#   CUDA_VISIBLE_DEVICES=0 bash watch_and_eval.sh
#

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "======================================"
echo "Checkpoint Watcher & Auto Evaluator"
echo "======================================"
echo ""
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "Starting watcher..."
echo ""

# Run the Python watcher script
# You can customize these arguments as needed
python watch_and_eval.py \
    --watch_dir ../../sft_output \
    --eval_split test \
    --batch_size 1 \
    --max_new_tokens 1024 \
#    --max_samples 500 \
    --check_interval 60 \
    --stabilization_time 30

echo ""
echo "======================================"
echo "Watcher stopped"
echo "======================================"
