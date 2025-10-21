#!/bin/bash
#
# Evaluation script for trained checkpoints
#
# Run from the evaluation folder:
#   cd evaluation
#   CUDA_VISIBLE_DEVICES=1 bash eval.sh
#
# Or from the parent directory:
#   CUDA_VISIBLE_DEVICES=1 bash evaluation/eval.sh
#

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

CHECKPOINT_PATH="${1:-../../sft_output/checkpoint-5000}"
#CHECKPOINT_PATH="${1:-/home/PaddleOCR-VL}"
SPLIT="${2:-test}"

echo "======================================"
echo "Evaluating Checkpoint"
echo "======================================"
echo ""
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Split: $SPLIT"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo ""

python eval_checkpoint.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --split "$SPLIT" \
    --batch_size 1 \
    --max_samples 100 \
    --max_new_tokens 512 \
    --output_file "eval_results_${SPLIT}.json"

echo ""
echo "======================================"
echo "Evaluation complete!"
echo "Results saved to: eval_results_${SPLIT}.json"
echo "======================================"
