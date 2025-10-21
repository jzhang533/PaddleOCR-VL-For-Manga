# Evaluation Scripts

This directory contains scripts for evaluating PaddleOCR-VL checkpoints.

## Files

- **`eval_checkpoint.py`**: Core evaluation script that runs evaluation on a single checkpoint
- **`eval.sh`**: Shell wrapper for evaluating a single checkpoint
- **`watch_and_eval.py`**: Automatic checkpoint watcher that evaluates new checkpoints as they appear
- **`watch_and_eval.sh`**: Shell wrapper for the checkpoint watcher
- **`view_results.py`**: View and compare evaluation results from multiple checkpoints
- **`results/`**: Directory where evaluation results are saved (auto-created)
- **`evaluation.log`**: JSON-lines log file tracking all evaluation runs (auto-created)

## Usage

### Option 1: Automatic Evaluation (Recommended for Training)

Start the checkpoint watcher to automatically evaluate new checkpoints as training progresses:

```bash
# Run on GPU 1 (recommended to use different GPU from training)
cd evaluation
CUDA_VISIBLE_DEVICES=1 bash watch_and_eval.sh
```

The watcher will:
1. Monitor `../sft_output/` for new `checkpoint-*` directories
2. Wait for checkpoints to finish saving (stabilization period)
3. Automatically run evaluation on each new checkpoint
4. Save results to `results/checkpoint-{step}_test.json`
5. Log all evaluations to `evaluation.log`

**Customization:**

```bash
# Customize settings directly
CUDA_VISIBLE_DEVICES=1 python watch_and_eval.py \
    --watch_dir ../sft_output \
    --eval_split test \
    --batch_size 32 \
    --max_samples 500 \
    --check_interval 60 \
    --stabilization_time 30
```

**Parameters:**
- `--watch_dir`: Directory to watch for checkpoints (default: `../sft_output`)
- `--eval_split`: Dataset split to evaluate (`test` or `val`, default: `test`)
- `--batch_size`: Batch size for evaluation (default: `32`)
- `--max_samples`: Maximum samples to evaluate, `None` for all (default: `None`)
- `--max_new_tokens`: Max tokens to generate (default: `512`)
- `--check_interval`: Seconds between checking for new checkpoints (default: `60`)
- `--stabilization_time`: Seconds to wait after detecting checkpoint to ensure save is complete (default: `30`)
- `--results_dir`: Where to save results (default: `results`)
- `--log_file`: Evaluation log file (default: `evaluation.log`)

### Option 2: Manual Evaluation (Single Checkpoint)

Evaluate a specific checkpoint:

```bash
cd evaluation

# Evaluate a specific checkpoint
CUDA_VISIBLE_DEVICES=1 bash eval.sh /path/to/checkpoint-1000 test

# Or use the Python script directly
CUDA_VISIBLE_DEVICES=1 python eval_checkpoint.py \
    --checkpoint_path ../sft_output/checkpoint-1000 \
    --split test \
    --batch_size 32 \
    --max_samples 500
```

## Output Files

### Evaluation Results (`results/checkpoint-{step}_test.json`)

Each evaluation creates a JSON file with:
```json
{
    "checkpoint_path": "/path/to/checkpoint-1000",
    "split": "test",
    "num_samples": 500,
    "metrics": {
        "cer": 0.0234,
        "wer": 0.1234,
        "exact_match": 0.8765
    },
    "predictions": [
        {
            "index": 0,
            "reference": "ground truth text",
            "prediction": "predicted text",
            "cer": 0.0
        },
        ...
    ]
}
```

### Evaluation Log (`evaluation.log`)

JSON-lines format with one entry per evaluation:
```json
{"timestamp": "2025-10-31T12:34:56", "checkpoint_name": "checkpoint-1000", "status": "success", "checkpoint_step": 1000, "eval_split": "test", "elapsed_time": 123.45, "output_file": "results/checkpoint-1000_test.json"}
{"timestamp": "2025-10-31T13:45:67", "checkpoint_name": "checkpoint-2000", "status": "success", "checkpoint_step": 2000, "eval_split": "test", "elapsed_time": 125.67, "output_file": "results/checkpoint-2000_test.json"}
```

## Workflow Example

**Terminal 1 - Training:**
```bash
cd /path/to/paddleocr-vl-sft
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

**Terminal 2 - Auto Evaluation:**
```bash
cd /path/to/paddleocr-vl-sft/evaluation
CUDA_VISIBLE_DEVICES=1 bash watch_and_eval.sh
```

The watcher will automatically evaluate each checkpoint as it's saved during training, and you can monitor progress by checking the `results/` directory or `evaluation.log`.

## Stopping the Watcher

Press `Ctrl+C` to gracefully stop the checkpoint watcher. It will display a summary of evaluated checkpoints before exiting.

## Viewing Results

After evaluations complete, view and compare results:

```bash
cd evaluation

# View all results sorted by checkpoint
python view_results.py

# View sorted by CER (Character Error Rate)
python view_results.py --metric cer --sort_by metric

# View from custom results directory
python view_results.py --results_dir results --metric cer
```

This will display a table like:
```
================================================================================
Evaluation Results Summary (3 checkpoint(s))
================================================================================

Checkpoint      Split    Samples     CER          WER          EXACT_MATCH 
--------------------------------------------------------------------------------
checkpoint-1000 test     500         0.0234       0.1234       0.8765      
checkpoint-2000 test     500         0.0198       0.1156       0.8901      
checkpoint-3000 test     500         0.0187       0.1089       0.9012      

================================================================================
Best CER: checkpoint-3000 (0.0187)
================================================================================
```

## Troubleshooting

**"Watch directory does not exist"**
- Make sure training has started and created the `sft_output` directory
- Or adjust `--watch_dir` to point to the correct location

**"Checkpoint still changing"**
- The watcher detected a checkpoint but it's still being saved
- It will retry on the next check cycle
- You can increase `--stabilization_time` if your checkpoints take longer to save

**Evaluations skipped**
- Check `evaluation.log` to see which checkpoints have been evaluated
- Delete entries from the log file if you want to re-evaluate specific checkpoints
