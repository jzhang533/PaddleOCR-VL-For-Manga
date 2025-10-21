# Quick Start Guide: Automatic Checkpoint Evaluation

This guide shows you how to set up automatic evaluation while training.

## Setup (One-time)

```bash
cd /path/to/paddleocr-vl-sft/evaluation
chmod +x watch_and_eval.sh watch_and_eval.py eval.sh
mkdir -p results
```

## Usage

### Step 1: Start Training (Terminal 1)

```bash
cd /path/to/paddleocr-vl-sft
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

### Step 2: Start Auto-Evaluation (Terminal 2)

```bash
cd /path/to/paddleocr-vl-sft/evaluation
CUDA_VISIBLE_DEVICES=1 bash watch_and_eval.sh
```

That's it! The watcher will:
- Detect new checkpoints automatically
- Wait for checkpoint to finish saving
- Run evaluation on a separate GPU
- Save results to `results/checkpoint-{step}_test.json`

### Step 3: Monitor Progress

```bash
# View evaluation results
cd /path/to/paddleocr-vl-sft/evaluation
python view_results.py

# Or check the log
tail -f evaluation.log

# Or check results directory
ls -lh results/
```

## Output Examples

**Console output from watcher:**
```
======================================================================
🔍 Checkpoint Watcher & Automatic Evaluator
======================================================================
Watch directory: /path/to/sft_output
Results directory: /path/to/evaluation/results
Evaluation split: test
Check interval: 60s
Stabilization time: 30s
======================================================================

👀 Watching for new checkpoints... (Ctrl+C to stop)

🆕 Found 1 new checkpoint(s) to evaluate

📊 Processing: checkpoint-1000
  Path: /path/to/sft_output/checkpoint-1000
  Waiting for checkpoint to stabilize (30s)...
  ✓ Checkpoint stable (size: 3.45 GB)
  🚀 Starting evaluation...
  ✅ Evaluation completed in 125.3s
  💾 Results saved to: results/checkpoint-1000_test.json
  📈 Metrics:
      cer: 0.0234
      wer: 0.1234
      exact_match: 0.8765
```

**Results summary:**
```bash
$ python view_results.py
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

## Customization

Edit `watch_and_eval.sh` to change defaults:

```bash
python watch_and_eval.py \
    --watch_dir ../sft_output \
    --eval_split test \
    --batch_size 32 \
    --max_samples 500 \          # Evaluate on 500 samples (faster)
    --check_interval 60 \        # Check every 60 seconds
    --stabilization_time 30      # Wait 30s for checkpoint to finish saving
```

## Tips

1. **Use different GPUs**: Train on GPU 0, evaluate on GPU 1
2. **Adjust sample size**: Use `--max_samples 100` for quick checks during development
3. **Check frequently**: Use `--check_interval 30` to catch checkpoints faster
4. **Monitor the log**: `tail -f evaluation.log` to see evaluation history
5. **Re-evaluate**: Delete entries from `evaluation.log` to re-evaluate checkpoints

## Stopping

Press `Ctrl+C` in the watcher terminal. It will show a summary before exiting.

## Files Generated

```
evaluation/
├── results/
│   ├── checkpoint-1000_test.json    # Detailed results for checkpoint 1000
│   ├── checkpoint-2000_test.json    # Detailed results for checkpoint 2000
│   └── checkpoint-3000_test.json    # Detailed results for checkpoint 3000
└── evaluation.log                   # JSON-lines log of all evaluations
```

Each result file contains:
- Metrics (CER, WER, exact match, etc.)
- All predictions with references
- Checkpoint path and configuration
- Timestamp and evaluation duration
