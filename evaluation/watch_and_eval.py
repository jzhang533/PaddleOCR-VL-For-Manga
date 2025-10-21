#!/usr/bin/env python
"""
Checkpoint Watcher and Automatic Evaluator

This script watches the sft_output folder for new checkpoints and automatically
evaluates them when they appear. It tracks which checkpoints have been evaluated
to avoid redundant work.

Usage:
    # Run with default settings
    CUDA_VISIBLE_DEVICES=1 python watch_and_eval.py

    # Customize watch folder and settings
    CUDA_VISIBLE_DEVICES=1 python watch_and_eval.py \
        --watch_dir ../sft_output \
        --eval_split test \
        --batch_size 16 \
        --max_samples 500 \
        --check_interval 60 \
        --stabilization_time 30

The script will:
1. Monitor the specified folder for checkpoint-* directories
2. Wait for checkpoint to stabilize (finished saving)
3. Automatically run evaluation
4. Save results to evaluation/results/checkpoint-{step}_eval.json
5. Keep a log of evaluated checkpoints to avoid re-evaluation
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch for new checkpoints and evaluate them automatically"
    )
    parser.add_argument(
        "--watch_dir",
        type=str,
        default="../sft_output",
        help="Directory to watch for checkpoints (relative to evaluation folder)",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--check_interval",
        type=int,
        default=60,
        help="Seconds between checking for new checkpoints",
    )
    parser.add_argument(
        "--stabilization_time",
        type=int,
        default=30,
        help="Seconds to wait after detecting new checkpoint (to ensure save is complete)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results (relative to evaluation folder)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="evaluation.log",
        help="Log file to track evaluation history",
    )
    return parser.parse_args()


def get_checkpoint_number(checkpoint_dir: Path) -> int:
    """Extract checkpoint number from checkpoint-XXXX directory name."""
    try:
        return int(checkpoint_dir.name.split("-")[1])
    except (IndexError, ValueError):
        return -1


def get_checkpoint_size(checkpoint_dir: Path) -> int:
    """Get total size of checkpoint directory in bytes."""
    total_size = 0
    for file_path in checkpoint_dir.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def is_checkpoint_stable(checkpoint_dir: Path, stabilization_time: int) -> bool:
    """
    Check if checkpoint has finished saving by monitoring directory size.

    Returns True if the directory size hasn't changed for stabilization_time seconds.
    """
    print(f"  Waiting for checkpoint to stabilize ({stabilization_time}s)...")

    initial_size = get_checkpoint_size(checkpoint_dir)
    time.sleep(stabilization_time)
    final_size = get_checkpoint_size(checkpoint_dir)

    is_stable = initial_size == final_size

    if is_stable:
        print(f"  ✓ Checkpoint stable (size: {final_size / (1024**3):.2f} GB)")
    else:
        print(
            f"  ✗ Checkpoint still changing (was {initial_size / (1024**3):.2f} GB, "
            f"now {final_size / (1024**3):.2f} GB)"
        )

    return is_stable


def load_evaluated_checkpoints(log_file: Path) -> set:
    """Load the set of already-evaluated checkpoint names."""
    if not log_file.exists():
        return set()

    evaluated = set()
    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                evaluated.add(entry["checkpoint_name"])
            except (json.JSONDecodeError, KeyError):
                continue

    return evaluated


def log_evaluation(log_file: Path, checkpoint_name: str, status: str, details: dict):
    """Append evaluation record to log file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_name": checkpoint_name,
        "status": status,
        **details,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def find_new_checkpoints(watch_dir: Path, evaluated: set) -> list:
    """Find checkpoint directories that haven't been evaluated yet."""
    if not watch_dir.exists():
        return []

    checkpoints = []
    for item in watch_dir.iterdir():
        if (
            item.is_dir()
            and item.name.startswith("checkpoint-")
            and item.name not in evaluated
        ):
            checkpoints.append(item)

    # Sort by checkpoint number
    checkpoints.sort(key=get_checkpoint_number)
    return checkpoints


def run_evaluation(
    checkpoint_path: Path,
    eval_split: str,
    batch_size: int,
    max_samples: int,
    max_new_tokens: int,
    output_file: Path,
) -> tuple[bool, str]:
    """
    Run evaluation on a checkpoint.

    Returns:
        (success: bool, message: str)
    """
    cmd = [
        sys.executable,  # Use same Python interpreter
        "eval_checkpoint.py",
        "--checkpoint_path",
        str(checkpoint_path.resolve()),
        "--split",
        eval_split,
        "--batch_size",
        str(batch_size),
        "--max_new_tokens",
        str(max_new_tokens),
        "--output_file",
        str(output_file),
    ]

    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])

    print(f"  Running: {' '.join(cmd)}")
    print()

    try:
        # Run without capturing output - let it print to console in real-time
        subprocess.run(
            cmd,
            check=True,
        )
        return True, "Success"
    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code {e.returncode}"
        return False, error_msg


def main():
    args = parse_args()

    # Set up paths (relative to evaluation folder)
    script_dir = Path(__file__).parent
    watch_dir = (script_dir / args.watch_dir).resolve()
    results_dir = script_dir / args.results_dir
    log_file = script_dir / args.log_file

    # Create results directory
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("🔍 Checkpoint Watcher & Automatic Evaluator")
    print("=" * 70)
    print(f"Watch directory: {watch_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Evaluation split: {args.eval_split}")
    print(f"Check interval: {args.check_interval}s")
    print(f"Stabilization time: {args.stabilization_time}s")
    print(f"Log file: {log_file}")
    print("=" * 70)
    print()

    if not watch_dir.exists():
        print(f"❌ Error: Watch directory does not exist: {watch_dir}")
        print("   Please start training first or adjust --watch_dir")
        return 1

    # Load previously evaluated checkpoints
    evaluated = load_evaluated_checkpoints(log_file)
    if evaluated:
        print(f"📋 Found {len(evaluated)} previously evaluated checkpoint(s)")
        print()

    print("👀 Watching for new checkpoints... (Ctrl+C to stop)")
    print()

    try:
        while True:
            # Find new checkpoints
            new_checkpoints = find_new_checkpoints(watch_dir, evaluated)

            if new_checkpoints:
                print(f"🆕 Found {len(new_checkpoints)} new checkpoint(s) to evaluate")
                print()

                for checkpoint_dir in new_checkpoints:
                    checkpoint_name = checkpoint_dir.name
                    checkpoint_num = get_checkpoint_number(checkpoint_dir)

                    print(f"📊 Processing: {checkpoint_name}")
                    print(f"  Path: {checkpoint_dir}")

                    # Wait for checkpoint to stabilize
                    if not is_checkpoint_stable(
                        checkpoint_dir, args.stabilization_time
                    ):
                        print(
                            "  ⏳ Checkpoint still being saved, will retry next cycle"
                        )
                        print()
                        continue

                    # Set up output file
                    output_file = (
                        results_dir / f"{checkpoint_name}_{args.eval_split}.json"
                    )

                    print("  🚀 Starting evaluation...")
                    start_time = time.time()

                    # Run evaluation
                    success, message = run_evaluation(
                        checkpoint_path=checkpoint_dir,
                        eval_split=args.eval_split,
                        batch_size=args.batch_size,
                        max_samples=args.max_samples,
                        max_new_tokens=args.max_new_tokens,
                        output_file=output_file,
                    )

                    elapsed_time = time.time() - start_time

                    if success:
                        print(f"  ✅ Evaluation completed in {elapsed_time:.1f}s")
                        print(f"  💾 Results saved to: {output_file}")

                        # Try to load and display key metrics
                        try:
                            with open(output_file) as f:
                                results = json.load(f)
                                metrics = results.get("metrics", {})
                                print("  📈 Metrics:")
                                for key, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        print(f"      {key}: {value:.4f}")
                        except Exception:
                            pass

                        log_evaluation(
                            log_file,
                            checkpoint_name,
                            "success",
                            {
                                "checkpoint_step": checkpoint_num,
                                "eval_split": args.eval_split,
                                "elapsed_time": elapsed_time,
                                "output_file": str(output_file),
                            },
                        )
                    else:
                        print(f"  ❌ Evaluation failed: {message}")
                        log_evaluation(
                            log_file,
                            checkpoint_name,
                            "failed",
                            {
                                "checkpoint_step": checkpoint_num,
                                "error": message,
                            },
                        )

                    # Mark as evaluated (even if failed, to avoid retrying continuously)
                    evaluated.add(checkpoint_name)
                    print()

            # Wait before next check
            time.sleep(args.check_interval)

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("🛑 Stopping checkpoint watcher")
        print(f"📊 Total checkpoints evaluated: {len(evaluated)}")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
