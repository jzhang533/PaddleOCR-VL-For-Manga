#!/usr/bin/env python
"""
View and compare evaluation results from multiple checkpoints.

Usage:
    python view_results.py
    python view_results.py --results_dir results --metric cer
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="View evaluation results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cer",
        help="Metric to display (cer, wer, exact_match, etc.)",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="checkpoint",
        choices=["checkpoint", "metric"],
        help="Sort results by checkpoint number or metric value",
    )
    return parser.parse_args()


def extract_checkpoint_number(filename: str) -> int:
    """Extract checkpoint number from result filename."""
    try:
        # checkpoint-1000_test.json -> 1000
        parts = filename.split("_")[0].split("-")
        return int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return 0


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    # Load all result files
    results = []
    for result_file in results_dir.glob("*-*_*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
                checkpoint_num = extract_checkpoint_number(result_file.name)
                metrics = data.get("metrics", {})

                results.append(
                    {
                        "checkpoint": checkpoint_num,
                        "filename": result_file.name,
                        "split": data.get("split", "unknown"),
                        "metrics": metrics,
                    }
                )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {result_file.name}: {e}")
            continue

    if not results:
        print(f"No evaluation results found in {results_dir}")
        return 1

    # Sort results
    if args.sort_by == "checkpoint":
        results.sort(key=lambda x: x["checkpoint"])
    else:
        results.sort(key=lambda x: x["metrics"].get(args.metric, float("inf")))

    # Display results
    print("=" * 80)
    print(f"Evaluation Results Summary ({len(results)} checkpoint(s))")
    print("=" * 80)
    print()

    # Find all unique metrics
    all_metrics = set()
    for r in results:
        all_metrics.update(r["metrics"].keys())
    all_metrics = sorted(all_metrics)

    # Print header
    header = f"{'Filename':<30} {'Checkpoint':<15} {'Split':<8}"
    for metric in all_metrics:
        header += f" {metric.upper():<12}"
    print(header)
    print("-" * len(header))

    # Print each result
    for r in results:
        line = f"{r['filename']:<30} checkpoint-{r['checkpoint']:<4} {r['split']:<8}"
        for metric in all_metrics:
            value = r["metrics"].get(metric, float("nan"))
            if isinstance(value, float):
                line += f" {value:<12.4f}"
            else:
                line += f" {value:<12}"
        print(line)

    print()
    print("=" * 80)

    # Find best checkpoint
    if args.metric in all_metrics:
        best = min(results, key=lambda x: x["metrics"].get(args.metric, float("inf")))
        print(
            f"Best {args.metric.upper()}: checkpoint-{best['checkpoint']} "
            f"({best['metrics'][args.metric]:.4f})"
        )
        print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
