"""
Evaluation script for trained PaddleOCR-VL checkpoints.

This script evaluates a trained checkpoint on a separate GPU by:
1. Loading the test/validation dataset
2. Generating predictions (completion only)
3. Computing CER (Character Error Rate) and accuracy metrics

Usage:
    cd evaluation
    python eval_checkpoint.py \
        --checkpoint_path ../../sft_output \
        --split test \
        --batch_size 8 \
        --max_samples 1000

You can specify which GPU to use with CUDA_VISIBLE_DEVICES:
    CUDA_VISIBLE_DEVICES=1 python eval_checkpoint.py --checkpoint_path ../../sft_output
"""

import argparse
import sys
from pathlib import Path


# Add parent directory to path to import ocr_dataset
sys.path.insert(0, str(Path(__file__).parent.parent))

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from ocr_dataset import MangaDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
original_model_path = "/home/PaddleOCR-VL"


def collate_fn(batch):
    """
    Collate function for inference batching.

    Follows custom_collator.py pattern for image handling.
    Dataset returns: {"images": [PIL.Image], "messages": [...]}
    Processor expects: images=[[img1], [img2], ...] (list-of-lists)
    """
    # Collect images as list-of-lists (each sample's images)
    # IMPORTANT: Keep as [[img1], [img2], ...] - processor expects this!
    images = [sample["images"] for sample in batch]

    # Prepare messages for inference (user messages only)
    # For chat template, we need: [[user_msg1], [user_msg2], ...]
    user_messages = [[sample["messages"][0]] for sample in batch]

    # Keep original messages for extracting references
    full_messages = [sample["messages"] for sample in batch]

    return {
        "images": images,
        "user_messages": user_messages,
        "full_messages": full_messages,
    }


def generate_predictions(model, processor, dataloader, max_new_tokens=1024):
    """
    Generate predictions for the entire dataset using batch processing.

    Returns only the completion part (generated text), not the prompt.

    Args:
        model: The trained model
        processor: The processor for tokenization
        dataloader: DataLoader with test samples
        max_new_tokens: Maximum tokens to generate

    Returns:
        predictions: List of predicted texts (completions only)
        references: List of ground truth texts
    """
    model.eval()
    predictions = []
    references = []

    print("\nGenerating predictions in batches...")
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Extract references from assistant messages
            for msgs in batch["full_messages"]:
                # Extract ground truth from assistant message (second message)
                # The content is a list with one dict: [{"type": "text", "text": "..."}]
                assistant_content = msgs[1]["content"]
                if isinstance(assistant_content, list):
                    # New format: [{"type": "text", "text": "..."}]
                    reference_text = assistant_content[0]["text"]
                else:
                    # Fallback for old format (just in case)
                    reference_text = assistant_content
                references.append(reference_text)

            # Apply chat template to all user messages in batch
            texts = processor.apply_chat_template(
                batch["user_messages"], tokenize=False, add_generation_prompt=True
            )

            # Process images and text together
            # Images: [[img1], [img2], ...] - list of single-image lists
            # Use RIGHT padding (same as training)
            inputs = processor(
                text=texts,
                images=batch["images"],
                return_tensors="pt",
                padding=True,
            )

            inputs = {
                k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }

            # Generate for the batch
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

            # Decode only the generated tokens (completion only, no prompt)
            # With padding, model.generate() outputs: [padded_input | new_tokens]
            # Slice at padded length (same for all samples in batch)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = generated[:, input_length:]

            # Batch decode all predictions
            pred_texts = processor.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            predictions.extend(pred_texts)

    return predictions, references


def compute_metrics(predictions, references):
    """
    Compute CER and accuracy metrics.

    Args:
        predictions: List of predicted texts (completions only)
        references: List of ground truth texts

    Returns:
        dict: Dictionary with 'cer' and 'accuracy' metrics
    """
    cer_metric = evaluate.load("cer")

    # Remove whitespace for fair comparison
    predictions_clean = ["".join(text.split()) for text in predictions]
    references_clean = ["".join(text.split()) for text in references]

    # Compute CER
    cer = cer_metric.compute(predictions=predictions_clean, references=references_clean)

    # Compute exact match accuracy
    predictions_array = np.array(predictions_clean)
    references_array = np.array(references_clean)
    accuracy = (predictions_array == references_array).mean()

    return {"cer": cer, "accuracy": accuracy, "num_samples": len(predictions)}


def print_sample_predictions(predictions, references, num_samples=5):
    """Print sample predictions for qualitative analysis."""
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)

    for i in range(min(num_samples, len(predictions))):
        print(f"\nSample {i + 1}:")
        print(
            f"  Reference: {references[i][:100]}{'...' if len(references[i]) > 100 else ''}"
        )
        print(
            f"  Predicted: {predictions[i][:100]}{'...' if len(predictions[i]) > 100 else ''}"
        )
        match = (
            "✓"
            if predictions[i].replace(" ", "") == references[i].replace(" ", "")
            else "✗"
        )
        print(f"  Match: {match}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained PaddleOCR-VL checkpoint"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained checkpoint directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1). "
        "Note: batch_size > 1 may produce incorrect results due to "
        "padding issues with vision-language models",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: None, all samples)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per sample (default: 1024)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save predictions to JSON file (default: None, no output file)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CHECKPOINT EVALUATION")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint_path}")
    print(f"Split: {args.split}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")

    # Load model and processor
    print(f"\nLoading model from {args.checkpoint_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",
    ).eval()

    processor = AutoProcessor.from_pretrained(
        original_model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Set pad_token_id in generation config to avoid warning
    # Using eos_token as pad_token is standard for decoder-only models
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    print("✓ Model loaded successfully")

    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = MangaDataset(
        split=args.split,
        limit_size=args.max_samples,
        augment=False,  # No augmentation for evaluation
        use_synthetic=False,  # Only use MANGA109 data for evaluation
    )

    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Single worker for sequential generation
    )

    # Generate predictions
    predictions, references = generate_predictions(
        model, processor, dataloader, args.max_new_tokens
    )

    # Compute metrics
    print("\n" + "=" * 80)
    print("COMPUTING METRICS")
    print("=" * 80)

    metrics = compute_metrics(predictions, references)

    print(f"\nResults on {args.split} set:")
    print(f"  Samples: {metrics['num_samples']}")
    print(f"  CER (Character Error Rate): {metrics['cer']:.4f}")
    print(
        f"  Accuracy (Exact Match): {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)"
    )

    # Print sample predictions
    print_sample_predictions(predictions, references, num_samples=5)

    # Save predictions if requested
    if args.output_file:
        import json

        output = {
            "checkpoint": args.checkpoint_path,
            "split": args.split,
            "metrics": metrics,
            "predictions": [
                {
                    "reference": ref,
                    "prediction": pred,
                    "match": pred.replace(" ", "") == ref.replace(" ", ""),
                }
                for pred, ref in zip(predictions, references)
            ],
        }

        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Predictions saved to: {args.output_file}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    main()
