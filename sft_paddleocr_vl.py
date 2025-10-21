"""
Supervised Fine-Tuning (SFT) script for PaddleOCR-VL-0.9B using TRL library.

This version uses MangaDataset for training with both synthetic and
Manga109 data.

Available tasks:
- 'ocr' -> 'OCR:'
- 'table' -> 'Table Recognition:'
- 'chart' -> 'Chart Recognition:'
- 'formula' -> 'Formula Recognition:'
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from custom_collator import CustomDataCollatorForVisionLanguageModeling
from metrics import OCRMetrics
from ocr_dataset import MangaDataset


# ==================== Configuration ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== Data Classes ====================
@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_path: str = field(
        default="../PaddleOCR-VL", metadata={"help": "Path to the PaddleOCR-VL model"}
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={
            "help": (
                "Enable Flash Attention 2 for faster training. "
                "Requires flash-attn package and A100/H100 GPU. "
                "Provides 2-4x speedup."
            )
        },
    )


@dataclass
class DataArguments:
    """Arguments for dataset configuration."""

    split: str = field(
        default="train", metadata={"help": "Dataset split to use: 'train' or 'val'"}
    )
    eval_split: str = field(
        default="test",
        metadata={"help": "Dataset split to use for evaluation: 'test' or 'val'"},
    )
    max_length: int = field(
        default=2048,
        metadata={
            "help": (
                "Maximum sequence length (includes image + text tokens). "
                "For PaddleOCR-VL: images can be 400-2000+ tokens depending "
                "on size. Set high enough to avoid truncating image tokens, "
                "or use None to disable truncation."
            )
        },
    )
    eval_limit_size: Optional[int] = field(
        default=1000,
        metadata={
            "help": (
                "Limit eval dataset size to reduce memory usage. "
                "Default 1000 samples. Use None for full eval set."
            )
        },
    )
    skip_packages: Optional[str] = field(
        default=None,
        metadata={
            "help": ("Comma-separated list of package IDs to skip (e.g., '0,1,2')")
        },
    )
    pad_to_multiple_of: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "Pad sequence length to multiple of this value for GPU "
                "optimization. Use 8 for A100/H100 with mixed precision "
                "(FP16/BF16), 64 for older GPUs, or None to disable."
            )
        },
    )


def train():
    """Main training function."""

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set remove_unused_columns to False to avoid errors
    training_args.remove_unused_columns = False

    # Load model and processor
    print(f"Loading model from {model_args.model_path}...")

    # Prepare model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": DEVICE,
    }

    # Add Flash Attention 2 if enabled
    if model_args.use_flash_attention_2:
        print("🚀 Flash Attention 2 enabled (requires flash-attn package)")
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_args.model_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Parse skip_packages if provided
    skip_packages = None
    if data_args.skip_packages:
        skip_packages = [int(x.strip()) for x in data_args.skip_packages.split(",")]

    # Load dataset using MangaDataset
    print(f"\nLoading {data_args.split} dataset...")
    train_dataset = MangaDataset(
        split=data_args.split,
        skip_packages=skip_packages,
    )

    print(f"Training dataset size: {len(train_dataset)}")

    # Load evaluation dataset
    print(f"\nLoading {data_args.eval_split} dataset for evaluation...")
    eval_dataset = MangaDataset(
        split=data_args.eval_split,
        limit_size=data_args.eval_limit_size,
        augment=False,  # No augmentation for eval
        skip_packages=skip_packages,
    )

    print(f"Evaluation dataset size: {len(eval_dataset)}")

    my_collator = CustomDataCollatorForVisionLanguageModeling(
        processor,
        max_length=data_args.max_length,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
    )

    print("Initialize metrics")
    metrics = OCRMetrics(processor)

    print("Initialize Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=my_collator,
        compute_metrics=metrics.compute_metrics,
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        print(f"Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)


    # Save model
    print(f"\nSaving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    print("✓ Training complete!")


if __name__ == "__main__":
    train()
