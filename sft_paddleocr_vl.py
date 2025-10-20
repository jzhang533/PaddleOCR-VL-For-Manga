# -*- coding: utf-8 -*-
"""
Supervised Fine-Tuning (SFT) script for PaddleOCR-VL-0.9B using TRL library.

This version supports loading data from a caption file in format: image_filename|caption
Expected structure:
    data/
    ├── captions.txt (lines: "image1.jpg|caption text here")
    └── images/
        ├── image1.jpg
        ├── image2.jpg
        └── ...

Available tasks:
- 'ocr' -> 'OCR:'
- 'table' -> 'Table Recognition:'
- 'chart' -> 'Chart Recognition:'
- 'formula' -> 'Formula Recognition:'
"""

import os
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from PIL import Image

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)


# ==================== Configuration ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
}


# ==================== Data Classes ====================
@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_path: str = field(
        default="/home/aistudio/.paddlex/official_models/PaddleOCR-VL/PaddleOCR-VL-0.9B",
        metadata={"help": "Path to the PaddleOCR-VL model"}
    )


@dataclass
class DataArguments:
    """Arguments for dataset configuration."""
    images_dir: str = field(
        default="./data/images",
        metadata={"help": "Path to the images directory"}
    )
    caption_file: str = field(
        default="./data/labels.txt",
        metadata={"help": "Path to the caption file (format: image_filename|caption)"}
    )
    task: str = field(
        default="ocr",
        metadata={"help": "Task type: 'ocr', 'table', 'chart', or 'formula'"}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length for training"}
    )


# ==================== Dataset Preparation ====================
def load_dataset_from_caption_file(images_dir: str, caption_file: str) -> Dataset:
    """Load dataset from a caption file."""
    images = []
    texts = []
    
    print(f"Loading captions from {caption_file}...")
    
    try:
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line:
                    continue
                
                parts = line.split('|', 1)
                if len(parts) != 2:
                    print(f"⚠️  Warning: Line {line_num} skipped (invalid format): {line}")
                    continue
                
                image_name, caption = parts
                image_name = image_name.strip()
                caption = caption.strip()
                
                image_path = os.path.join(images_dir, image_name)
                
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found (line {line_num}): {image_path}")
                
                images.append(image_path)
                texts.append(caption)
    
    except FileNotFoundError as e:
        raise Exception(f"Error loading caption file: {e}")
    
    if not images:
        raise ValueError(f"No valid image-caption pairs found in {caption_file}")
    
    print(f"✓ Loaded {len(images)} image-caption pairs")
    
    dataset = Dataset.from_dict({
        "image": images,
        "text": texts,
    })
    
    return dataset


# ==================== Custom Data Collator ====================
class MultimodalCollator:
    """Custom collator for multimodal data - handles batching and padding."""
    
    def __init__(self, processor, task: str, max_seq_length: int = 1024):
        self.processor = processor
        self.task = task
        self.max_seq_length = max_seq_length
        self.task_prompt = TASK_PROMPTS.get(task, TASK_PROMPTS["ocr"])
    
    def __call__(self, batch):
        """
        Process a batch of data.
        
        Args:
            batch: List of dicts with 'image' and 'text' keys
        
        Returns:
            Dictionary with processed batch data ready for model
        """
        images = []
        texts = []
        
        # Load images and prepare texts
        for item in batch:
            # Load image
            image_path = item["image"]
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                raise
            
            # Prepare text with prompt
            caption = item["text"]
            messages = [{"role": "user", "content": self.task_prompt}]
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            full_text = text_prompt + caption
            texts.append(full_text)
        
        # Process all inputs together with proper padding
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        
        # Prepare labels (same as input_ids for language modeling)
        inputs["labels"] = inputs["input_ids"].clone()
        
        # Mask padding tokens in labels (don't compute loss on padding)
        inputs["labels"][inputs["attention_mask"] == 0] = -100
        
        return inputs


def train():
    """Main training function."""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set remove_unused_columns to False to avoid errors
    training_args.remove_unused_columns = False
    
    # Validate paths
    if not os.path.exists(data_args.images_dir):
        raise FileNotFoundError(f"Images directory not found: {data_args.images_dir}")
    if not os.path.exists(data_args.caption_file):
        raise FileNotFoundError(f"Caption file not found: {data_args.caption_file}")
    
    # Load model and processor
    print(f"Loading model from {model_args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Load dataset
    print(f"\nLoading dataset...")
    train_dataset = load_dataset_from_caption_file(
        images_dir=data_args.images_dir,
        caption_file=data_args.caption_file,
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Sample 0 - Image: {train_dataset[0]['image']}, Text length: {len(train_dataset[0]['text'])}")
    
    # Create data collator
    data_collator = MultimodalCollator(
        processor=processor,
        task=data_args.task,
        max_seq_length=data_args.max_seq_length,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    print("✓ Training complete!")


if __name__ == "__main__":
    train()