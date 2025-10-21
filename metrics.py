"""
Metrics computation for PaddleOCR-VL evaluation.

This module provides CER (Character Error Rate) and accuracy metrics
for evaluating OCR model performance during training.
"""

import evaluate
import numpy as np


class OCRMetrics:
    """Compute CER and accuracy metrics for OCR evaluation."""

    def __init__(self, processor):
        """
        Initialize metrics with processor for decoding predictions.

        Args:
            processor: The model processor with tokenizer for decoding
        """
        # self.cer_metric = evaluate.load('../evaluate/metrics/cer')
        self.cer_metric = evaluate.load("cer")
        self.processor = processor

    def compute_metrics(self, pred):
        """
        Compute CER and accuracy metrics from predictions.

        Args:
            pred: EvalPrediction object with predictions and label_ids

        Returns:
            dict: Dictionary containing 'cer' and 'accuracy' metrics
        """
        label_ids = pred.label_ids
        pred_ids = pred.predictions

        # Handle logits vs token IDs
        if len(pred_ids.shape) == 3:
            # Predictions are logits, take argmax
            pred_ids = np.argmax(pred_ids, axis=-1)

        print(f"Shapes - labels: {label_ids.shape}, predictions: {pred_ids.shape}")

        # Decode predictions
        pred_str = self.processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )

        # Replace -100 with pad_token_id for decoding labels
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # Remove whitespace for fair comparison
        pred_str = np.array(["".join(text.split()) for text in pred_str])
        label_str = np.array(["".join(text.split()) for text in label_str])

        results = {}

        # Compute CER (Character Error Rate)
        try:
            results["cer"] = self.cer_metric.compute(
                predictions=pred_str, references=label_str
            )
        except Exception as e:
            print(f"Error computing CER: {e}")
            print(f"Sample predictions: {pred_str[:3]}")
            print(f"Sample labels: {label_str[:3]}")
            results["cer"] = 1.0  # Worst case CER

        # Compute exact match accuracy
        results["accuracy"] = (pred_str == label_str).mean()

        return results
