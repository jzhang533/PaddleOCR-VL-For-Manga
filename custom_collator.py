from dataclasses import dataclass, field
from typing import Any, Optional, Union

from transformers import ProcessorMixin


@dataclass
class CustomDataCollatorForVisionLanguageModeling:
    """
    Vision-language data collator with completion-only loss support.

    This collator extends TRL's DataCollatorForVisionLanguageModeling approach
    to support completion-only training with messages format datasets.
    It masks instruction/prompt tokens so the model only learns from assistant
    responses, following the standard instruction-tuning pattern.

    Why custom: TRL's DataCollatorForVisionLanguageModeling has limitations:
    - completion_only_loss only works with prompt-completion format
    - prompt-completion format doesn't support pad_to_multiple_of for VL
    - This implementation works with messages format + all features

    Args:
        processor: The model processor with tokenizer
        max_length: Maximum sequence length (image + text tokens)
        pad_to_multiple_of: Pad to multiple for GPU efficiency
            (e.g., 8 for A100)
        return_tensors: Format of returned tensors
        response_template: String marking start of assistant response.
            Default "Assistant: " (with space) matches PaddleOCR-VL's
            chat template. Everything before (and including) this template
            is masked.

    Example:
        >>> from transformers import AutoProcessor
        >>> processor = AutoProcessor.from_pretrained("PaddleOCR-VL")
        >>> collator = CustomDataCollatorForVisionLanguageModeling(
        ...     processor,
        ...     max_length=2048,
        ...     pad_to_multiple_of=8,
        ...     response_template="Assistant: ",
        ... )
        >>> examples = [
        ...     {
        ...         "images": [image],
        ...         "messages": [
        ...             {
        ...                 "role": "user",
        ...                 "content": [
        ...                     {"type": "image", "image": image},
        ...                     {"type": "text", "text": "OCR:"},
        ...                 ],
        ...             },
        ...             {
        ...                 "role": "assistant",
        ...                 "content": [
        ...                     {"type": "text", "text": "recognized text"},
        ...                 ],
        ...             },
        ...         ],
        ...     }
        ... ]
        >>> batch = collator(examples)

    Note: For PaddleOCR-VL with dynamic resolution:
    - Small images: ~400-800 tokens
    - Large images: 1000-2000+ tokens
    - Set max_length high enough to avoid truncating image tokens
    """

    processor: ProcessorMixin
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    response_template: Union[str, list[int]] = field(default="Assistant: ")
    instruction_template: Optional[Union[str, list[int]]] = None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate examples into a batch."""
        # Collect images as list-of-lists (each example's images)
        images = [example["images"] for example in examples]

        # Extract messages and apply chat template
        messages = [example["messages"] for example in examples]
        texts = self.processor.apply_chat_template(messages, tokenize=False)

        # Process images and text together
        # IMPORTANT: Images must be list-of-lists matching the processor's
        # expectation: [[img1], [img2], ...] for single-image samples
        output = self.processor(
            text=texts,
            images=images,
            padding=True,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=False,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )

        # Create labels from input_ids
        labels = output["input_ids"].clone()

        # Mask padding tokens
        labels[output["attention_mask"] == 0] = -100

        # Mask instruction/prompt tokens (completion-only training)
        # This follows the pattern from TRL's DataCollatorForCompletionOnlyLM
        response_token_ids = self._get_token_ids(self.response_template)

        for i in range(len(labels)):
            response_positions = self._find_subsequence(
                output["input_ids"][i], response_token_ids
            )

            if not response_positions:
                # If response template not found, mask everything
                # (safety fallback)
                labels[i, :] = -100
                continue

            # Mask everything up to and including the response template
            # The actual response starts after the template
            response_start = response_positions[0] + len(response_token_ids)
            labels[i, :response_start] = -100

        output["labels"] = labels

        return output

    def _get_token_ids(self, template: Union[str, list[int]]) -> list[int]:
        """Convert template string to token IDs if needed."""
        if isinstance(template, str):
            return self.processor.tokenizer.encode(template, add_special_tokens=False)
        return template

    def _find_subsequence(self, tensor, subsequence: list[int]) -> list[int]:
        """
        Find all starting positions of subsequence in tensor.

        This is more robust than searching for a single token, as it
        handles multi-token templates correctly.

        Returns:
            list[int]: List of starting positions where subsequence is found
        """
        positions = []
        subseq_len = len(subsequence)

        for i in range(len(tensor) - subseq_len + 1):
            if all(tensor[i + j] == subsequence[j] for j in range(subseq_len)):
                positions.append(i)

        return positions
