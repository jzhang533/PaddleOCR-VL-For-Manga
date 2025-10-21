"""
This script includes four task prompts (prompts) and allows switching by modifying the CHOSEN_TASK line without any command line parameters.

Available tasks (CHOSEN_TASK):

- 'ocr' -> 'OCR:'
- 'table' -> 'Table Recognition:'
- 'chart' -> 'Chart Recognition:'
- 'formula' -> 'Formula Recognition:'
To add/modify prompts, change the PROMPTS dictionary as needed.
"""

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHOSEN_TASK = "ocr"  # Options: 'ocr' | 'table' | 'chart' | 'formula'
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
}

# model_path = "/home/PaddleOCR-VL"
model_path = "../sft_output/checkpoint-10000"

model = (
    AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    .to(DEVICE)
    .eval()
)

processor_path = "/home/PaddleOCR-VL"
processor = AutoProcessor.from_pretrained(
    processor_path, trust_remote_code=True, use_fast=True
)

# Set pad_token_id to avoid warning during generation
if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

folder_path = Path("./data/images")

for item_path in tqdm(sorted(folder_path.iterdir()), desc="Processing images"):
    image = Image.open(item_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPTS[CHOSEN_TASK]},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {
        k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = generated[:, input_length:]  # Slice only new tokens
    answer = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    print(item_path)
    print(answer)
