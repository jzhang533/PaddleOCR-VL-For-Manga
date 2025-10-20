# -*- coding: utf-8 -*-
"""
This script includes four task prompts (prompts) and allows switching by modifying the CHOSEN_TASK line without any command line parameters.

Available tasks (CHOSEN_TASK):

- 'ocr' -> 'OCR:'
- 'table' -> 'Table Recognition:'
- 'chart' -> 'Chart Recognition:'
- 'formula' -> 'Formula Recognition:'
To add/modify prompts, change the PROMPTS dictionary as needed.
"""

from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHOSEN_TASK = "ocr"  # Options: 'ocr' | 'table' | 'chart' | 'formula'
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
}

#model_path = "../PaddleOCR-VL/PaddleOCR-VL-0.9B"
model_path = "./sft_output"
image_path = "./data/images/00.jpg"
image = Image.open(image_path).convert("RGB")

model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, dtype=torch.bfloat16
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

messages = [{"role": "user", "content": PROMPTS[CHOSEN_TASK]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(text=[text], images=[image], return_tensors="pt")
inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)

resp = processor.batch_decode(generated, skip_special_tokens=True)[0]
answer = resp.split(text)[-1].strip()
print(answer)
