---
language: en
license: mit
tags:
  - text-generation
  - pytorch
  - mistral
  - lora
  - peft
  - veterinary
  - fine-tuning
base_model: mistralai/Mistral-7B-Instruct-v0.3
---

# Pet Expert — Mistral 7B + LoRA

Fine-tuned Mistral 7B Instruct on a synthetic veterinary dataset
covering 37 cat and dog breeds from the Oxford-IIIT Pet dataset.

## Model details

- **Base model**: mistralai/Mistral-7B-Instruct-v0.3
- **Method**: LoRA (r=16, alpha=32) with 4-bit NF4 quantization
- **Dataset**: Synthetic — 185 instruction/response pairs
- **Domain**: Veterinary — breed temperament, health, training advice
- **Framework**: HuggingFace PEFT + bitsandbytes

## Results (RAGAS evaluation)

| Metric | Value |
|--------|-------|
| Faithfulness | 0.491 |
| Answer relevancy | 0.9364 |
| Evaluated on | 37 questions |

## Related model

This LLM adapter is part of a two-model project.
The companion CV classifier is available at:
`flaviodell/oxford-pets-resnet50`

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("flaviodell/pet-expert-mistral7b-lora")
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "flaviodell/pet-expert-mistral7b-lora")
model.eval()

prompt = (
    "<|system|>\nYou are an expert veterinarian.\n"
    "<|user|>\nWhat are the health concerns for a Persian cat?\n"
    "<|assistant|>\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
