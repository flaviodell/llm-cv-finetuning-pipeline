import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.utils.common import get_env


def get_model_size_mb(model) -> float:
    # Calculate total memory occupied by model parameters in MB
    total_bytes = sum(
        p.nelement() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


def get_gpu_memory_mb() -> float:
    # Return currently allocated GPU memory in MB
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def analyze_quantization(config: dict):
    model_name = config["llm"]["model_name"]
    token = get_env("HF_TOKEN")

    results = {}

    print("=" * 55)
    print("Quantization analysis — Mistral 7B")
    print("=" * 55)

    # --- FP16 (half precision, no quantization) ---
    print("\nLoading FP16 model...")
    torch.cuda.reset_peak_memory_stats()

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token
    )
    fp16_params_mb  = get_model_size_mb(model_fp16)
    fp16_gpu_mb     = torch.cuda.max_memory_allocated() / (1024 ** 2)
    results["fp16"] = {"params_mb": fp16_params_mb, "gpu_mb": fp16_gpu_mb}

    print(f"  Parameter size : {fp16_params_mb:.0f} MB")
    print(f"  Peak GPU memory: {fp16_gpu_mb:.0f} MB")

    del model_fp16
    torch.cuda.empty_cache()

    # --- 4-bit quantization (same config used during training) ---
    print("\nLoading 4-bit quantized model + LoRA adapter...")
    torch.cuda.reset_peak_memory_stats()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=token
    )

    # Load the LoRA adapter on top of the quantized base model
    adapter_path = "models/llm/best_lora_adapter"
    model_4bit = PeftModel.from_pretrained(base_model, adapter_path)

    quant_params_mb  = get_model_size_mb(model_4bit)
    quant_gpu_mb     = torch.cuda.max_memory_allocated() / (1024 ** 2)
    results["4bit"]  = {"params_mb": quant_params_mb, "gpu_mb": quant_gpu_mb}

    print(f"  Parameter size : {quant_params_mb:.0f} MB")
    print(f"  Peak GPU memory: {quant_gpu_mb:.0f} MB")

    # --- Summary ---
    print("\n" + "=" * 55)
    print("Summary")
    print("=" * 55)
    reduction_params = (1 - quant_params_mb / fp16_params_mb) * 100
    reduction_gpu    = (1 - quant_gpu_mb    / fp16_gpu_mb)    * 100
    print(f"  Parameter size reduction : {reduction_params:.1f}%")
    print(f"  GPU memory reduction     : {reduction_gpu:.1f}%")
    print(f"  FP16 GPU usage : {fp16_gpu_mb:.0f} MB  ({fp16_gpu_mb/1024:.1f} GB)")
    print(f"  4-bit GPU usage: {quant_gpu_mb:.0f} MB  ({quant_gpu_mb/1024:.1f} GB)")
    print("=" * 55)

    return results, model_4bit


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    analyze_quantization(config)
