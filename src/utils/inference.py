import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.utils.common import get_env


def load_finetuned_model(config: dict):
    model_name   = config["llm"]["model_name"]
    adapter_path = "models/llm/best_lora_adapter"
    token        = get_env("HF_TOKEN")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=token
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 300,
    temperature: float = 0.7
) -> str:
    # Format using the same prompt structure used during training
    prompt = (
        f"<|system|>\nYou are an expert veterinarian and animal behaviorist "
        f"with 20 years of experience specializing in domestic cat and dog breeds. "
        f"You provide detailed, technically accurate, and practical advice.\n"
        f"<|user|>\n{question}\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    # Decode only the newly generated tokens, not the prompt
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_inference_test(config: dict):
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model(config)

    # Test questions covering different breeds and question types
    test_questions = [
        "What are the main health concerns for a Persian cat owner?",
        "How should I train a Beagle puppy given its stubborn nature?",
        "Compare the temperament of a Maine Coon and a Siamese cat.",
        "What exercise routine would you recommend for a Saint Bernard?",
        "I have a Sphynx cat — what specific grooming does it need despite having no fur?"
    ]

    print("\n" + "=" * 55)
    print("Inference test — fine-tuned Mistral 7B (pet expert)")
    print("=" * 55)

    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}] USER: {question}")
        response = generate_response(model, tokenizer, question)
        print(f"    ASSISTANT: {response[:400]}...")
        print("-" * 55)


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    run_inference_test(config)
