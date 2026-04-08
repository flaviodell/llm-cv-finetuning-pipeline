import json
import os
import torch
import yaml
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.utils.common import get_env, set_seed
from src.utils.inference import generate_response


def load_eval_questions(data_dir: str = "data/processed") -> list:
    # Load validation set as evaluation questions
    questions = []
    with open(f"{data_dir}/val.jsonl", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            questions.append({
                "question":    ex["messages"][1]["content"],
                "ground_truth": ex["messages"][2]["content"],
                "breed":       ex["breed"]
            })
    return questions


def build_ragas_dataset(
    model,
    tokenizer,
    questions: list,
    max_new_tokens: int = 300
) -> Dataset:
    print(f"Generating answers for {len(questions)} questions...")
    records = []

    for i, item in enumerate(questions):
        print(f"  [{i + 1}/{len(questions)}] {item['breed']}")
        answer = generate_response(
            model, tokenizer, item["question"],
            max_new_tokens=max_new_tokens
        )
        records.append({
            "question":         item["question"],
            "answer":           answer,
            "ground_truth":     item["ground_truth"],
            # RAGAS requires a contexts field even for non-RAG evaluation
            "contexts":         [item["ground_truth"]]
        })

    return Dataset.from_list(records)


def run_llm_evaluation(config: dict):
    set_seed(config["project"]["seed"])
    output_dir = Path("outputs/llm")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fine-tuned model
    print("Loading fine-tuned model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained("models/llm/best_lora_adapter")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config["llm"]["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        token=get_env("HF_TOKEN")
    )
    model = PeftModel.from_pretrained(base_model, "models/llm/best_lora_adapter")
    model.eval()

    # Build evaluation dataset
    questions    = load_eval_questions()
    ragas_dataset = build_ragas_dataset(model, tokenizer, questions)

    # Use Groq as the judge LLM for RAGAS
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama3-8b-8192",
            api_key=get_env("GROQ_API_KEY"),
            temperature=0
        )
    )

    print("\nRunning RAGAS evaluation...")
    scores = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm
    )

    results = {
        "faithfulness":     round(float(scores["faithfulness"]), 4),
        "answer_relevancy": round(float(scores["answer_relevancy"]), 4),
        "num_questions":    len(questions)
    }

    print(f"\nRAGAS results:")
    print(f"  Faithfulness     : {results['faithfulness']}")
    print(f"  Answer relevancy : {results['answer_relevancy']}")

    with open(output_dir / "llm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to outputs/llm/llm_results.json")

    return results


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    run_llm_evaluation(config)
