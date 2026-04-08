import json
import time
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# All 37 Oxford Pets breeds with species tag
BREEDS = [
    {"name": "Abyssinian",               "species": "cat"},
    {"name": "Bengal",                   "species": "cat"},
    {"name": "Birman",                   "species": "cat"},
    {"name": "Bombay",                   "species": "cat"},
    {"name": "British Shorthair",        "species": "cat"},
    {"name": "Egyptian Mau",             "species": "cat"},
    {"name": "Maine Coon",               "species": "cat"},
    {"name": "Persian",                  "species": "cat"},
    {"name": "Ragdoll",                  "species": "cat"},
    {"name": "Russian Blue",             "species": "cat"},
    {"name": "Siamese",                  "species": "cat"},
    {"name": "Sphynx",                   "species": "cat"},
    {"name": "American Bulldog",         "species": "dog"},
    {"name": "American Pit Bull Terrier","species": "dog"},
    {"name": "Basset Hound",             "species": "dog"},
    {"name": "Beagle",                   "species": "dog"},
    {"name": "Boxer",                    "species": "dog"},
    {"name": "Chihuahua",                "species": "dog"},
    {"name": "English Cocker Spaniel",   "species": "dog"},
    {"name": "English Setter",           "species": "dog"},
    {"name": "German Shorthaired Pointer","species": "dog"},
    {"name": "Great Pyrenees",           "species": "dog"},
    {"name": "Havanese",                 "species": "dog"},
    {"name": "Japanese Chin",            "species": "dog"},
    {"name": "Keeshond",                 "species": "dog"},
    {"name": "Leonberger",               "species": "dog"},
    {"name": "Miniature Pinscher",       "species": "dog"},
    {"name": "Newfoundland",             "species": "dog"},
    {"name": "Pomeranian",               "species": "dog"},
    {"name": "Pug",                      "species": "dog"},
    {"name": "Saint Bernard",            "species": "dog"},
    {"name": "Samoyed",                  "species": "dog"},
    {"name": "Scottish Terrier",         "species": "dog"},
    {"name": "Shiba Inu",                "species": "dog"},
    {"name": "Staffordshire Bull Terrier","species": "dog"},
    {"name": "Wheaten Terrier",          "species": "dog"},
    {"name": "Yorkshire Terrier",        "species": "dog"},
]

# Conversation templates to create variety in the dataset
QUESTION_TEMPLATES = [
    "I am considering adopting a {name}. Can you give me a detailed analysis of this breed?",
    "As a vet, what should I know about the {name} breed in terms of health and temperament?",
    "What are the behavioral traits and training requirements of the {name}?",
    "Can you provide a comprehensive profile of the {name} for a first-time owner?",
    "What health issues should I watch for in a {name}, and how should I manage them?",
    "Describe the ideal living environment and daily routine for a {name}.",
]

SYSTEM_PROMPT = """You are an expert veterinarian and animal behaviorist with 20 years of 
experience specializing in domestic cat and dog breeds. You provide detailed, technically 
accurate, and practical advice about breed characteristics, health predispositions, 
behavioral traits, and training recommendations. Your responses are structured, 
professional, and evidence-based."""


def generate_response(client: Groq, breed: dict, question: str) -> str | None:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Error generating response for {breed['name']}: {e}")
        return None


def generate_dataset(
    output_dir: str = "data/processed",
    examples_per_breed: int = 5,
    delay_seconds: float = 2.0
):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Checkpoint file — saves progress so you can resume if interrupted
    checkpoint_path = output_path / "generated_raw.jsonl"
    
    # Load already generated examples if resuming
    already_generated = set()
    all_examples = []
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                all_examples.append(ex)
                already_generated.add((ex["breed"], ex["question_template_idx"]))
        print(f"Resuming — {len(all_examples)} examples already generated.")

    total = len(BREEDS) * examples_per_breed
    generated = len(all_examples)
    failed = 0

    with open(checkpoint_path, "a", encoding="utf-8") as checkpoint_file:
        for breed in BREEDS:
            for template_idx, template in enumerate(QUESTION_TEMPLATES[:examples_per_breed]):
                
                # Skip if already generated in a previous run
                if (breed["name"], template_idx) in already_generated:
                    continue

                question = template.format(name=breed["name"])
                print(f"[{generated + 1}/{total}] {breed['name']} — template {template_idx + 1}")

                response = None
                retries = 3

                for attempt in range(retries):
                    response = generate_response(client, breed, question)
                    if response:
                        break
                    wait = delay_seconds * (attempt + 2)
                    print(f"  Rate limit hit, waiting {wait}s before retry {attempt + 1}/{retries}...")
                    time.sleep(wait)

                if response:
                    example = {
                        "messages": [
                            {"role": "system",    "content": SYSTEM_PROMPT},
                            {"role": "user",      "content": question},
                            {"role": "assistant", "content": response}
                        ],
                        "breed":                breed["name"],
                        "species":              breed["species"],
                        "question_template_idx": template_idx
                    }
                    all_examples.append(example)
                    # Write immediately so progress is never lost
                    checkpoint_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                    checkpoint_file.flush()
                    generated += 1
                else:
                    print(f"  FAILED after {retries} retries — skipping.")
                    failed += 1

                time.sleep(delay_seconds)

    # Final split and save
    import random
    random.seed(42)
    random.shuffle(all_examples)

    split_idx  = int(len(all_examples) * 0.8)
    train_data = all_examples[:split_idx]
    val_data   = all_examples[split_idx:]

    with open(output_path / "train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(output_path / "val.jsonl", "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDone. Generated: {generated} | Failed: {failed}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")


if __name__ == "__main__":
    generate_dataset(examples_per_breed=5)
