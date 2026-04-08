import json
from pathlib import Path
from collections import Counter


def inspect_dataset(data_dir: str = "data/processed"):
    for split in ["train", "val"]:
        path = Path(data_dir) / f"{split}.jsonl"
        examples = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))

        species_count = Counter(e["species"] for e in examples)
        avg_len = sum(
            len(e["messages"][2]["content"]) for e in examples
        ) / len(examples)

        print(f"\n--- {split.upper()} SET ---")
        print(f"Total examples : {len(examples)}")
        print(f"Dogs           : {species_count['dog']}")
        print(f"Cats           : {species_count['cat']}")
        print(f"Avg response length: {avg_len:.0f} chars")
        print(f"\nSample example:")
        sample = examples[0]
        print(f"  User: {sample['messages'][1]['content']}")
        print(f"  Assistant (first 200 chars): {sample['messages'][2]['content'][:200]}...")


if __name__ == "__main__":
    inspect_dataset()
