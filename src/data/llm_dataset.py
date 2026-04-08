import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path


class PetExpertDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        self.examples = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                self.examples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]["messages"]

        # Format as a single string: system + user + assistant
        prompt = (
            f"<|system|>\n{messages[0]['content']}\n"
            f"<|user|>\n{messages[1]['content']}\n"
            f"<|assistant|>\n{messages[2]['content']}"
        )

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids      = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Mask padding tokens in labels so they don't contribute to loss
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels
        }


def get_llm_dataloaders(
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    batch_size: int = 4,
    data_dir: str = "data/processed"
):
    train_dataset = PetExpertDataset(
        data_path=f"{data_dir}/train.jsonl",
        tokenizer=tokenizer,
        max_length=max_length
    )
    val_dataset = PetExpertDataset(
        data_path=f"{data_dir}/val.jsonl",
        tokenizer=tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"LLM dataset — train: {len(train_dataset)} | val: {len(val_dataset)}")
    return train_loader, val_loader
