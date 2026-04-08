import torch
import wandb
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm

from src.data.llm_dataset import get_llm_dataloaders
from src.utils.common import set_seed, get_env


def load_model_and_tokenizer(config: dict):
    model_name = config["llm"]["model_name"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=get_env("HF_TOKEN")
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=get_env("HF_TOKEN")
    )

    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def apply_lora(model, config: dict):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["llm"]["lora_r"],
        lora_alpha=config["llm"]["lora_alpha"],
        lora_dropout=config["llm"]["lora_dropout"],
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()

    return total_loss / len(loader)


def run_llm_training(config: dict):
    set_seed(config["project"]["seed"])

    wandb.init(
        project=config["wandb"]["project"],
        name="llm-lora-pet-expert",
        config=config["llm"]
    )

    model, tokenizer = load_model_and_tokenizer(config)
    model = apply_lora(model, config)
    device = next(model.parameters()).device

    train_loader, val_loader = get_llm_dataloaders(
        tokenizer=tokenizer,
        max_length=config["llm"]["max_length"],
        batch_size=config["llm"]["batch_size"]
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["llm"]["learning_rate"]
    )

    best_val_loss = float("inf")
    checkpoint_dir = Path("models/llm/best_lora_adapter")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["llm"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss   = validate(model, val_loader, device)

        print(f"Epoch {epoch} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        wandb.log({
            "llm/train_loss": train_loss,
            "llm/val_loss":   val_loss,
            "epoch":          epoch
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(checkpoint_dir))
            tokenizer.save_pretrained(str(checkpoint_dir))
            print(f"  Adapter saved (val_loss: {best_val_loss:.4f})")

    wandb.finish()
    print("LLM fine-tuning complete.")


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    run_llm_training(config)
