import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
import wandb
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> dict:
    model.train()
    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        acc_metric.update(preds, labels)
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    accuracy = acc_metric.compute().item()

    return {"train/loss": avg_loss, "train/accuracy": accuracy}


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> dict:
    model.eval()
    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_metric  = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    accuracy = acc_metric.compute().item()
    f1       = f1_metric.compute().item()

    return {
        "val/loss":     avg_loss,
        "val/accuracy": accuracy,
        "val/f1_macro": f1
    }


def run_training(config: dict):
    import yaml
    from pathlib import Path
    from src.data.dataset import get_dataloaders
    from src.training.model_cv import build_cv_model
    from src.utils.common import set_seed, get_device

    set_seed(config["project"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, _, classes = get_dataloaders(
        image_size=config["cv"]["image_size"],
        batch_size=config["cv"]["batch_size"],
        seed=config["project"]["seed"]
    )
    num_classes = len(classes)

    # Build model
    model = build_cv_model(num_classes=num_classes, freeze_backbone=True).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["cv"]["learning_rate"]
    )

    # LR scheduler: reduce LR when val/loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # W&B init
    wandb.init(
        project=config["wandb"]["project"],
        name="cv-resnet50-oxford-pets",
        config=config["cv"]
    )

    # Training loop
    best_val_loss = float("inf")
    checkpoint_path = Path("models/cv/best_model.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["cv"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['cv']['epochs']}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, num_classes
        )
        val_metrics = validate(
            model, val_loader, criterion, device, num_classes
        )

        # Merge and log all metrics
        metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
        wandb.log(metrics)

        # Print summary
        print(
            f"  train loss: {train_metrics['train/loss']:.4f} | "
            f"train acc: {train_metrics['train/accuracy']:.4f} | "
            f"val loss: {val_metrics['val/loss']:.4f} | "
            f"val acc: {val_metrics['val/accuracy']:.4f} | "
            f"val F1: {val_metrics['val/f1_macro']:.4f}"
        )

        # Save best checkpoint
        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":    best_val_loss,
                "val_accuracy": val_metrics["val/accuracy"],
                "num_classes": num_classes,
                "classes":     classes
            }, checkpoint_path)
            print(f"  Checkpoint saved (val_loss: {best_val_loss:.4f})")

        scheduler.step(val_metrics["val/loss"])

    wandb.finish()
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {checkpoint_path}")


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    run_training(config)
