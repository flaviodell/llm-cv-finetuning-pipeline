import torch
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from src.data.dataset import get_dataloaders
from src.training.model_cv import build_cv_model
from src.utils.common import set_seed, get_device


def load_cv_model(checkpoint_path: str, num_classes: int, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_cv_model(num_classes=num_classes, freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(f"Model loaded — best val_accuracy: {checkpoint['val_accuracy']:.4f}")
    return model, checkpoint


def evaluate_cv(config: dict):
    set_seed(config["project"]["seed"])
    device = get_device()

    output_dir = Path("outputs/cv")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test dataloader only
    _, _, test_loader, classes = get_dataloaders(
        image_size=config["cv"]["image_size"],
        batch_size=config["cv"]["batch_size"],
        seed=config["project"]["seed"]
    )
    num_classes = len(classes)

    checkpoint_path = "models/cv/best_model.pth"
    model, checkpoint = load_cv_model(checkpoint_path, num_classes, device)

    # Metrics
    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_metric  = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    cm_metric  = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    print("Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            cm_metric.update(preds, labels)

    accuracy = acc_metric.compute().item()
    f1       = f1_metric.compute().item()
    cm       = cm_metric.compute().cpu().numpy()

    print(f"\nTest set results:")
    print(f"  Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  F1 macro : {f1:.4f}")

    # Save metrics to JSON
    results = {
        "test_accuracy":    round(accuracy, 4),
        "test_f1_macro":    round(f1, 4),
        "num_classes":      num_classes,
        "best_val_accuracy": round(checkpoint["val_accuracy"], 4),
        "best_val_loss":    round(checkpoint["val_loss"], 4)
    }
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to outputs/cv/cv_results.json")

    # Confusion matrix plot
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title("Confusion matrix — ResNet50 fine-tuned on Oxford Pets", fontsize=14)
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("True", fontsize=11)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0,  fontsize=6)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("  Confusion matrix saved to outputs/cv/confusion_matrix.png")

    # Per-class accuracy plot
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    sorted_idx    = per_class_acc.argsort()

    plt.figure(figsize=(14, 8))
    bars = plt.barh(
        [classes[i] for i in sorted_idx],
        [per_class_acc[i] for i in sorted_idx],
        color="steelblue"
    )
    plt.axvline(x=accuracy, color="red", linestyle="--", label=f"Mean accuracy ({accuracy:.2f})")
    plt.xlabel("Accuracy")
    plt.title("Per-class accuracy — Oxford Pets test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_accuracy.png", dpi=150)
    plt.close()
    print("  Per-class accuracy saved to outputs/cv/per_class_accuracy.png")

    return results


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    evaluate_cv(config)
