import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch
from src.data.dataset import get_dataloaders
from collections import Counter
from pathlib import Path


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)

def inspect_batch():
    train_loader, _, _, classes = get_dataloaders(batch_size=8)
    images, labels = next(iter(train_loader))

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, ax in enumerate(axes.flat):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(classes[labels[i].item()], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/cv/sample_batch.png", dpi=150)
    plt.show()
    print("Immagine salvata in outputs/cv/sample_batch.png")

def plot_class_distribution():
    raw = datasets.OxfordIIITPet(root=Path("data/raw"), split="trainval", download=False)
    counts = Counter([raw.classes[l] for l in raw._labels])

    classes = list(counts.keys())
    values  = list(counts.values())

    plt.figure(figsize=(16, 5))
    plt.bar(classes, values, color="steelblue")
    plt.xticks(rotation=90, fontsize=7)
    plt.title("Distribuzione classi — Oxford Pets (trainval)")
    plt.tight_layout()
    plt.savefig("outputs/cv/class_distribution.png", dpi=150)
    plt.show()
    print("Grafico salvato in outputs/cv/class_distribution.png")

if __name__ == "__main__":
    inspect_batch()
    plot_class_distribution()
    