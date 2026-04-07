import torch
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
from pathlib import Path
from src.data.transforms import get_train_transforms, get_val_transforms

RAW_DIR = Path("data/raw")

def get_dataloaders(
    image_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42
):
    # Dataset train+val with augmentation
    full_trainval = datasets.OxfordIIITPet(
        root=RAW_DIR,
        split="trainval",
        transform=get_train_transforms(image_size),
        download=False
    )

    # Calculate split dimensions
    total = len(full_trainval)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset_raw = random_split(
        full_trainval,
        [train_size, val_size],
        generator=generator
    )

    # For validation set, substitute the transformations (no augmentation)
    val_dataset_raw.dataset.transform = get_val_transforms(image_size)

    # Separate test set
    test_dataset = datasets.OxfordIIITPet(
        root=RAW_DIR,
        split="test",
        transform=get_val_transforms(image_size),
        download=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset_raw,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    print(f"Train: {train_size} immagini | Val: {val_size} | Test: {len(test_dataset)}")
    print(f"Classi: {len(full_trainval.classes)}")

    return train_loader, val_loader, test_loader, full_trainval.classes
