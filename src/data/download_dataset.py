import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_oxford_pets():
    print("Download Oxford Pets in corso...")
    datasets.OxfordIIITPet(
        root=RAW_DIR,
        split="trainval",
        download=True
    )
    datasets.OxfordIIITPet(
        root=RAW_DIR,
        split="test",
        download=True
    )
    print("Download completato.")

if __name__ == "__main__":
    download_oxford_pets()