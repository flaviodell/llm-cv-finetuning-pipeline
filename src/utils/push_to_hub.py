import torch
import json
import yaml
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from src.utils.common import get_env


def push_cv_model_to_hub(config: dict, repo_name: str = "oxford-pets-resnet50"):
    from dotenv import load_dotenv
    load_dotenv()

    token     = get_env("HF_TOKEN")
    api       = HfApi()
    hf_user   = api.whoami(token=token)["name"]
    repo_id   = f"{hf_user}/{repo_name}"

    print(f"Pushing CV model to: https://huggingface.co/{repo_id}")

    # Create repo if it does not exist
    create_repo(
        repo_id=repo_id,
        token=token,
        private=False,
        exist_ok=True
    )

    checkpoint_path = Path("models/cv/best_model.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load checkpoint to extract metadata
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Build model card content
    cv_results_path = Path("outputs/cv/cv_results.json")
    cv_results = {}
    if cv_results_path.exists():
        with open(cv_results_path, "r", encoding="utf-8") as f:
            cv_results = json.load(f)

    model_card = f"""---
language: en
license: mit
tags:
  - image-classification
  - pytorch
  - resnet
  - oxford-pets
  - computer-vision
  - transfer-learning
datasets:
  - oxford-iiit-pet
metrics:
  - accuracy
  - f1
---

# Oxford Pets — ResNet50 classifier

Fine-tuned ResNet50 on the Oxford-IIIT Pet dataset for 37-class breed classification.

## Model details

- **Base model**: ResNet50 pretrained on ImageNet
- **Dataset**: Oxford-IIIT Pet (37 breeds, ~5900 training images)
- **Task**: Multi-class image classification
- **Training**: Transfer learning — backbone frozen, classifier head fine-tuned
- **Framework**: PyTorch + torchvision

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | {cv_results.get('test_accuracy', 'N/A')} |
| Test F1 macro | {cv_results.get('test_f1_macro', 'N/A')} |
| Best val accuracy | {cv_results.get('best_val_accuracy', 'N/A')} |

## Usage

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="best_model.pth"
)

# Load model
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model = models.resnet50()
in_features = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3),
    torch.nn.Linear(in_features, 37)
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = Image.open("your_image.jpg").convert("RGB")
tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(tensor)
    pred_idx = output.argmax(dim=1).item()
    print(checkpoint["classes"][pred_idx])
```

## Classes

{checkpoint.get('classes', [])}
"""

    # Save model card locally
    model_card_path = Path("outputs/cv/MODEL_CARD.md")
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card)

    # Upload files to Hub
    print("Uploading best_model.pth...")
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="best_model.pth",
        repo_id=repo_id,
        token=token
    )

    print("Uploading model card...")
    api.upload_file(
        path_or_fileobj=str(model_card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token
    )

    # Upload evaluation outputs
    for filename in ["cv_results.json", "confusion_matrix.png", "per_class_accuracy.png"]:
        path = Path(f"outputs/cv/{filename}")
        if path.exists():
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"outputs/{filename}",
                repo_id=repo_id,
                token=token
            )

    print(f"\nCV model pushed successfully.")
    print(f"View at: https://huggingface.co/{repo_id}")
    return repo_id


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    push_cv_model_to_hub(config)
