---
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
| Test accuracy | 0.9008 |
| Test F1 macro | 0.8989 |
| Best val accuracy | 0.9185 |

## Usage

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="flaviodell/oxford-pets-resnet50",
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

['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier']
