import torch
import torch.nn as nn
import torchvision.models as models


def build_cv_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    # Load ResNet50 pretrained on ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        # Freeze all layers except the final classifier -> transfer learning
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer to match our number of classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_cv_model(num_classes=37, freeze_backbone=True)
    print(f"Trainable parameters: {count_trainable_params(model):,}")
