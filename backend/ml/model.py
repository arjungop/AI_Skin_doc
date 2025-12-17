import torch
import torch.nn as nn
from torchvision import models


class SkinClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_feats = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_feats, num_classes)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_feats = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_feats, num_classes)
        else:
            raise ValueError("Unsupported backbone: " + backbone)

    def forward(self, x):
        return self.backbone(x)

