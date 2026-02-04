"""
Hierarchical Skin Disease Classifier
Level 1: 5 Categories (Cancer, Benign, Inflammatory, Infectious, Pigmentary)
Level 2: 19 Diseases
"""
import torch
import torch.nn as nn
from torchvision import models



class HierarchicalSkinClassifier(nn.Module):
    """
    Hierarchical classifier with two heads:
    - Category head: 5 classes (Cancer, Benign, Inflammatory, Infectious, Pigmentary)
    - Disease head: 19 classes (specific diseases)
    """
    def __init__(self, num_categories: int = 5, num_diseases: int = 19, 
                 backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        
        # Backbone (feature extractor)
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features  # 512 for resnet18
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features  # 2048 for resnet50
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove original fc layer
        self.backbone.fc = nn.Identity()
        
        # Dual classification heads
        self.category_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_categories)
        )
        
        self.disease_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_diseases)
        )
        
        self.num_categories = num_categories
        self.num_diseases = num_diseases

    def forward(self, x):
        """
        Returns:
            category_logits: (batch, num_categories)
            disease_logits: (batch, num_diseases)
        """
        features = self.backbone(x)  # (batch, in_features)
        category_logits = self.category_head(features)
        disease_logits = self.disease_head(features)
        return category_logits, disease_logits

    def predict(self, x):
        """For inference: returns predicted category and disease indices"""
        cat_logits, dis_logits = self.forward(x)
        cat_pred = torch.argmax(cat_logits, dim=1)
        dis_pred = torch.argmax(dis_logits, dim=1)
        return cat_pred, dis_pred
