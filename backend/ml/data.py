import os
from typing import Tuple
from torchvision import datasets, transforms


def make_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def make_datasets(root: str, img_size: int = 224, val_split: float = 0.2):
    """
    Expects directory structure:
      root/
        benign/
        malignant/
    """
    train_tf, val_tf = make_transforms(img_size)
    full = datasets.ImageFolder(root=root, transform=train_tf)
    if val_split <= 0:
        return full, None
    n_total = len(full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_subset, val_subset = torch.utils.data.random_split(full, [n_train, n_val])  # type: ignore
    # Apply val transforms to val subset
    val_subset.dataset.transform = val_tf  # type: ignore
    return train_subset, val_subset

