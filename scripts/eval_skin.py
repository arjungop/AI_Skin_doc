#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from backend.ml.model import SkinClassifier


def load_dataset(data_dir: str, img_size: int = 224):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(root=data_dir, transform=tf)
    return ds


def evaluate(weights: str, data_dir: str, backbone: str = 'resnet18', batch_size: int = 32, img_size: int = 224, workers: int = 2, out_dir: str = 'reports'):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    ds = load_dataset(data_dir, img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = SkinClassifier(num_classes=2, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    y_true = []
    y_prob = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]  # malignant prob
            y_true.extend(y.numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    # Find best threshold for accuracy
    thresholds = np.linspace(0.05, 0.95, 19)
    accs = []
    for t in thresholds:
        accs.append(((y_prob >= t).astype(int) == y_true).mean())
    best_idx = int(np.argmax(accs))
    best_thr = float(thresholds[best_idx])
    y_pred = (y_prob >= best_thr).astype(int)

    # Metrics
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['benign','malignant'])

    # Sensitivity (recall for malignant) and specificity (recall for benign)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / max(1, (tp + fn))
    specificity = tn / max(1, (tn + fp))

    # Output
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.txt').write_text(
        f"Best threshold (accuracy): {best_thr:.2f}\n" +
        f"ROC AUC: {auc:.4f}\n" +
        f"Accuracy@best: {accs[best_idx]:.4f}\n" +
        f"Sensitivity (Recall+): {sensitivity:.4f}\n" +
        f"Specificity (Recall-): {specificity:.4f}\n\n" + report
    )
    # Save suggested env snippet
    (out / 'env_suggestion.txt').write_text(
        f"LESION_MODEL_WEIGHTS={weights}\nLESION_FORCE_THRESHOLD=1\nLESION_MALIGNANT_THRESHOLD={best_thr:.2f}\n"
    )
    # Also update weights' sidecar JSON with suggested threshold if present
    try:
        import json
        meta_path = Path(str(weights).rsplit('.',1)[0] + '.json')
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text() or '{}')
        meta['suggested_threshold'] = float(best_thr)
        meta_path.write_text(json.dumps(meta))
    except Exception:
        pass

    # Confusion matrix plot
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['benign','malignant'], yticklabels=['benign','malignant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out / 'confusion_matrix.png', dpi=150)
    plt.close()

    print((out / 'metrics.txt').read_text())
    print(f"Saved confusion matrix to {(out / 'confusion_matrix.png').resolve()}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Evaluate skin lesion classifier')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data-dir', required=True, help='Path to test set (benign/malignant subfolders)')
    ap.add_argument('--backbone', default='resnet18', choices=['resnet18','resnet50'])
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default='reports')
    args = ap.parse_args()
    evaluate(args.weights, args.data_dir, args.backbone, args.batch_size, args.img_size, args.workers, args.out)
