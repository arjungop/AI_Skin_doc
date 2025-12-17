#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import time
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # add project root to path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from backend.ml.model import SkinClassifier
from backend.ml.data import make_transforms
import json


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    train_tf, val_tf = make_transforms(args.img_size)
    train_root = os.path.join(args.data_dir, 'train') if args.split_subdirs else args.data_dir
    train_ds = datasets.ImageFolder(root=train_root, transform=train_tf)
    if args.split_subdirs:
        val_root = os.path.join(args.data_dir, 'val')
        if os.path.isdir(val_root):
            val_ds = datasets.ImageFolder(root=val_root, transform=val_tf)
        else:
            # Fallback: create a random val split from train
            n_total = len(train_ds)
            n_val = max(1, int(n_total * args.val_split))
            n_train = max(1, n_total - n_val)
            train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])  # type: ignore
            if val_ds is not None:
                val_ds.dataset.transform = val_tf  # type: ignore
    else:
        # simple split
        n_total = len(train_ds)
        n_val = int(n_total * args.val_split)
        n_train = n_total - n_val
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])  # type: ignore
        if val_ds is not None:
            val_ds.dataset.transform = val_tf  # type: ignore

    # Pin memory is useful on CUDA; MPS/CPU ignore or warn.
    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin) if val_ds else None

    # Helpful context
    try:
        n_train = len(train_ds) if not isinstance(train_ds, torch.utils.data.Subset) else len(train_ds)
        n_val = (len(val_ds) if val_ds is not None else 0) if not isinstance(val_ds, torch.utils.data.Subset) else len(val_ds)
        steps = len(train_loader)
        print(f"Train samples: {n_train} | Val samples: {n_val} | Batches/epoch: {steps}")
    except Exception:
        pass

    model = SkinClassifier(num_classes=2, backbone=args.backbone, pretrained=not args.no_pretrained).to(device)
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith('backbone.fc'):
                p.requires_grad = False

    # Class-weighted loss to help with imbalance
    try:
        if isinstance(train_ds, torch.utils.data.Subset):
            base = train_ds.dataset  # type: ignore
            idxs = train_ds.indices  # type: ignore
            targets = [base.samples[i][1] for i in idxs]
        else:
            targets = [y for _, y in train_ds.samples]  # type: ignore
        import numpy as np
        counts = np.bincount(targets, minlength=2) + 1e-6
        weights = (counts.sum() / counts).astype('float32')
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Class weights: {weights}")
    except Exception:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val = 0.0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0
        for bi, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += accuracy(logits.detach(), y)
            n_batches += 1
            if bi % max(1, (len(train_loader)//10 or 1)) == 1:
                # periodic progress (about 10 updates per epoch)
                print(f"  - epoch {epoch} [{bi}/{len(train_loader)}] loss {loss.item():.4f}", flush=True)
        scheduler.step()
        train_loss = running_loss / max(1, n_batches)
        train_acc = running_acc / max(1, n_batches)

        val_acc = None
        if val_loader:
            model.eval()
            ra, nb = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    ra += accuracy(logits, y)
                    nb += 1
            val_acc = ra / max(1, nb)
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), str(out_path))
        else:
            # no val split; save periodically
            if epoch == args.epochs:
                torch.save(model.state_dict(), str(out_path))

        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} - {dt:.1f}s - loss {train_loss:.4f} - acc {train_acc:.3f} - val_acc {val_acc if val_acc is not None else '-'}")

        # Early stop on target accuracy
        if val_loader and args.target_acc is not None and val_acc is not None and val_acc >= args.target_acc:
            print(f"Early stop: reached target val_acc {val_acc:.3f} >= {args.target_acc}")
            break

    # Optional quick fine-tune: unfreeze and run 1 short epoch at lower LR
    if args.freeze_backbone and val_loader:
        try:
            for p in model.parameters():
                p.requires_grad = True
            ft_lr = max(args.lr * 0.2, 1e-5)
            optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=args.weight_decay)
            model.train()
            print(f"Fine-tuning for 1 epoch at lr={ft_lr}")
            ra = 0.0; nb = 0
            for bi, (x, y) in enumerate(train_loader, start=1):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward(); optimizer.step()
                if bi % max(1, (len(train_loader)//5 or 1)) == 0:
                    print(f"  - finetune [{bi}/{len(train_loader)}] loss {loss.item():.4f}")
            # Validate after finetune
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    ra += accuracy(logits, y); nb += 1
            v = ra / max(1, nb)
            print(f"Post-finetune val_acc: {v:.3f}")
            if v > best_val:
                torch.save(model.state_dict(), str(out_path))
        except Exception as e:
            print("Finetune skipped:", e)

    # Save metadata: class mapping, malignant index, backbone
    try:
        if isinstance(train_ds, torch.utils.data.Subset):
            class_to_idx = train_ds.dataset.class_to_idx  # type: ignore
        else:
            class_to_idx = train_ds.class_to_idx
        malignant_idx = class_to_idx.get('malignant') if isinstance(class_to_idx, dict) else None
        meta = {
            'class_to_idx': class_to_idx,
            'malignant_index': int(malignant_idx) if malignant_idx is not None else 1,
            'backbone': args.backbone,
        }
        meta_path = Path(str(out_path).rsplit('.',1)[0] + '.json')
        meta_path.write_text(json.dumps(meta))
        print(f"Saved metadata to: {meta_path}")
    except Exception as e:
        print('Warning: could not save class mapping meta:', e)

    print(f"Saved weights to: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train skin lesion classifier (benign vs malignant)")
    p.add_argument('--data-dir', required=True, help='Path to dataset. If --split-subdirs is used, expects train/ and val/ subfolders.')
    p.add_argument('--split-subdirs', action='store_true', help='Use data_dir/train and data_dir/val instead of random split')
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18','resnet50'])
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--output', type=str, default='backend/ml/weights/skin_resnet18.pth')
    p.add_argument('--target-acc', type=float, default=None, help='Early stop when val accuracy >= target')
    p.add_argument('--fast', action='store_true', help='Fast mode: fewer epochs, smaller images, freeze backbone for <5 min runs')
    args = p.parse_args()
    if args.fast:
        # Heuristics to keep training under ~5 minutes on typical laptops
        if args.epochs > 3:
            args.epochs = 3
        if args.img_size > 192:
            args.img_size = 192
        args.freeze_backbone = True if not args.freeze_backbone else args.freeze_backbone
        if args.backbone == 'resnet50':
            # Smaller backbone is faster
            args.backbone = 'resnet18'
        if args.target_acc is None:
            args.target_acc = 0.92
        # Modestly higher LR speeds convergence when freezing
        if not args.no_pretrained and args.lr < 5e-4:
            args.lr = 5e-4
    train(args)
