#!/usr/bin/env python3
"""Prepare a uniform, ImageFolder-compatible dataset layout.

Creates:
  <out_dir>/
    train/benign/, train/malignant/
    val/benign/,   val/malignant/
    test/benign/,  test/malignant/

Sources supported:
- ISIC 2019: uses isic_data/isic_2019/ISIC_2019_Training_Input + ISIC_2019_Training_GroundTruth.csv
- DiverseDerm (Kaggle skin-cancer9-classesisic): uses datasets/diverse_derm/.../Train and .../Test

Notes:
- By default we symlink files into the output tree to avoid duplicating many GB.
- The split is stratified by label (benign/malignant) and reproducible via --seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    src: Path
    label: str  # 'benign' | 'malignant'
    source: str  # 'isic2019' | 'diversederm'


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """Create dst from src using symlink, hardlink, or copy."""
    ensure_dir(dst.parent)

    if dst.exists():
        return

    if mode == "symlink":
        # Use relative symlinks to keep the tree movable.
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)
        return

    if mode == "hardlink":
        os.link(src, dst)
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    raise ValueError(f"Unknown mode: {mode}")


def stratified_split(
    samples: List[Sample],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Dict[str, List[Sample]]:
    """Split into train/val/test stratified by label."""
    if not (0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0 and (val_frac + test_frac) < 1.0):
        raise ValueError("val_frac and test_frac must be in [0,1) and val+test < 1")

    rng = random.Random(seed)

    by_label: Dict[str, List[Sample]] = {"benign": [], "malignant": []}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    out: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    for label, group in by_label.items():
        if not group:
            continue
        group = list(group)
        rng.shuffle(group)

        n = len(group)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        # Keep at least 1 sample in train when possible.
        if n >= 3 and (n_test + n_val) >= n:
            n_test = min(n_test, n - 2)
            n_val = min(n_val, n - 1 - n_test)

        test = group[:n_test]
        val = group[n_test : n_test + n_val]
        train = group[n_test + n_val :]

        out["test"].extend(test)
        out["val"].extend(val)
        out["train"].extend(train)

    # Mix labels within each split for better IO locality.
    for k in out:
        rng.shuffle(out[k])

    return out


def load_isic2019_samples(repo_root: Path) -> List[Sample]:
    base = repo_root / "isic_data" / "isic_2019"
    img_dir = base / "ISIC_2019_Training_Input"
    gt_csv = base / "ISIC_2019_Training_GroundTruth.csv"

    if not img_dir.is_dir() or not gt_csv.exists():
        return []

    # ISIC 2019 GT is one-hot columns (MEL, NV, BCC, AK, BKL, DF, VASC, SCC)
    malignant_cols = {"MEL", "BCC", "SCC", "AK"}

    rows: Dict[str, Dict[str, str]] = {}
    with gt_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"No header found in {gt_csv}")
        for r in reader:
            image_id = (r.get("image") or r.get("image_id") or "").strip()
            if not image_id:
                continue
            rows[image_id] = r

    samples: List[Sample] = []
    for p in img_dir.iterdir():
        if not p.is_file() or not is_image(p):
            continue
        image_id = p.stem
        r = rows.get(image_id)
        if not r:
            continue

        is_malignant = False
        for c in malignant_cols:
            v = (r.get(c) or "0").strip()
            if v in {"1", "1.0", "True", "true"}:
                is_malignant = True
                break

        label = "malignant" if is_malignant else "benign"
        samples.append(Sample(src=p, label=label, source="isic2019"))

    return samples


def load_diversederm_samples(repo_root: Path) -> Tuple[List[Sample], List[Sample]]:
    """Return (train_pool, test_set) from DiverseDerm."""
    base = repo_root / "datasets" / "diverse_derm" / "Skin cancer ISIC The International Skin Imaging Collaboration"
    train_dir = base / "Train"
    test_dir = base / "Test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        return ([], [])

    malignant_classes = {
        "melanoma",
        "basal cell carcinoma",
        "squamous cell carcinoma",
        "actinic keratosis",
    }

    def _collect(split_dir: Path, source_tag: str) -> List[Sample]:
        out: List[Sample] = []
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            cls = class_dir.name.strip().lower()
            label = "malignant" if cls in malignant_classes else "benign"
            for p in class_dir.rglob("*"):
                if p.is_file() and is_image(p):
                    out.append(Sample(src=p, label=label, source=source_tag))
        return out

    train_pool = _collect(train_dir, "diversederm")
    test_set = _collect(test_dir, "diversederm")
    return (train_pool, test_set)


def write_metadata(out_dir: Path, meta: dict) -> None:
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare uniform lesion dataset (benign vs malignant) with train/val/test splits")
    ap.add_argument("--out", default="data/lesions_binary", help="Output directory to create")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (from train pool)")
    ap.add_argument("--test-frac", type=float, default=0.1, help="Test fraction (only applies to ISIC2019; DiverseDerm has fixed Test)")
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    ap.add_argument("--no-isic2019", action="store_true", help="Exclude ISIC 2019 from the output")
    ap.add_argument("--no-diversederm", action="store_true", help="Exclude DiverseDerm from the output")
    ap.add_argument("--clean", action="store_true", help="Delete output directory before writing")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out).resolve()

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    # Ensure split dirs exist (ImageFolder expects split/label)
    for split in ("train", "val", "test"):
        for label in ("benign", "malignant"):
            ensure_dir(out_dir / split / label)

    all_splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}

    sources_used: List[str] = []

    if not args.no_isic2019:
        isic_samples = load_isic2019_samples(repo_root)
        if isic_samples:
            sources_used.append("isic2019")
            isic_split = stratified_split(isic_samples, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
            for k in all_splits:
                all_splits[k].extend(isic_split[k])

    if not args.no_diversederm:
        dd_train, dd_test = load_diversederm_samples(repo_root)
        if dd_train or dd_test:
            sources_used.append("diversederm")
            # DiverseDerm provides Test; split Train into train/val
            dd_split = stratified_split(dd_train, val_frac=args.val_frac, test_frac=0.0, seed=args.seed)
            all_splits["train"].extend(dd_split["train"])
            all_splits["val"].extend(dd_split["val"])
            all_splits["test"].extend(dd_test)

    if not sources_used:
        raise SystemExit(
            "No supported datasets found. Expected ISIC2019 under isic_data/isic_2019 and/or DiverseDerm under datasets/diverse_derm."
        )

    # Materialize files
    counts: Dict[str, Dict[str, int]] = {s: {"benign": 0, "malignant": 0} for s in ("train", "val", "test")}
    per_source: Dict[str, Dict[str, Dict[str, int]]] = {}

    for split, samples in all_splits.items():
        for s in samples:
            # Prefix the filename with source to reduce collisions
            dst_name = f"{s.source}__{s.src.name}"
            dst = out_dir / split / s.label / dst_name
            link_or_copy(s.src, dst, args.mode)
            counts[split][s.label] += 1
            per_source.setdefault(s.source, {}).setdefault(split, {"benign": 0, "malignant": 0})
            per_source[s.source][split][s.label] += 1

    meta = {
        "out_dir": str(out_dir),
        "mode": args.mode,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "sources": sources_used,
        "counts": counts,
        "counts_by_source": per_source,
        "layout": "ImageFolder: <out>/{train,val,test}/{benign,malignant}/*.jpg",
    }
    write_metadata(out_dir, meta)

    print(json.dumps(meta["counts"], indent=2))
    print(f"\nWrote dataset to: {out_dir}")
    print(f"Metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
