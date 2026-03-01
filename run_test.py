#!/usr/bin/env python3
"""Quick inference test — run on sample images from the test dataset."""
import sys, math
from pathlib import Path
from PIL import Image, ImageStat

sys.path.insert(0, str(Path(__file__).parent))
from backend.inference import get_inference

weights = "backend/ml/weights/best_model.pth"
inf = get_inference(weights=weights)
print(f"Model loaded | classes: {len(inf.classes)} | device: {inf.device}")
print()

# If a path is passed as argument, run on that image
if len(sys.argv) > 1:
    img_path = sys.argv[1]
    img = Image.open(img_path)
    r = inf.predict_image(img)
    stat = ImageStat.Stat(img.convert("RGB"))
    print(f"Image      : {img_path}")
    print(f"Size       : {img.size[0]}x{img.size[1]}  brightness: {stat.mean[0]:.0f}/255")
    print(f"Prediction : {r['label'].upper()}")
    print(f"Confidence : {r['probability']*100:.1f}%")
    print(f"p_malignant: {r['p_malignant']*100:.1f}%")
    print()
    print("All probabilities:")
    for cls, prob in sorted(r["all_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:25s} {prob*100:5.1f}%  {bar}")
    # Entropy — high entropy = model is uncertain/confused (out-of-distribution image)
    probs = list(r["all_probs"].values())
    entropy = -sum(p * math.log(p + 1e-9) for p in probs)
    max_entropy = math.log(20)
    ood = entropy > max_entropy * 0.5
    print(f"\nEntropy: {entropy:.2f}/{max_entropy:.2f}  →  {'⚠️  Model is CONFUSED — image may be out-of-distribution (not a clinical close-up)' if ood else '✅ Model is confident'}")
    sys.exit(0)

# Otherwise run on one sample from each class in the test set
test_data = Path("jarvis-training/data/test")
if not test_data.exists():
    print("No test data found. Pass an image path as argument:")
    print("  python run_test.py /path/to/your/image.png")
    sys.exit(1)

import random
random.seed(42)

classes_to_test = ["melanoma", "eczema", "nevus", "fungal", "psoriasis",
                   "impetigo", "scabies", "drug_eruption", "lupus", "angioma"]

total_correct = 0
total_tested = 0
for cls in classes_to_test:
    cls_dir = test_data / cls
    if not cls_dir.exists():
        continue
    imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
    if not imgs:
        continue
    # Random sample of 20 for a representative estimate
    sample = random.sample(imgs, min(20, len(imgs)))
    correct = 0
    for p in sample:
        img = Image.open(p)
        r = inf.predict_image(img)
        if r["label"] == cls:
            correct += 1
    total_correct += correct
    total_tested += len(sample)
    pct = correct / len(sample) * 100
    bar = "█" * correct + "░" * (len(sample) - correct)
    print(f"{cls:25s} {correct:2d}/{len(sample)} {pct:5.0f}%  [{bar}]")

print()
print(f"Mini-test accuracy: {total_correct}/{total_tested} = {total_correct/total_tested*100:.1f}%")
print("(Full official result:  96.58% over 25,904 images from eval_results.json)")

