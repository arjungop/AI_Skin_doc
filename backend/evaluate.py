"""
Quick evaluation / smoke test for the skin classifier.

Uses the same SkinInference wrapper that the API uses, so it tests
the real prediction pipeline end-to-end.

Usage:
    python -m backend.evaluate                          # uses LESION_MODEL_WEIGHTS from .env
    python -m backend.evaluate path/to/image.jpg        # predict a single image
"""

import sys
import os
from pathlib import Path
from PIL import Image

# Ensure .env is loaded so LESION_MODEL_WEIGHTS is visible
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from backend.inference import get_inference


def evaluate_single(image_path: str):
    """Run the model on a single image and print results."""
    infer = get_inference()
    img = Image.open(image_path).convert("RGB")
    result = infer.predict_image(img)
    print(f"Image:       {image_path}")
    print(f"Prediction:  {result['label']}")
    print(f"Confidence:  {result['probability']:.4f}")
    print(f"P(malignant):{result['p_malignant']:.4f}")
    print(f"Top-5:")
    sorted_probs = sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True)
    for name, prob in sorted_probs[:5]:
        print(f"  {name:25s} {prob:.4f}")
    return result


def smoke_test():
    """Verify the model loads and can run inference on a dummy image."""
    infer = get_inference()
    if infer.model is None:
        print("ERROR: Model not loaded. Check LESION_MODEL_WEIGHTS.")
        sys.exit(1)
    print(f"Model loaded: {len(infer.classes)} classes, device={infer.device}")
    # Dummy inference
    dummy = Image.new("RGB", (384, 384), color=(128, 100, 90))
    result = infer.predict_image(dummy)
    print(f"Smoke test:  label={result['label']}, prob={result['probability']:.4f}")
    print("OK — model is functional.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluate_single(sys.argv[1])
    else:
        smoke_test()