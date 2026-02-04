#!/usr/bin/env python3
"""
Hierarchical Inference Script
Outputs: Category (Level 1) + Disease (Level 2)
"""
import argparse
import sys
import pathlib
import json
import torch
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from backend.ml.model import HierarchicalSkinClassifier


def load_metadata(model_path: str):
    """Load model metadata (hierarchy, classes)"""
    meta_path = model_path.replace('.pth', '.json')
    try:
        with open(meta_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Metadata not found at {meta_path}")
        return None


def predict(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load metadata
    meta = load_metadata(args.model_path)
    if meta is None:
        print("Error: Model metadata required for hierarchical inference")
        return
    
    hierarchy = meta["hierarchy"]
    folder_classes = meta["folder_classes"]  # Disease names in order
    categories = hierarchy["categories"]
    
    num_categories = len(categories)
    num_diseases = len(folder_classes)

    # Load Model
    model = HierarchicalSkinClassifier(
        num_categories=num_categories,
        num_diseases=num_diseases,
        backbone=meta.get("backbone", "resnet18"),
        pretrained=False
    ).to(device)
    
    try:
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {args.model_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess image
    img = Image.open(args.image).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        cat_logits, dis_logits = model(input_tensor)
        cat_probs = torch.softmax(cat_logits, dim=1)[0]
        dis_probs = torch.softmax(dis_logits, dim=1)[0]
    
    # Get predictions
    cat_idx = torch.argmax(cat_probs).item()
    dis_idx = torch.argmax(dis_probs).item()
    
    predicted_category = categories[cat_idx]
    predicted_disease = folder_classes[dis_idx]
    cat_confidence = float(cat_probs[cat_idx])
    dis_confidence = float(dis_probs[dis_idx])

    # Output
    print("\n" + "=" * 50)
    print("HIERARCHICAL CLASSIFICATION RESULTS")
    print("=" * 50)
    
    print(f"\n🏥 CATEGORY (Level 1): {predicted_category.upper()}")
    print(f"   Confidence: {cat_confidence:.1%}")
    
    print(f"\n🔬 DISEASE (Level 2): {predicted_disease.replace('_', ' ').title()}")
    print(f"   Confidence: {dis_confidence:.1%}")
    
    print("\n--- Category Probabilities ---")
    for i, cat in enumerate(categories):
        bar = "█" * int(cat_probs[i] * 20)
        print(f"  {cat:15} {cat_probs[i]:.1%} {bar}")
    
    print("\n--- Top 5 Disease Predictions ---")
    top5_idx = torch.topk(dis_probs, 5).indices
    for idx in top5_idx:
        disease = folder_classes[idx]
        prob = dis_probs[idx]
        bar = "█" * int(prob * 20)
        print(f"  {disease:25} {prob:.1%} {bar}")
    
    print("=" * 50)
    
    # Return as JSON for API usage
    result = {
        "category": predicted_category,
        "category_confidence": cat_confidence,
        "disease": predicted_disease,
        "disease_confidence": dis_confidence,
        "all_category_probs": {cat: float(cat_probs[i]) for i, cat in enumerate(categories)},
        "top5_diseases": [
            {"disease": folder_classes[idx], "confidence": float(dis_probs[idx])} 
            for idx in top5_idx
        ]
    }
    
    if args.json:
        print("\n--- JSON Output ---")
        print(json.dumps(result, indent=2))
    
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hierarchical Skin Disease Inference")
    p.add_argument('--image', required=True, help='Path to image file')
    p.add_argument('--model-path', default='backend/ml/weights/hierarchical_skin.pth', 
                   help='Path to .pth checkpoint')
    p.add_argument('--json', action='store_true', help='Output JSON format')
    
    args = p.parse_args()
    predict(args)
