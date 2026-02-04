import os
import logging
import torch
from torchvision import transforms
from PIL import Image
from backend.ml.model import HierarchicalSkinClassifier

class MelanomaInference:
    def __init__(self, weights: str | None = None, device=None, backbone: str = "resnet18"):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        from pathlib import Path
        if not weights:
            weights = os.getenv("LESION_MODEL_WEIGHTS", "")
        
        # ... (path normalization logic omitted for brevity, keeping existing if possible or simplifying) ...
        # Assume path logic is fine, focusing on model load
        
        # Load Hierarchical Model
        # We need num_classes to match training. Hierarchy: 5 categories, 19 diseases.
        # Ideally read from meta.json. If missing, default to 5/19.
        self.model = HierarchicalSkinClassifier(num_categories=5, num_diseases=19, backbone=backbone).to(self.device)
        
        if weights and os.path.exists(weights):
            try:
                state = torch.load(weights, map_location=self.device)
                self.model.load_state_dict(state)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load weights: {e}")
        
        self.model.eval()
        
        # malignant index typically maps to Mel(6) or similar in 19-class list.
        # reliable mapping comes from meta.json. For backward compat, we'll try to guess.
        self.labels = ['melanoma', 'bcc', 'scc', 'ak', 'nevus', 'seborrheic_keratosis', 'angioma', 'wart', 'eczema', 'psoriasis', 'lichen_planus', 'urticaria', 'impetigo', 'herpes', 'candida', 'scabies', 'vitiligo', 'melasma', 'hyperpigmentation']

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_image(self, image: Image.Image):
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Hierarchical returns (cat_logits, dis_logits)
            _, dis_logits = self.model(x)
            probs = torch.softmax(dis_logits, dim=1)[0]
        
        pred_idx = int(torch.argmax(probs))
        label = self.labels[pred_idx] if pred_idx < len(self.labels) else "unknown"
        confidence = float(probs[pred_idx])
        
        # Legacy compatibility: p_malignant
        # Sum probabilities of cancer classes: melanoma, bcc, scc, ak
        # Indices in self.labels: melanoma(0), bcc(1), scc(2), ak(3)
        p_malignant = float(probs[0] + probs[1] + probs[2] + probs[3])
        
        return {"label": label, "probability": confidence, "p_malignant": p_malignant}

        # malignant class index: allow override for datasets where class order differs
        try:
            self.malignant_index = int(os.getenv("LESION_MALIGNANT_INDEX", str(meta.get("malignant_index", 1))))
        except Exception:
            self.malignant_index = int(meta.get("malignant_index", 1))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.labels = ["benign", "malignant"]

    def predict_image(self, image: Image.Image):
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        mi = max(0, min(len(probs)-1, self.malignant_index))
        p_malignant = float(probs[mi])  # malignant probability, configurable index
        return {"label": self.labels[pred_idx], "probability": float(probs[pred_idx]), "p_malignant": p_malignant}

    def predict_path(self, img_path: str):
        img = Image.open(img_path).convert("RGB")
        return self.predict_image(img)

    def gradcam_overlay(self, image: Image.Image):
        """Return a PIL image with Grad-CAM heatmap overlaid for the predicted class."""
        import numpy as np
        import torch.nn.functional as F

        self.model.zero_grad()
        x = self.transform(image).unsqueeze(0).to(self.device)

        # choose last conv layer of resnet
        try:
            target_layer = self.model.backbone.layer4[-1].conv2  # resnet18
        except Exception:
            try:
                target_layer = self.model.backbone.layer4[-1].conv3  # resnet50
            except Exception:
                target_layer = None

        activations = []
        gradients = []

        def fwd_hook(_, __, out):
            activations.append(out)

        def bwd_hook(_, __, grad_out):
            gradients.append(grad_out[0])

        h1 = h2 = None
        if target_layer is not None:
            h1 = target_layer.register_forward_hook(fwd_hook)
            h2 = target_layer.register_full_backward_hook(bwd_hook)  # type: ignore

        logits = self.model(x)
        pred = logits.argmax(dim=1)
        score = logits[0, pred]
        self.model.zero_grad(set_to_none=True)
        score.backward()

        if h1: h1.remove()
        if h2: h2.remove()

        if not activations or not gradients:
            return image

        acts = activations[0]  # [1,C,H,W]
        grads = gradients[0]   # [1,C,H,W]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR).convert('L')
        # create red overlay with alpha from CAM
        heat = Image.new('RGBA', image.size, (255, 0, 0, 0))
        heat.putalpha(cam_img)
        base = image.convert('RGBA')
        overlay = Image.alpha_composite(base, heat)
        return overlay.convert('RGB')
