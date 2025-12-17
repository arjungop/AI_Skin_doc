import os
import logging
import torch
from torchvision import transforms
from PIL import Image
from backend.ml.model import SkinClassifier


class MelanomaInference:
    def __init__(self, weights: str | None = None, device=None, backbone: str = "resnet18"):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        from pathlib import Path
        if not weights:
            weights = os.getenv("LESION_MODEL_WEIGHTS", "")
        # Auto-discover a weights file if not provided or missing
        proj_root = Path(__file__).resolve().parents[1]
        def _normalize(p: str) -> str:
            p = os.path.expanduser(str(p))
            if not os.path.isabs(p):
                return str((proj_root / p).resolve())
            return p
        weights = _normalize(weights) if weights else ""
        if not weights or not os.path.exists(weights):
            # pick the most recent .pth under backend/ml/weights
            cand_dir = proj_root / "backend" / "ml" / "weights"
            cands = sorted(cand_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cands:
                weights = str(cands[0].resolve())
            else:
                # fallback to default path (may not exist)
                weights = str((proj_root / "backend/ml/weights/skin_resnet18.pth").resolve())

        # Read meta if available
        meta = {}
        try:
            import json
            meta_path = os.path.splitext(weights)[0] + ".json"
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f) or {}
            meta["weights_path"] = weights
        except Exception as e:
            logging.getLogger(__name__).warning("Could not read model meta: %s", e)

        # Decide backbone: env > meta > arg default
        backbone = os.getenv("LESION_MODEL_BACKBONE", meta.get("backbone", backbone))

        # Build and try to load; if shape mismatch, try alternate resnet
        self.model = SkinClassifier(num_classes=2, backbone=backbone, pretrained=False).to(self.device)
        try:
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
        except Exception as e:
            logging.getLogger(__name__).warning("load_state_dict failed for %s, backbone=%s: %s", weights, backbone, e)
            alt = "resnet50" if backbone == "resnet18" else "resnet18"
            self.model = SkinClassifier(num_classes=2, backbone=alt, pretrained=False).to(self.device)
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
            logging.getLogger(__name__).info("Loaded weights with alternate backbone %s", alt)
        self.model.eval()

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
