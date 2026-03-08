"""
Inference wrapper for the trained ConvNeXt-Large 20-class skin classifier.

Checkpoint format produced by jarvis-training/train.py:
  {
    "model_state_dict": ...,   # regular weights
    "ema_state_dict":   ...,   # EMA weights (preferred, better generalisation)
    "classes":          [...], # list of 20 class name strings
    "config": {backbone, drop_rate, drop_path_rate, ...}
  }

Public API (unchanged):
  inf    = get_inference()
  result = inf.predict_image(pil_image)
  # -> {"label": str, "probability": float, "p_malignant": float, "all_probs": dict}

Environment variables:
  LESION_MODEL_WEIGHTS   Path to best_model.pth  (default: checkpoints/best_model.pth)
  LESION_IMG_SIZE        Input resolution         (default: 384)
"""

from __future__ import annotations

import os
import logging
import threading
from pathlib import Path

import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image

log = logging.getLogger(__name__)

# Cancer class names — used to compute p_malignant
CANCER_CLASSES   = {"melanoma", "ak"}
DEFAULT_IMG_SIZE = 384
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]

# Post-hoc logit bias.
# melanoma +1.5: fine-tuned model's melanoma class was corrupted by BCC
# training data (845 BCC samples mapped to "melanoma"). This temporary bias
# ensures any residual melanoma signal (>0.5% raw) triggers hasCancerSignal.
# Remove once the model is retrained with the corrected PADUFES_MAP.
LOGIT_BIAS: dict[str, float] = {
    "melanoma": 1.5,
}

# Temperature for softmax: T=1.0 (neutral) — model's raw probabilities.
SOFTMAX_TEMPERATURE: float = 1.0

# --------------------------------------------------------------------------- #
#  Singleton cache                                                             #
# --------------------------------------------------------------------------- #
_model_cache: dict[str, "SkinInference"] = {}
_model_lock = threading.Lock()


def get_inference(weights: str | None = None, **kwargs) -> "SkinInference":
    """Return a cached SkinInference instance (singleton per weights path)."""
    key = weights or "__default__"
    if key not in _model_cache:
        with _model_lock:
            if key not in _model_cache:
                _model_cache[key] = SkinInference(weights=weights, **kwargs)
    return _model_cache[key]


# --------------------------------------------------------------------------- #
#  Inference class                                                             #
# --------------------------------------------------------------------------- #
class SkinInference:
    def __init__(
        self,
        weights: str | None = None,
        device: str | None = None,
        img_size: int | None = None,
        # legacy kwargs accepted but ignored
        backbone: str = "",
        **_kwargs,
    ):
        # ── Device ────────────────────────────────────────────────────────── #
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # ── Resolve checkpoint path ───────────────────────────────────────── #
        if not weights:
            weights = os.getenv("LESION_MODEL_WEIGHTS", "checkpoints/best_model.pth")
        weights_path = Path(weights)
        if not weights_path.is_absolute():
            weights_path = (Path(__file__).resolve().parents[1] / weights_path).resolve()

        # ── Load checkpoint ───────────────────────────────────────────────── #
        self.classes: list[str] = []
        self.model = None

        if weights_path.exists():
            self._load_checkpoint(weights_path)
        else:
            log.warning(
                "Checkpoint not found at %s — predictions will be empty. "
                "Set LESION_MODEL_WEIGHTS to the correct path.",
                weights_path,
            )

        # ── Cancer indices for p_malignant ────────────────────────────────── #
        self._cancer_indices: list[int] = [
            i for i, c in enumerate(self.classes) if c in CANCER_CLASSES
        ]

        # ── Logit bias vector (applied before softmax) ────────────────────── #
        bias = torch.zeros(len(self.classes))
        for cls, val in LOGIT_BIAS.items():
            if cls in self.classes:
                bias[self.classes.index(cls)] = val
        self._logit_bias = bias

        # ── Transform (must match training: 384 px center-crop) ───────────── #
        self.img_size = int(img_size or os.getenv("LESION_IMG_SIZE", DEFAULT_IMG_SIZE))
        resize_to = int(self.img_size * 1.143)
        self.transform = T.Compose([
            T.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        log.info(
            "SkinInference ready | device=%s | classes=%d | img_size=%d | "
            "cancer_indices=%s",
            self.device, len(self.classes), self.img_size, self._cancer_indices,
        )

    # ---------------------------------------------------------------------- #
    def _load_checkpoint(self, path: Path):
        try:
            import timm
        except ImportError:
            raise RuntimeError("timm is required: pip install timm>=1.0.0")

        log.info("Loading checkpoint: %s", path)
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)

        self.classes = ckpt.get("classes", [])
        if not self.classes:
            raise ValueError(f"Checkpoint {path} does not contain a 'classes' list.")

        # IMPORTANT: SkinDataset in train.py uses sorted() (alphabetical) for
        # class_to_idx, so model output neurons are in alphabetical order.
        # The checkpoint's 'classes' key may reflect class_info.json ordering
        # (non-alphabetical). Sort here to match training label indices.
        self.classes = sorted(self.classes)

        config   = ckpt.get("config", {})
        backbone = config.get("backbone", "convnext_large.fb_in22k_ft_in1k_384")
        drop_r   = config.get("drop_rate", 0.4)
        drop_p   = config.get("drop_path_rate", 0.3)

        candidates = [
            backbone,
            "convnext_large.fb_in22k_ft_in1k_384",
            "convnext_large.fb_in22k_ft_in1k",
            "convnext_large",
        ]
        model = None
        for name in dict.fromkeys(candidates):
            try:
                model = timm.create_model(
                    name, pretrained=False,
                    num_classes=len(self.classes),
                    drop_rate=drop_r,
                    drop_path_rate=drop_p,
                )
                log.info("  Backbone: %s", name)
                break
            except Exception:
                continue

        if model is None:
            raise RuntimeError(f"Could not instantiate backbone for checkpoint: {path}")

        if "ema_state_dict" in ckpt:
            model.load_state_dict(ckpt["ema_state_dict"])
            log.info("  Loaded EMA weights")
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            log.info("  Loaded model weights")
        else:
            raise ValueError(f"Checkpoint {path} contains no model weights.")

        self.model = model.to(self.device)
        self.model.eval()

    # ---------------------------------------------------------------------- #
    def _normalize_input(self, image: Image.Image) -> Image.Image:
        """
        Minimal preprocessing: only clip genuinely overexposed photos (mean > 220).
        Histogram equalization was removed — it destroys pigmentation intensity
        information (dark patches on hyperpigmentation, melanoma, nevus) which is
        the primary diagnostic signal for most skin conditions.
        """
        import PIL.ImageEnhance as _IE
        from PIL import ImageStat as _IS

        rgb = image.convert("RGB")

        # Only intervene when the image is truly blown out (> 220/255 mean).
        # A mild contrast boost (1.1) slightly sharpens feature boundaries
        # without altering the relative intensity distribution.
        mean_brightness = sum(_IS.Stat(rgb).mean) / 3.0
        if mean_brightness > 220:
            rgb = _IE.Contrast(rgb).enhance(1.1)

        return rgb

    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> dict:
        """
        Run inference on a PIL image.

        Returns:
            {
                "label":       str,    # predicted class name
                "probability": float,  # confidence for predicted class (0–1)
                "p_malignant": float,  # sum of cancer-class probabilities (0–1)
                "all_probs":   dict,   # {class_name: prob} for all 20 classes
            }
        """
        if self.model is None or not self.classes:
            return {"label": "unknown", "probability": 0.0,
                    "p_malignant": 0.0, "all_probs": {},
                    "entropy": 1.0, "is_low_confidence": True}

        # Basic Test-Time Augmentation (TTA)
        # We will create 3 views of the same image: Original, Horizontal Flip, Vertical Flip
        
        image = self._normalize_input(image)
        orig_tensor = self.transform(image.convert("RGB")).to(self.device)
        
        # Create augmented versions
        hf_tensor = T.functional.hflip(orig_tensor)
        vf_tensor = T.functional.vflip(orig_tensor)
        
        # Stack into a mini-batch of size 3
        x = torch.stack([orig_tensor, hf_tensor, vf_tensor])

        use_amp = self.device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits_batch = self.model(x)

        # Average the logits across the 3 TTA views
        logits = logits_batch.mean(dim=0).float().cpu()
        
        # Apply logit bias + temperature scaling
        logits = logits + self._logit_bias
        probs = F.softmax(logits / SOFTMAX_TEMPERATURE, dim=0)

        pred_idx    = int(probs.argmax())
        label       = self.classes[pred_idx]
        probability = float(probs[pred_idx])
        p_malignant = float(probs[self._cancer_indices].sum()) if self._cancer_indices else 0.0
        all_probs   = {cls: float(probs[i]) for i, cls in enumerate(self.classes)}

        # Entropy-based uncertainty (normalised 0→1; > 0.85 = confused)
        probs_list  = probs.tolist()
        n_cls       = len(probs_list)
        entropy_raw = -sum(p * math.log(p + 1e-12) for p in probs_list)
        max_entropy = math.log(n_cls) if n_cls > 1 else 1.0
        entropy_norm = entropy_raw / max_entropy  # 0 = certain, 1 = uniform
        is_low_confidence = entropy_norm > 0.85 or probability < 0.20

        return {
            "label":              label,
            "probability":        probability,
            "p_malignant":        p_malignant,
            "all_probs":          all_probs,
            "entropy":            round(entropy_norm, 4),
            "is_low_confidence":  is_low_confidence,
        }

    def predict_path(self, img_path: str) -> dict:
        """Convenience wrapper: load image from disk and predict."""
        return self.predict_image(Image.open(img_path))
