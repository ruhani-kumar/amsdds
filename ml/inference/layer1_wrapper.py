import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ml.models import load_checkpoint
from ml.preprocessing import shades_of_gray, eval_tf


class Layer1Model:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.device = device
        self.model, ckpt = load_checkpoint(ckpt_path, device)
        self.classes = ckpt["classes"]
        self.temperature = float(ckpt["temperature"])
        self.color_constancy = bool(ckpt.get("color_constancy", True))
        self.log_n = float(np.log(len(self.classes)))

    def _tensor(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        if self.color_constancy:
            img = Image.fromarray(shades_of_gray(np.array(img)))
        return eval_tf(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, img: Image.Image) -> np.ndarray:
        """Calibrated class probabilities, shape (n_classes,)."""
        logits = self.model(self._tensor(img))
        return F.softmax(logits / self.temperature, dim=1)[0].cpu().numpy()

    def confidence(self, probs: np.ndarray):
        i = int(probs.argmax())
        return self.classes[i], float(probs[i])

    def entropy(self, probs: np.ndarray) -> float:
        """Shannon entropy normalised by log(n_classes) -> [0, 1], so the
        threshold in configs/thresholds.yaml is class-count independent."""
        return float(-(probs * np.log(probs + 1e-12)).sum() / self.log_n)

    def predict_full(self, img: Image.Image):
        """Returns (label, confidence, normalised_entropy, probs)."""
        p = self.predict(img)
        label, conf = self.confidence(p)
        return label, conf, self.entropy(p), p
