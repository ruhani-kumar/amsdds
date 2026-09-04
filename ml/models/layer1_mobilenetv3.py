import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

N_CLASSES = 7


def build_model(n_classes: int = N_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    m = mobilenet_v3_large(weights=weights)
    m.classifier[3] = nn.Linear(1280, n_classes)
    return m


def load_checkpoint(path: str, device: str = "cpu"):
    """Returns (model_in_eval_mode, checkpoint_dict).

    The checkpoint carries `temperature` and `color_constancy` — both are
    load-bearing at inference. Prefer ml.inference.Layer1Model, which applies
    them for you.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state"])
    model.eval().to(device)
    return model, ckpt
