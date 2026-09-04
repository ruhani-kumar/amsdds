"""Frozen feature extractors, behind one interface.

Everything downstream (extraction, head training, calibration) consumes
`Backbone` and nothing else. Swapping PanDerm for DINOv2 is a config string,
not a refactor — which matters because PanDerm weights are distributed via a
Google Drive link that may gate you.

Contract:
    bb = build_backbone("panderm_base", device="cuda")
    bb.dim          -> int, feature dimension
    bb.transform    -> torchvision transform, PIL -> [3,H,W]
    bb.encode(x)    -> [N, dim] float32 on CPU, no grad

Colour constancy is NOT applied here. It happens upstream on the PIL image,
once, exactly as the v1 pipeline does it — see src/data.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torchvision.transforms as T

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _eval_transform(size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # resize short side by 1.14x then centre crop — matches the v1 convention
    # (324 -> 299) so cropping behaviour is comparable across experiments.
    return T.Compose([
        T.Resize(int(round(size * 1.14))),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


@dataclass
class Backbone:
    name: str
    model: nn.Module
    transform: Callable
    dim: int
    img_size: int
    device: str = "cuda"
    _amp: bool = field(default=True, repr=False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[N,3,H,W] -> [N,dim] on CPU. Handles both timm-style `forward_features`
        + pooling and plain `forward` returning a pooled vector."""
        x = x.to(self.device, non_blocking=True)
        use_amp = self._amp and self.device == "cuda"
        with torch.autocast("cuda", torch.float16, enabled=use_amp):
            if hasattr(self.model, "forward_features"):
                f = self.model.forward_features(x)
                f = _pool(f)
            else:
                f = self.model(x)
                if f.ndim > 2:
                    f = _pool(f)
        return f.float().cpu()


def _pool(f: torch.Tensor) -> torch.Tensor:
    """Reduce whatever the backbone emitted to [N, D].
    ViT: [N, tokens, D] -> mean over patch tokens (drop CLS at index 0 if the
    token count looks like grid+1). CNN: [N, D, H, W] -> global average pool."""
    if f.ndim == 3:
        n_tok = f.shape[1]
        for extra in (1, 5, 0):                 # CLS, CLS+4 registers, none
            g = int(round((n_tok - extra) ** 0.5))
            if g * g == n_tok - extra:
                return f[:, extra:].mean(1)
        return f.mean(1)
    if f.ndim == 4:
        return f.mean((2, 3))
    return f


# ----------------------------------------------------------------- PanDerm
def _build_panderm(ckpt_path: str, device: str, img_size: int = 224) -> Backbone:
    """PanDerm_Base is a BEiT/CAEv2-style ViT-B/16, NOT a plain timm ViT.

    Confirmed from the released checkpoint: 186 tensors = 12 blocks x 15
    (gamma_1, gamma_2, split q_bias/v_bias, fused qkv.weight, 2 norms, 2 mlp
    layers) + cls_token + pos_embed + patch_embed.proj.{weight,bias} + final
    norm.{weight,bias}. timm's `Beit` with absolute position embeddings and
    relative-position-bias OFF reproduces that layout exactly.

    Loading this into `vit_base_patch16_224` would silently drop LayerScale and
    the attention biases — well-shaped features, quietly wrong. Hence the
    strict-ish check below: anything missing means STOP.
    """
    from timm.models.beit import Beit

    m = Beit(img_size=img_size, patch_size=16, embed_dim=768, depth=12,
             num_heads=12, init_values=0.1, use_abs_pos_emb=True,
             use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
             global_pool='token', num_classes=0)

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "module", "teacher", "student"):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
            break
    sd = {}
    for k, v in obj.items():
        for pre in ("module.", "backbone.", "encoder.", "visual."):
            if k.startswith(pre):
                k = k[len(pre):]
        sd[k] = v

    bad = m.load_state_dict(sd, strict=False)
    missing = [k for k in bad.missing_keys if not k.startswith(("head", "fc_norm"))]
    unexpected = list(bad.unexpected_keys)
    print(f"[panderm] loaded {len(sd)} tensors | missing={len(missing)} "
          f"unexpected={len(unexpected)}")
    if missing or unexpected:
        print(f"[panderm] missing:    {missing[:6]}")
        print(f"[panderm] unexpected: {unexpected[:6]}")
        raise RuntimeError(
            "PanDerm state dict did not map cleanly. Do NOT proceed — the "
            "features would be plausible and wrong. Fall back to dinov2_base."
        )
    print("[panderm] clean load, all weights mapped")
    return Backbone("panderm_base", m.eval().to(device),
                    _eval_transform(img_size), m.num_features, img_size, device)


# ------------------------------------------------------------ timm fallbacks
def _build_timm(model_id: str, device: str, img_size: int, name: str) -> Backbone:
    """ViTs need img_size at creation so timm interpolates the positional
    embedding — dinov2 defaults to 518px and will otherwise throw on 224px
    input. CNNs reject the kwarg, so fall back to their native size."""
    import timm

    try:
        m = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=img_size)
    except (TypeError, RuntimeError) as e:
        print(f"[timm] img_size kwarg rejected ({type(e).__name__}); using native size")
        m = timm.create_model(model_id, pretrained=True, num_classes=0)
        img_size = m.default_cfg.get("input_size", (3, img_size, img_size))[-1]

    cfg = m.default_cfg
    tf = _eval_transform(img_size, cfg.get("mean", IMAGENET_MEAN), cfg.get("std", IMAGENET_STD))
    return Backbone(name, m.eval().to(device), tf, m.num_features, img_size, device)


_TIMM_FALLBACKS = {
    # id, img_size — all pretrained, all download without a gate
    "dinov2_base":   ("vit_base_patch14_dinov2.lvd142m", 224),   # 224/14 = 16x16 grid
    "dinov2_small":  ("vit_small_patch14_dinov2.lvd142m", 224),  # faster on T4
    "vit_base":      ("vit_base_patch16_224.augreg2_in21k_ft_in1k", 224),
    "convnext_base": ("convnext_base.fb_in22k_ft_in1k", 288),
    "effnetv2_m":    ("tf_efficientnetv2_m.in21k_ft_in1k", 384),
}


def build_backbone(name: str, device: str = "cuda", panderm_ckpt: str | None = None) -> Backbone:
    if name.startswith("panderm"):
        if not panderm_ckpt or not os.path.isfile(panderm_ckpt):
            raise FileNotFoundError(
                f"PanDerm checkpoint not found at {panderm_ckpt!r}. "
                f"Either fetch it, or fall back: build_backbone('dinov2_base')."
            )
        return _build_panderm(panderm_ckpt, device)
    if name in _TIMM_FALLBACKS:
        model_id, size = _TIMM_FALLBACKS[name]
        return _build_timm(model_id, device, size, name)
    raise KeyError(f"unknown backbone {name!r}; have {['panderm_base', *_TIMM_FALLBACKS]}")
