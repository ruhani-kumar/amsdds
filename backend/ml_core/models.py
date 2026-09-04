import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large


# ------------------------------------------------ Layer 1: multimodal MobileNet
class MultiModalMobileNetV3(nn.Module):
    """Matches the training notebook exactly: image branch 1280-d
    (classifier[3]=Identity) + metadata 19->64->32 + fusion 1312->128->7."""
    def __init__(self, metadata_dim, num_classes):
        super().__init__()
        self.image_model = mobilenet_v3_large()
        self.image_model.classifier[3] = nn.Identity()
        self.metadata_model = nn.Sequential(nn.Linear(metadata_dim, 64), nn.ReLU(),
                                            nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(1280 + 32, 128), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, x, meta):
        return self.classifier(torch.cat([self.image_model(x), self.metadata_model(meta)], 1))


class Layer1:
    """Returns (logits, 1280-d image features). Builds the 19-dim metadata
    vector from optional user metadata using the encoders stored IN the
    checkpoint; missing fields fall back to neutral (mean age / unknown)."""
    def __init__(self, path, device):
        c = torch.load(path, map_location="cpu", weights_only=False)
        self.classes = list(c["classes"])
        self.temperature = float(c["temperature"])            # 0.959
        self.sex_cat = list(c["sex_categories"])              # female, male, unknown
        self.loc_cat = list(c["location_categories"])         # 15 sites incl unknown
        self.age_mu, self.age_sd = float(c["age_mean"]), float(c["age_std"])
        self.meta_dim = int(c["metadata_dim"])                # 19
        m = MultiModalMobileNetV3(self.meta_dim, len(self.classes))
        m.load_state_dict(c["model_state"])
        self.m = m.eval().to(device)
        self.device = device

    def encode_meta(self, age=None, sex=None, localization=None):
        a = self.age_mu if age in (None, "") else float(age)
        s = str(sex).strip().lower() if sex else "unknown"
        l = str(localization).strip().lower() if localization else "unknown"
        if s not in self.sex_cat: s = "unknown"
        if l not in self.loc_cat: l = "unknown"
        v = [(a - self.age_mu) / self.age_sd]
        v += [float(s == c) for c in self.sex_cat]      # 3
        v += [float(l == c) for c in self.loc_cat]      # 15  -> 19 total
        assert len(v) == self.meta_dim, f"meta dim {len(v)} != {self.meta_dim}"
        return torch.tensor([v], dtype=torch.float32)

    @torch.no_grad()
    def run(self, x, meta_vec):
        x, meta_vec = x.to(self.device), meta_vec.to(self.device)
        with torch.autocast("cuda", torch.float16, enabled=self.device == "cuda"):
            feat = self.m.image_model(x).float()
            logits = self.m.classifier(torch.cat([feat, self.m.metadata_model(meta_vec)], 1)).float()
        return logits.cpu(), feat.cpu()


# ------------------------------------------------ Layer 2: encoder + GOA heads
class RTDetrEncoder(nn.Module):
    def __init__(self, hf_name, state):
        super().__init__()
        from transformers import RTDetrForObjectDetection
        base = RTDetrForObjectDetection.from_pretrained(hf_name).model
        self.backbone, self.input_proj, self.encoder = base.backbone, base.encoder_input_proj, base.encoder
        # load fine-tuned weights over the pretrained skeleton (head keys ignored)
        missing = self.load_state_dict(state, strict=False).missing_keys
        assert not missing, f"L2 encoder mismatch: {missing[:5]}"
        self.eval()
    @torch.no_grad()
    def pooled(self, x):
        mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=x.device)
        feats = self.backbone(x, mask)
        srcs = [self.input_proj[i](f) for i, (f, _) in enumerate(feats)]
        enc = self.encoder(inputs_embeds=srcs)[0]
        return torch.cat([f.mean((2, 3)) for f in enc], 1)     # 768-d


def _build_head(hparams, d_in, n_classes):
    hidden = int(hparams["hidden"])
    if hidden < 8:
        return nn.Sequential(nn.Dropout(hparams["drop"]), nn.Linear(d_in, n_classes))
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                         nn.Dropout(hparams["drop"]), nn.Linear(hidden, n_classes))


class Layer2:
    """Frozen encoder + one or more GOA heads, selectable per request."""
    def __init__(self, enc_path, hf_name, head_paths, default_head, n_classes, device):
        c = torch.load(enc_path, map_location="cpu", weights_only=False)
        self.enc = RTDetrEncoder(c.get("model_name", hf_name), c["state"]).to(device)
        for p in self.enc.parameters(): p.requires_grad_(False)
        self.device, self.heads, self.default = device, {}, default_head
        for name, path in head_paths.items():
            if not os.path.exists(path):
                print(f"[models] L2 head '{name}' missing at {path} — skipped"); continue
            g = torch.load(path, map_location="cpu", weights_only=False)
            head = _build_head(g["hparams"], g["feat_mu"].shape[0], n_classes)
            head.load_state_dict(g["head"])
            self.heads[name] = {
                "head": head.eval().to(device),
                "mu": torch.tensor(g["feat_mu"], dtype=torch.float32, device=device),
                "sd": torch.tensor(g["feat_sd"], dtype=torch.float32, device=device),
                "T": float(g["temperature"]),
            }
        assert self.default in self.heads, f"default head '{self.default}' not loaded"
        print(f"[models] L2 heads loaded: {list(self.heads)} (default {self.default})")

    @torch.no_grad()
    def run(self, x, head_name=None):
        h = self.heads.get(head_name or self.default) or self.heads[self.default]
        name = head_name if head_name in self.heads else self.default
        with torch.autocast("cuda", torch.float16, enabled=self.device == "cuda"):
            f = self.enc.pooled(x.to(self.device)).float()
        logits = h["head"]((f - h["mu"]) / h["sd"]).float().cpu()
        return logits, h["T"], name


# ------------------------------------------------ Layer 2: PanDerm (current default)
class PanDermLayer2:
    """Single, self-contained Layer 2 classifier: PanDerm_Base (BEiT ViT-B/16),
    fully fine-tuned end-to-end on HAM10000 (backbone + head + temperature all
    bundled in one checkpoint from amsdds_v2/outputs/layer2_panderm_ft_v1.pt).

    Kept API-compatible with the RT-DETR ``Layer2.run(x, head_name)`` contract
    (same ``(logits, temperature, head_name_used)`` return) so ``Engine`` needs
    no changes. ``head_name`` is accepted but ignored: unlike RT-DETR's shared
    frozen encoder + selectable ham/hampad heads, PanDerm's two candidate heads
    (amsdds_v2/outputs/layer2_panderm_probe.pt and layer2_panderm_pad.pt) were
    trained on features from TWO DIFFERENT backbones (frozen pretrained
    PanDerm_Base vs. the HAM-fine-tuned backbone — see
    amsdds_v2/notebooks/panderm.ipynb, CELL P4's own note), so they cannot
    share one loaded encoder the way the RT-DETR heads do. Rather than load two
    full ViT-B/16 backbones to fake a head switch, Layer 2 is a single model.
    """
    def __init__(self, ckpt_path, device):
        from timm.models.beit import Beit
        c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        a = c["arch"]
        m = Beit(img_size=a["img_size"], patch_size=a["patch_size"], embed_dim=a["embed_dim"],
                 depth=a["depth"], num_heads=a["num_heads"], init_values=a["init_values"],
                 use_abs_pos_emb=a["use_abs_pos_emb"], use_rel_pos_bias=a["use_rel_pos_bias"],
                 use_shared_rel_pos_bias=a["use_shared_rel_pos_bias"], global_pool=a["global_pool"],
                 drop_path_rate=a["drop_path_rate"], num_classes=a["num_classes"])
        m.load_state_dict(c["model_state"], strict=True)
        self.m = m.eval().to(device)
        self.device = device
        self.classes = list(c["classes"])
        self.temperature = float(c["temperature"])
        self.default = "panderm"
        self.heads = {"panderm": {"T": self.temperature}}   # shape expected by Engine.model_info()

    @torch.no_grad()
    def run(self, x, head_name=None):
        with torch.autocast("cuda", torch.float16, enabled=self.device == "cuda"):
            logits = self.m(x.to(self.device)).float()
        return logits.cpu(), self.temperature, self.default


# ------------------------------------------------ OOD
class MahalanobisOOD:
    def __init__(self, path, threshold=None):
        z = np.load(path, allow_pickle=True)
        self.mus = z["mus"].astype(np.float32)
        self.prec = z["prec"].astype(np.float32)
        self.threshold = float(threshold if threshold is not None else z["unk_thr"])
    def score(self, feat):
        f = feat.numpy().astype(np.float32)
        d = f[:, None, :] - self.mus[None]
        return float(np.einsum("nci,ij,ncj->nc", d, self.prec, d).min())


def load_all(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wd = os.environ.get("AMSDDS_WEIGHTS", cfg["weights_dir"])
    l1 = Layer1(os.path.join(wd, cfg["layer1"]["file"]), device)

    # --- Layer 2 backbone select: `panderm` (current) or `rtdetr` (fallback,
    #     kept for reference/comparison — see backend/ml_core/README.md).
    backbone = cfg["layer2"]["backbone"]
    l2cfg = cfg["layer2"][backbone]
    if backbone == "panderm":
        l2 = PanDermLayer2(os.path.join(wd, l2cfg["file"]), device)
    elif backbone == "rtdetr":
        heads = {k: os.path.join(wd, v) for k, v in l2cfg["heads"].items()}
        l2 = Layer2(os.path.join(wd, l2cfg["file"]), l2cfg["hf_name"],
                    heads, l2cfg["default_head"], len(cfg["classes"]), device)
    else:
        raise ValueError(f"unknown layer2.backbone {backbone!r}; expected 'panderm' or 'rtdetr'")

    op = os.path.join(wd, cfg["ood"]["file"])
    # --- INTEGRATION CHANGE: honour `ood.enabled` (default true). Temporarily
    #     false because the shipped npz was fit on the retired image-only Layer
    #     1. The MahalanobisOOD maths above is untouched. See ml_core/README.md.
    ood_enabled = cfg["ood"].get("enabled", True)
    ood = (MahalanobisOOD(op, cfg["ood"].get("threshold"))
           if ood_enabled and os.path.exists(op) else None)
    if ood is None:
        print(f"[models] OOD disabled (enabled={ood_enabled}, npz_exists={os.path.exists(op)})")
    assert l1.classes == list(cfg["classes"]), "class order mismatch L1 vs config"
    print(f"[models] ready on {device}: L1 T={l1.temperature:.3f}, OOD={'on' if ood else 'off'}")
    return l1, l2, ood, device
