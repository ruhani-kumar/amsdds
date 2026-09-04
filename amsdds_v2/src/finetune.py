"""End-to-end fine-tuning of PanDerm_Base (BEiT ViT-B/16) for 7-class HAM10000.

Uses the PanDerm authors' recommended recipe (batch 128, lr 5e-4, 50 epochs,
layer decay 0.65, drop_path 0.2, wd 0.05, mixup 0.8 / cutmix 1.0, weighted
sampler, TTA at test) but with our own loop rather than their
run_class_finetuning.py — that script pins torch 2.4.1 and drags in the CAEv2
furnace tree, which is an hour of dependency work on Colab for no gain.

T4 notes:
  * fp16 + GradScaler (Turing has no bf16).
  * Physical batch 32 x accum 4 = effective 128. ViT-B/16 @224 fp16 backward
    at batch 32 sits around 9 GB, comfortably inside 16 GB.
  * The loop is DATA-bound, not compute-bound, because shades-of-gray colour
    constancy is pure CPU and Colab gives you 2 vCPUs. Run `cache_cc_images`
    once first — it pays for itself within three epochs.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import CLASSES, MALIGNANT, shades_of_gray

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# --------------------------------------------------------------- CC image cache
def cache_cc_images(df, out_dir="/content/cc_cache", size=288):
    """Apply colour constancy once, save 288px JPEGs to LOCAL disk.

    Colour constancy on a 600x450 array costs ~15 ms. Over 50 epochs that is
    ~90 minutes of pure CPU repeated for no reason. 288px leaves headroom for
    RandomResizedCrop to 224. Local disk, never Drive — Drive is a network
    mount and random reads from it will destroy throughput.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths, t0 = [], time.perf_counter()
    for i, (src, iid) in enumerate(zip(df["path"], df["image_id"])):
        dst = f"{out_dir}/{iid}.jpg"
        if not os.path.exists(dst):
            img = Image.open(src).convert("RGB")
            img = Image.fromarray(shades_of_gray(np.array(img), 6))
            img.thumbnail((size * 2, size * 2))
            img.resize((size, size), Image.BICUBIC).save(dst, quality=95)
        paths.append(dst)
        if i % 2000 == 0:
            print(f"  {i}/{len(df)}  {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"cached {len(paths)} in {(time.perf_counter()-t0)/60:.1f} min -> {out_dir}")
    return paths


class CCDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.tf = list(paths), list(labels), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def train_transform(size=224):
    # No colour jitter: colour constancy already normalised illumination, and
    # jittering after it re-introduces exactly the variation it removed.
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(30)], p=0.5),
        T.ToTensor(), T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def eval_transform(size=224):
    return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


# ------------------------------------------------------------------- model
def build_finetune_model(ckpt_path, n_classes=7, img_size=224, drop_path=0.2,
                         device="cuda"):
    """Same verified BEiT layout as the frozen path, plus a fresh head.
    head.{weight,bias} are the ONLY keys allowed to be missing."""
    from timm.models.beit import Beit

    m = Beit(img_size=img_size, patch_size=16, embed_dim=768, depth=12,
             num_heads=12, init_values=0.1, use_abs_pos_emb=True,
             use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
             global_pool="token", drop_path_rate=drop_path,
             num_classes=n_classes)

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "module"):
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
            break
    sd = {k.replace("module.", ""): v for k, v in obj.items()}
    bad = m.load_state_dict(sd, strict=False)
    missing = [k for k in bad.missing_keys if not k.startswith("head")]
    assert not missing and not bad.unexpected_keys, \
        f"bad load: missing={missing[:5]} unexpected={list(bad.unexpected_keys)[:5]}"
    nn.init.trunc_normal_(m.head.weight, std=0.01)
    nn.init.zeros_(m.head.bias)
    print(f"[ft] PanDerm loaded, fresh {n_classes}-class head")
    return m.to(device)


def layer_decay_groups(model, base_lr, wd=0.05, decay=0.65):
    """Lower layers get exponentially smaller LR. Essential when fine-tuning a
    foundation model on 7k images — a flat LR washes out the pretraining that
    is the entire reason for using PanDerm."""
    n_layers = len(model.blocks)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("blocks."):
            layer = int(name.split(".")[1]) + 1
        elif name.startswith(("cls_token", "pos_embed", "patch_embed")):
            layer = 0
        else:
            layer = n_layers + 1
        no_decay = p.ndim == 1 or name.endswith(".bias") or "gamma" in name
        key = (layer, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": base_lr * (decay ** (n_layers + 1 - layer)),
                           "weight_decay": 0.0 if no_decay else wd}
        groups[key]["params"].append(p)
    print(f"[ft] {len(groups)} param groups, lr {min(g['lr'] for g in groups.values()):.2e}"
          f" .. {max(g['lr'] for g in groups.values()):.2e}")
    return list(groups.values())


def make_sampler(labels):
    """Weighted sampler: each class drawn with probability proportional to
    1/count, so an epoch is roughly class-balanced."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    w = (1.0 / np.maximum(counts, 1))[labels]
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(labels), True)


# ------------------------------------------------------------------ training
def finetune(model, tr_paths, tr_y, va_paths, va_y, *, epochs=50, batch=32,
             accum=4, lr=5e-4, wd=0.05, layer_decay=0.65, warmup=10,
             mixup_a=0.8, cutmix_a=1.0, workers=2, device="cuda",
             out_path="/content/best_ft.pt"):
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy

    tr_dl = DataLoader(CCDataset(tr_paths, tr_y, train_transform()), batch_size=batch,
                       sampler=make_sampler(tr_y), num_workers=workers,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    va_dl = DataLoader(CCDataset(va_paths, va_y, eval_transform()), batch_size=64,
                       shuffle=False, num_workers=workers, pin_memory=True)

    mixup = Mixup(mixup_alpha=mixup_a, cutmix_alpha=cutmix_a, label_smoothing=0.1,
                  num_classes=len(CLASSES))
    crit = SoftTargetCrossEntropy()
    opt = torch.optim.AdamW(layer_decay_groups(model, lr, wd, layer_decay))
    scaler = torch.amp.GradScaler("cuda")

    base_lrs = [g["lr"] for g in opt.param_groups]
    steps = len(tr_dl) // accum
    best = {"f1": -1.0, "epoch": -1}
    hist = []

    for ep in range(epochs):
        model.train()
        t0, tot = time.perf_counter(), 0.0
        for it, (x, y) in enumerate(tr_dl):
            if it % accum == 0:                       # cosine w/ linear warmup
                prog = (ep + it / max(len(tr_dl), 1))
                f = (prog / warmup if prog < warmup else
                     0.5 * (1 + math.cos(math.pi * (prog - warmup) / max(epochs - warmup, 1))))
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * f

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xm, ym = mixup(x, y)
            with torch.autocast("cuda", torch.float16):
                loss = crit(model(xm), ym) / accum
            scaler.scale(loss).backward()
            tot += float(loss.detach()) * accum

            if (it + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

        model.eval()
        preds, ys = [], []
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            for x, y in va_dl:
                preds.append(model(x.to(device)).float().argmax(1).cpu()); ys.append(y)
        p, t = torch.cat(preds).numpy(), torch.cat(ys).numpy()
        f1 = f1_score(t, p, average="macro"); acc = float((p == t).mean())
        hist.append({"epoch": ep, "loss": tot / max(len(tr_dl), 1),
                     "val_acc": acc, "val_f1": f1})
        star = ""
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": ep}
            torch.save(model.state_dict(), out_path); star = "  *"
        print(f"ep{ep:3d}  loss {tot/max(len(tr_dl),1):.4f}  val_acc {acc:.4f} "
              f" val_f1 {f1:.4f}  {time.perf_counter()-t0:.0f}s{star}", flush=True)

    model.load_state_dict(torch.load(out_path, map_location=device))
    print(f"\nbest epoch {best['epoch']}  val_f1 {best['f1']:.4f}")
    return model.eval(), hist


@torch.no_grad()
def predict_tta(model, paths, labels, views=4, batch=64, device="cuda"):
    """Averages softmax over flip/rotate views. Returns [N,7] probabilities."""
    tf = eval_transform()
    ds = CCDataset(paths, labels, lambda im: torch.stack(
        [tf(v) for v in [im,
                         im.transpose(Image.FLIP_LEFT_RIGHT),
                         im.transpose(Image.FLIP_TOP_BOTTOM),
                         im.transpose(Image.ROTATE_90)][:views]]))
    out = []
    for x, _ in DataLoader(ds, batch_size=batch, num_workers=2):
        n, v = x.shape[0], x.shape[1]
        with torch.autocast("cuda", torch.float16):
            lg = model(x.reshape(-1, *x.shape[-3:]).to(device)).float()
        out.append(lg.reshape(n, v, -1).softmax(-1).mean(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def logits_plain(model, paths, labels, batch=64, device="cuda"):
    """Canonical-view logits — needed for temperature fitting on val."""
    dl = DataLoader(CCDataset(paths, labels, eval_transform()), batch_size=batch,
                    num_workers=2)
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", torch.float16):
            out.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(out)
