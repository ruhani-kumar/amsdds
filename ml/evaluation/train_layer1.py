import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader

from ml.models import build_model
from ml.preprocessing import HAM, eval_tf, train_tf
from ml.evaluation.metrics import ece, fit_temperature

SEED = 42
WEIGHT_CAP = 2.0
LABEL_SMOOTHING = 0.05


def class_weights(y, n_classes, cap=WEIGHT_CAP):
    """sqrt-inverse frequency, normalised to min 1.0, capped."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    return np.clip(w / w.min(), 1.0, cap)


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            L.append(model(x.to(device)).float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def val_f1(model, loader, device):
    L, Y = collect_logits(model, loader, device)
    return f1_score(Y.numpy(), L.argmax(1).numpy(), average="macro")


def train_stage(model, loader, val_loader, opt, sched, lossfn, scaler,
                epochs, tag, device, state):
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                loss = lossfn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        f1 = val_f1(model, val_loader, device)
        if f1 > state["best_f1"]:
            state["best_f1"] = f1
            state["best"] = {k: v.detach().clone().cpu()
                             for k, v in model.state_dict().items()}
        print(f"{tag} ep {ep:2d}  val macro-F1 {f1:.4f}  ({time.time()-t0:.0f}s)")


def main(a):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    df = pd.read_csv(a.splits)
    classes = sorted(df.dx.unique())
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    print({s: len(d) for s, d in zip(("train", "val", "test"), (tr, va, te))})

    dl_tr = DataLoader(HAM(tr, train_tf), a.batch, shuffle=True,
                       num_workers=a.workers, pin_memory=True, drop_last=True,
                       persistent_workers=a.workers > 0)
    dl_va = DataLoader(HAM(va, eval_tf), 64, num_workers=a.workers, pin_memory=True)
    dl_te = DataLoader(HAM(te, eval_tf), 64, num_workers=a.workers, pin_memory=True)

    model = build_model(len(classes)).to(device)

    w = class_weights(tr.y.values, len(classes))
    print("class weights:", dict(zip(classes, np.round(w, 2))))
    lossfn = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device),
                                 label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    state = {"best_f1": -1.0, "best": None}

    # Stage 1 — frozen backbone, classifier head only
    for p in model.features.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    train_stage(model, dl_tr, dl_va, opt,
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs1),
                lossfn, scaler, a.epochs1, "S1", device, state)

    # Stage 2 — full unfreeze, discriminative learning rates
    for p in model.features.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)
    train_stage(model, dl_tr, dl_va, opt,
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs2),
                lossfn, scaler, a.epochs2, "S2", device, state)

    model.load_state_dict(state["best"])
    model.to(device)
    print(f"\nbest val macro-F1 {state['best_f1']:.4f}")

    # Calibration fitted on VAL, evaluated on TEST
    lv, yv = collect_logits(model, dl_va, device)
    temp = fit_temperature(lv, yv, device)
    print(f"fitted temperature T = {temp:.4f}"
          f"  ({'over' if temp > 1 else 'under'}confident before scaling)")

    lt, yt = collect_logits(model, dl_te, device)
    yt = yt.numpy()
    p_raw = F.softmax(lt, 1).numpy()
    p_cal = F.softmax(lt / temp, 1).numpy()
    pred = p_cal.argmax(1)

    metrics = {
        "test_acc": float(accuracy_score(yt, pred)),
        "test_macro_f1": float(f1_score(yt, pred, average="macro")),
        "ece_raw": ece(p_raw, yt),
        "ece_cal": ece(p_cal, yt),
        "val_macro_f1": float(state["best_f1"]),
        "temperature": temp,
    }
    print(f"\ntest accuracy  {metrics['test_acc']:.4f}")
    print(f"test macro-F1  {metrics['test_macro_f1']:.4f}")
    print(f"ECE  raw {metrics['ece_raw']:.4f} -> calibrated {metrics['ece_cal']:.4f}")
    print("\n", classification_report(yt, pred, target_names=classes, digits=3))
    print(pd.DataFrame(confusion_matrix(yt, pred), index=classes, columns=classes))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({
        "state": state["best"],
        "classes": classes,
        "temperature": temp,
        "img_size": 224,
        "color_constancy": True,
        "weight_cap": WEIGHT_CAP,
        "label_smoothing": LABEL_SMOOTHING,
        "metrics": metrics,
    }, a.out)
    print("\nsaved:", a.out)

    if a.metrics:
        os.makedirs(os.path.dirname(a.metrics) or ".", exist_ok=True)
        json.dump(metrics, open(a.metrics, "w"), indent=2)
        print("saved:", a.metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="data/splits.csv")
    p.add_argument("--out", default="checkpoints/layer1_mobilenetv3_best.pt")
    p.add_argument("--metrics", default="docs/results/metrics_mobilenetv3.json")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs1", type=int, default=5)
    p.add_argument("--epochs2", type=int, default=25)
    p.add_argument("--workers", type=int, default=2)
    main(p.parse_args())
