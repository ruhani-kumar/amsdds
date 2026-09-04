"""Head training on frozen cached features.

Design choices, and why:
  * Logit-adjusted loss (Menon et al.) — train with `logits + tau*log(prior)`,
    infer with plain `logits`. Train-time version of the post-hoc tau hack.
  * Label smoothing — attacks peaked-and-wrong directly.
  * Manifold mixup on features — one of the most reliable calibration wins
    available, and free when features are already cached.
  * NO focal loss. It's the reflex pick for imbalance and it measurably
    degrades calibration, which is the exact thing being fixed here.
  * Temperature fitted on val by NLL AFTER training. Expect T > 1. The v1
    T=0.959 divides logits, i.e. it sharpened an already-overconfident model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .data import CLASSES, MALIGNANT


def build_head(d_in: int, n_classes: int, hidden: int = 512, drop: float = 0.3) -> nn.Module:
    """Mirrors v1 `_build_head` so exported checkpoints load unchanged:
    hidden < 8 means linear probe."""
    if hidden < 8:
        return nn.Sequential(nn.Dropout(drop), nn.Linear(d_in, n_classes))
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                         nn.Dropout(drop), nn.Linear(hidden, n_classes))


def standardize(train_f: np.ndarray):
    mu = train_f.mean(0)
    sd = train_f.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def train_head(fx_tr, y_tr, fx_va, y_va, prior, *, dim, hidden=512, drop=0.3,
               tau=1.0, smooth=0.1, mixup=0.2, lr=1e-3, wd=1e-4, epochs=60,
               batch=256, device="cuda", seed=0, verbose=True):
    """Features must already be standardized. Returns (head, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    head = build_head(dim, len(CLASSES), hidden, drop).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    Xtr = torch.tensor(fx_tr, dtype=torch.float32, device=device)
    Ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    Xva = torch.tensor(fx_va, dtype=torch.float32, device=device)
    Yva = torch.tensor(y_va, dtype=torch.long, device=device)
    logp = torch.tensor(np.log(prior + 1e-12), dtype=torch.float32, device=device)

    best = {"f1": -1.0, "state": None, "epoch": -1}
    hist = []
    n = len(Xtr)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]

            if mixup > 0:
                lam = float(np.random.beta(mixup, mixup))
                j = torch.randperm(len(xb), device=device)
                xb = lam * xb + (1 - lam) * xb[j]
                out = head(xb) + tau * logp          # logit adjustment
                loss = (lam * F.cross_entropy(out, yb, label_smoothing=smooth)
                        + (1 - lam) * F.cross_entropy(out, yb[j], label_smoothing=smooth))
            else:
                out = head(xb) + tau * logp
                loss = F.cross_entropy(out, yb, label_smoothing=smooth)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb)
        sched.step()

        head.eval()
        with torch.no_grad():
            pv = head(Xva).argmax(1).cpu().numpy()   # plain logits at inference
        f1 = f1_score(Yva.cpu().numpy(), pv, average="macro")
        acc = float((pv == Yva.cpu().numpy()).mean())
        hist.append({"epoch": ep, "loss": tot / n, "val_acc": acc, "val_f1": f1})

        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  ep{ep:3d}  loss {tot/n:.4f}  val_acc {acc:.4f}  val_f1 {f1:.4f}")

    head.load_state_dict(best["state"])
    if verbose:
        print(f"  best epoch {best['epoch']} val_f1 {best['f1']:.4f}")
    return head.eval(), hist


@torch.no_grad()
def logits_of(head, fx, device="cuda", batch=1024) -> np.ndarray:
    head.eval()
    out = []
    for i in range(0, len(fx), batch):
        xb = torch.tensor(fx[i:i + batch], dtype=torch.float32, device=device)
        out.append(head(xb).float().cpu().numpy())
    return np.concatenate(out)


def fit_temperature(logits: np.ndarray, y: np.ndarray, device="cuda") -> float:
    """Minimise val NLL over a single scalar T. Expect T > 1 for an
    overconfident model — that is the whole point."""
    L = torch.tensor(logits, dtype=torch.float32, device=device)
    Y = torch.tensor(y, dtype=torch.long, device=device)
    logT = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(L / logT.exp(), Y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def evaluate(probs: np.ndarray, y: np.ndarray, risk_threshold: float = 0.26) -> dict:
    """7-class metrics plus the binary malignant screening metrics — the ones
    that actually matter for a triage tool."""
    pred = probs.argmax(1)
    mal_idx = [CLASSES.index(c) for c in MALIGNANT]
    mal_p = probs[:, mal_idx].sum(1)
    is_mal = np.isin(y, mal_idx)
    flag = mal_p >= risk_threshold

    tp = int((flag & is_mal).sum()); fn = int((~flag & is_mal).sum())
    tn = int((~flag & ~is_mal).sum()); fp = int((flag & ~is_mal).sum())

    return {
        "acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ece": ece(probs, y),
        "mal_sensitivity": tp / max(tp + fn, 1),
        "mal_specificity": tn / max(tn + fp, 1),
        "per_class_f1": {c: round(float(v), 4) for c, v in
                         zip(CLASSES, f1_score(y, pred, average=None,
                                               labels=range(len(CLASSES))))},
    }


def sweep_risk_threshold(probs, y, target_sens=0.966):
    """Find the lowest threshold still meeting the v1 sensitivity, and report
    the specificity you get for it. v1 was sens 0.966 / spec 0.484 — beating
    that specificity at equal sensitivity is the headline claim."""
    mal_idx = [CLASSES.index(c) for c in MALIGNANT]
    mal_p = probs[:, mal_idx].sum(1)
    is_mal = np.isin(y, mal_idx)
    rows = []
    for t in np.linspace(0.02, 0.90, 89):
        flag = mal_p >= t
        sens = (flag & is_mal).sum() / max(is_mal.sum(), 1)
        spec = (~flag & ~is_mal).sum() / max((~is_mal).sum(), 1)
        rows.append((float(t), float(sens), float(spec)))
    ok = [r for r in rows if r[1] >= target_sens]
    return (max(ok, key=lambda r: r[2]) if ok else None), rows
