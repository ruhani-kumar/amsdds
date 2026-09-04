"""Shared evaluation metrics for both layers."""
import numpy as np
import torch
import torch.nn.functional as F


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error. The gap between confidence and accuracy,
    averaged over confidence bins. Guo et al. (2017) — the paper the problem
    statement cites for overconfidence."""
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(np.float32)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def normalised_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy / log(n_classes), so values land in [0, 1]."""
    n = probs.shape[1]
    return -(probs * np.log(probs + 1e-12)).sum(1) / np.log(n)


def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor,
                    device: str = "cpu") -> float:
    """Post-hoc temperature scaling, fitted by LBFGS on the VALIDATION split.
    Applied at inference as softmax(logits / T). T > 1 means the model was
    overconfident; T < 1 means underconfident."""
    val_logits, val_labels = val_logits.to(device), val_labels.to(device)
    log_t = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(val_logits / log_t.exp(), val_labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def dangerous_misses(y_true, y_pred, classes, malignant=("mel", "bcc", "akiec")):
    """Boolean mask: truly malignant, predicted benign.

    This is the failure mode the whole adaptive architecture exists to catch,
    and the metric the threshold sweep optimises against. On our test split
    Layer 1 produces 82 of these out of 269 malignant lesions.
    """
    mal_i = [classes.index(c) for c in malignant]
    ben_i = [i for i in range(len(classes)) if i not in mal_i]
    return np.isin(y_true, mal_i) & np.isin(y_pred, ben_i)
