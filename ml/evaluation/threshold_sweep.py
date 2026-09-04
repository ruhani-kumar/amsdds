import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.models import load_checkpoint
from ml.preprocessing import HAM, eval_tf
from ml.evaluation.metrics import normalised_entropy, dangerous_misses

MIN_FASTPATH_ACC = 0.90
ESC_LO, ESC_HI = 0.20, 0.35
UNCERTAIN_FLOOR = 0.50
CONF_GRID = np.round(np.arange(0.50, 0.96, 0.05), 2)
ENT_GRID = np.round(np.arange(0.10, 0.61, 0.05), 2)


@torch.no_grad()
def logits_for(model, frame, device, workers=2):
    loader = DataLoader(HAM(frame, eval_tf), 64, num_workers=workers, pin_memory=True)
    model.eval()
    L, Y = [], []
    for x, y in loader:
        with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            L.append(model(x.to(device)).float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y).numpy()


def calibrated(logits, temp):
    p = F.softmax(logits / temp, dim=1).numpy()
    return p, p.max(1), normalised_entropy(p), p.argmax(1)


def score(conf, ent, pred, y, dang, c_thr, e_thr):
    accept = (conf >= c_thr) & (ent <= e_thr)
    esc = ~accept
    return {
        "conf_thr": c_thr,
        "entropy_thr": e_thr,
        "escalation_rate": esc.mean(),
        "fastpath_acc": (pred[accept] == y[accept]).mean() if accept.sum() else np.nan,
        "fastpath_n": int(accept.sum()),
        "dang_caught": esc[dang].mean() if dang.sum() else np.nan,
        "dang_leaked": int((dang & accept).sum()),
        "esc_acc": (pred[esc] == y[esc]).mean() if esc.sum() else np.nan,
    }


def choose(S):
    ok = S[(S.fastpath_acc >= MIN_FASTPATH_ACC) &
           (S.escalation_rate.between(ESC_LO, ESC_HI))]
    if not len(ok):
        print(f"No pair meets acc>={MIN_FASTPATH_ACC} and escalation in "
              f"[{ESC_LO},{ESC_HI}]. Relaxing the escalation band.")
        ok = S[S.fastpath_acc >= MIN_FASTPATH_ACC]
    if not len(ok):
        print("Still nothing. Falling back to best fast-path accuracy.")
        ok = S.nlargest(20, "fastpath_acc")
    ranked = ok.sort_values(["dang_caught", "escalation_rate"],
                            ascending=[False, True])
    return ranked.iloc[0], ranked


def plot(S, c_thr, e_thr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("fastpath_acc", "Fast-path accuracy", "viridis"),
              ("escalation_rate", "Escalation rate", "magma"),
              ("dang_caught", "Dangerous misses caught", "viridis"),
              ("dang_leaked", "Dangerous misses leaked (count)", "magma_r")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, title, cmap) in zip(axes.ravel(), panels):
        piv = S.pivot_table(index="entropy_thr", columns="conf_thr", values=col)
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in piv.columns], rotation=45)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{i:.2f}" for i in piv.index])
        ax.set_xlabel("confidence threshold")
        ax.set_ylabel("entropy threshold")
        ax.set_title(title)
        ax.scatter([list(piv.columns).index(c_thr)],
                   [list(piv.index).index(e_thr)],
                   marker="*", s=320, c="red", edgecolors="white", zorder=5)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    print("saved:", path)


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(a.ckpt, device)
    classes, temp = ckpt["classes"], float(ckpt["temperature"])
    print(f"loaded {a.ckpt} | T={temp:.4f} | {len(classes)} classes")

    df = pd.read_csv(a.splits)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))

    Lv, yv = logits_for(model, va, device, a.workers)
    Lt, yt = logits_for(model, te, device, a.workers)
    _, cv, ev, predv = calibrated(Lv, temp)
    _, ct, et, predt = calibrated(Lt, temp)

    dang_v = dangerous_misses(yv, predv, classes)
    dang_t = dangerous_misses(yt, predt, classes)
    print(f"val dangerous misses  {dang_v.sum()}")
    print(f"test dangerous misses {dang_t.sum()}")

    S = pd.DataFrame([score(cv, ev, predv, yv, dang_v, c, e)
                      for c in CONF_GRID for e in ENT_GRID])
    best, ranked = choose(S)
    CONF = float(best.conf_thr)
    ENT = float(best.entropy_thr)

    print(f"\n{'='*58}\nCHOSEN (val):  conf {CONF}   entropy {ENT}\n{'='*58}")
    print(f"  escalation rate     {best.escalation_rate:.1%}")
    print(f"  fast-path accuracy  {best.fastpath_acc:.1%}  (n={best.fastpath_n})")
    print(f"  dangerous caught    {best.dang_caught:.1%}")
    print(f"  dangerous leaked    {int(best.dang_leaked)}")
    print(f"  escalated-pile acc  {best.esc_acc:.1%}  <- Layer 2 must beat this")
    print("\ntop 10 candidates:")
    print(ranked.head(10).round(3).to_string(index=False))

    # ---- verify on test
    accept = (ct >= CONF) & (et <= ENT)
    esc = ~accept
    R = {
        "conf_threshold": CONF,
        "entropy_threshold": ENT,
        "uncertain_floor": UNCERTAIN_FLOOR,
        "temperature": temp,
        "test_escalation_rate": float(esc.mean()),
        "test_fastpath_acc": float((predt[accept] == yt[accept]).mean()),
        "test_escalated_acc": float((predt[esc] == yt[esc]).mean()),
        "test_overall_acc": float((predt == yt).mean()),
        "test_dangerous_total": int(dang_t.sum()),
        "test_dangerous_caught": int((dang_t & esc).sum()),
        "test_dangerous_leaked": int((dang_t & accept).sum()),
        "compute_saving_vs_always_on": float(1 - esc.mean()),
    }
    R["test_dangerous_caught_pct"] = (R["test_dangerous_caught"] /
                                      max(R["test_dangerous_total"], 1))

    print(f"\n{'='*58}\nTEST SET at chosen thresholds\n{'='*58}")
    for k, v in R.items():
        print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")
    print(f"\n  {R['test_escalation_rate']:.0%} escalate -> "
          f"{R['compute_saving_vs_always_on']:.0%} never touch the transformer.")
    print(f"  Gate catches {R['test_dangerous_caught']}/"
          f"{R['test_dangerous_total']} malignant-called-benign "
          f"({R['test_dangerous_caught_pct']:.0%}). "
          f"{R['test_dangerous_leaked']} leak — report this honestly.")

    # ---- artefacts
    plot(S, CONF, ENT, a.figure)
    for path, obj in [(a.sweep_csv, S), (a.routing_json, R)]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    S.to_csv(a.sweep_csv, index=False)
    json.dump(R, open(a.routing_json, "w"), indent=2)

    yaml_txt = f"""# configs/thresholds.yaml
# DERIVED from the validation sweep, not asserted.
# See docs/results/threshold_sweep.csv and docs/figures/threshold_sweep.png

decision_engine:
  conf_threshold: {CONF}
  entropy_threshold: {ENT}
  uncertain_floor: {UNCERTAIN_FLOOR}

calibration:
  temperature: {temp:.4f}
  applied_to: layer1_logits

classes: {classes}
malignant_classes: ['mel', 'bcc', 'akiec']

# Measured on the held-out test split with the thresholds above.
measured:
  escalation_rate: {R['test_escalation_rate']:.4f}
  fastpath_accuracy: {R['test_fastpath_acc']:.4f}
  escalated_subset_accuracy: {R['test_escalated_acc']:.4f}
  compute_saving: {R['compute_saving_vs_always_on']:.4f}
  dangerous_misses_total: {R['test_dangerous_total']}
  dangerous_misses_caught: {R['test_dangerous_caught']}
  dangerous_misses_leaked: {R['test_dangerous_leaked']}
"""
    os.makedirs(os.path.dirname(a.yaml) or ".", exist_ok=True)
    open(a.yaml, "w").write(yaml_txt)
    print("saved:", a.yaml)

    pd.DataFrame({
        "image_id": te.image_id.values,
        "true": [classes[i] for i in yt],
        "pred": [classes[i] for i in predt],
        "confidence": ct, "entropy": et,
        "escalated": esc, "dangerous_miss": dang_t,
    }).to_csv(a.routing_detail, index=False)
    print("saved:", a.routing_detail)

    # ---- Layer 2 training subset
    Ltr, _ = logits_for(model, tr, device, a.workers)
    _, ctr, etr, _ = calibrated(Ltr, temp)
    esc_tr = ~((ctr >= CONF) & (etr <= ENT))
    pd.DataFrame({"image_id": tr.image_id.values, "dx": tr.dx.values,
                  "confidence": ctr, "entropy": etr,
                  "escalated": esc_tr}).to_csv(a.layer2_subset, index=False)

    print(f"\nsaved: {a.layer2_subset}")
    print(f"  {esc_tr.sum()} / {len(esc_tr)} train images ({esc_tr.mean():.1%}) "
          f"fall below the gate.")
    if esc_tr.mean() < R["test_escalation_rate"] * 0.6:
        print("\n  *** WARNING ***")
        print(f"  Train escalation ({esc_tr.mean():.1%}) is far below test "
              f"({R['test_escalation_rate']:.1%}).")
        print("  These images were seen during fine-tuning, so Layer 1 is")
        print("  overconfident on them and the gate barely fires. This subset")
        print("  is NOT representative of what Layer 2 will receive.")
        print("  Fix: mine out-of-fold (5-fold CV), or score unseen ISIC 2019")
        print("  and keep its bottom ~30% by confidence (blueprint 7.2).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/layer1_mobilenetv3_best.pt")
    p.add_argument("--splits", default="data/splits.csv")
    p.add_argument("--yaml", default="configs/thresholds.yaml")
    p.add_argument("--figure", default="docs/figures/threshold_sweep.png")
    p.add_argument("--sweep-csv", dest="sweep_csv",
                   default="docs/results/threshold_sweep.csv")
    p.add_argument("--routing-json", dest="routing_json",
                   default="docs/results/routing_results.json")
    p.add_argument("--routing-detail", dest="routing_detail",
                   default="data/test_routing_detail.csv")
    p.add_argument("--layer2-subset", dest="layer2_subset",
                   default="data/layer2_train_subset.csv")
    p.add_argument("--workers", type=int, default=2)
    main(p.parse_args())
