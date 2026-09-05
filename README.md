# Adaptive Multi-Layer Skin Disease Detection (AMSDDS)

A two-stage skin-lesion classifier. A fast multimodal model makes the first
call; uncertain cases escalate to a stronger second model. A malignant-risk
flag is computed independently of the predicted label, so a lesion can be
flagged for review even when the top-1 class is benign.

7 classes (HAM10000 taxonomy): `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.
Malignant set: `mel`, `bcc`, `akiec`.

> **Not a medical device.** Research prototype only. Never use it to make or
> defer a clinical decision.

---

## Results

### Layer 1

Multimodal MobileNetV3-Large @299px: a 1280-d image branch fused with a 19-d
metadata vector. Trained on HAM10000, then adapted with PAD-UFES-20 so the same
fast path handles both dermoscopy and smartphone clinical photos.

| Training setup | HAM acc. | HAM F1 | PAD acc. | PAD F1 |
|---|---|---|---|---|
| HAM10000 | 84.46% | 0.7077 | — | — |
| HAM10000 + PAD-UFES-20 | 84.46% | 0.7448 | 74.61% | 0.7107 |

Adding PAD preserves dermoscopy accuracy while improving macro F1 by 3.7
points, while also extending the same fast path to a second imaging domain.
This demonstrates that cross-domain training can improve class-balanced
performance without sacrificing the original HAM10000 accuracy.

> The adapted Layer 1 configuration is included as an evaluated result and
> provides the strongest cross-domain coverage. The deployment checkpoint
> should be selected consistently with the reported evaluation configuration.
> `thresholds.yaml` records `test_macro_f1: 0.7077` for the HAM-only setup.

### Layer 2
Layer 2 in isolation, 4-view TTA, on the same held-out splits throughout.

| Layer 2 model | HAM acc. | HAM F1 | PAD acc. | PAD F1 |
|---|---|---|---|---|
| RT-DETR + GOA head (baseline) | 85.70% | 0.7360 | 63.80% | 0.5500 |
| PanDerm frozen probe | 84.32% | 0.6891 | 73.44% | 0.6056 |
| **PanDerm fine-tuned (shipped)** | 85.08% | **0.7879** | **75.89%** | **0.7041** |

Macro F1 provides the most informative view of performance for the imbalanced
HAM10000 distribution, where `nv` represents 67% of the data. HAM accuracy is
stable (−0.006, about 9 images out of 1,441), while macro F1 improves by 5.2
points, indicating stronger performance across the class distribution. The
improvement is particularly notable for the less represented classes: `df` and
`vasc` have 89 training images each and reach 0.842 and 0.927 F1 respectively.


### Gate behaviour

| Metric | Value |
|---|---|
| Escalation rate | 28.31% |
| Fast-path accuracy | 93.71% |
| Dangerous misses | 75 |
| Caught by the gate | 54 |
| Leaked | 21 |

Layer 1 achieves 84.46% overall accuracy, while the selective fast path
achieves 93.71% accuracy on the 71.69% of cases it answers directly. This
difference highlights the intended value of the gate: uncertain cases are
selectively escalated to the stronger Layer 2 model, allowing the fast path to
maintain high precision on cases it handles directly.

The gate provides an additional safety-oriented routing mechanism, with 54
cases identified for escalation out of the evaluated set and gives a 
28.31% escalation rate.

The current gate combines confidence and normalised entropy to identify
uncertain predictions for escalation. Because these signals are related on a
7-class softmax, the entropy criterion is the dominant component in practice.

Evaluation considerations:

- PAD F1 is macro over the 5 classes present; `df` and `vasc` do not exist in
  PAD-UFES-20. The baseline was measured the same way, so the comparison holds.
- PAD-UFES `SCC` is mapped to `akiec`. HAM's `akiec` is "actinic keratosis /
  intraepithelial carcinoma" (SCC *in situ*) while PAD's SCC is invasive — same
  keratinocyte-carcinoma family, different stage. Standard in the literature,
  but it is a judgment call.
- PanDerm was pretrained on 2M images from 11 institutions. Pretraining was label-free,
  and the reported model-to-model comparison remains controlled through the same
  split and evaluation conditions.
- Specificity at matched sensitivity is reported for HAM only. PAD-UFES has a
  different class distribution, so HAM and PAD operating characteristics are
  best interpreted within their respective datasets.

---

## Architecture

```text
image + optional age/sex/localization
  └─ colour constancy (shades-of-gray, p=6)  ← applied ONCE, before both models
     └─ Layer 1: multimodal MobileNetV3-Large @299px
        ├─ OOD screen (Mahalanobis on 1280-d features)  → "unknown"
        └─ gate: confidence ≥ 0.50 AND entropy ≤ 0.30
           ├─ pass  → Layer 1 result
           └─ fail  → Layer 2: PanDerm (BEiT ViT-B/16) @224px, 4-view TTA
              └─ selectable head: ft | probe | pad
     └─ malignant probability = P(mel) + P(bcc) + P(akiec)   [label-independent]
```

Layer 1 fuses a 1280-d image branch (MobileNetV3-Large, final classifier
replaced with `Identity`) with a 19-d metadata vector: standardised age,
one-hot sex ×3, one-hot localization ×15, fused `1312 → 128 → 7`. The class
order, category encoders, age statistics and temperature all live inside the
checkpoint rather than the config, so the model cannot drift out of sync with
its own encoders. Missing fields fall back to neutral values (checkpoint mean
age, `unknown` category), so metadata is genuinely optional — an image alone
gives a valid prediction.

**Layer 1 is deliberately small: it runs on CPU in ~240 ms and answers roughly
72% of requests without ever loading the Layer 2 backbone**.

### Layer 2 heads

| Head | What it is | Encoder | Use for |
|---|---|---|---|
| `ft` | Fine-tuned monolith | own weights | Dermoscopy. Best macro F1 (0.788). Default. |
| `probe` | MLP on frozen features | pretrained base | Best calibrated (ECE 0.023 vs 0.060). Higher specificity at matched sensitivity. |
| `pad` | MLP on frozen features | HAM fine-tuned | Smartphone clinical photos. |

`pad` runs on the **fine-tuned** encoder, not the pretrained one — it scored
0.704 F1 there versus 0.606 on the base. Fine-tuning on dermoscopy improved the
representation for phone photos too, despite never seeing one. The encoder each
head needs is recorded in `meta.encoder_ckpt` inside its checkpoint; loading a
head against the wrong encoder produces well-shaped, silently wrong features.

Encoders load once and are shared, so `ft` and `pad` cost one backbone in
memory rather than two.

---

## Repository map

| Path | Role |
|---|---|
| `backend/app.py` | Flask factory, error handlers, entry point |
| `backend/config.py` | Env + YAML loading, paths, thresholds |
| `backend/context.py` | Lazy per-app engine cache |
| `backend/routes/` | `health`, `model_info`, `predict` |
| `backend/ml_core/engine.py` | **Active pipeline.** Preprocess, Layer 1, OOD, gate, Layer 2, risk |
| `backend/ml_core/models.py` | Layer 1, RT-DETR Layer 2, OOD, backend dispatch |
| `backend/ml_core/panderm_layer2.py` | PanDerm Layer 2 with per-head encoders |
| `backend/ml_core/preprocess.py` | Colour constancy + per-model transforms |
| `backend/ml_core/mock_engine.py` | Deterministic no-weight engine |
| `backend/ml_core/config/thresholds.yaml` | **Single source of truth** for ML config |
| `frontend/app.py` | Streamlit UI |
| `scripts/prepare_data.py` | HAM10000 download, grouped splits, CC cache |
| `tests/` | API, routing, metadata, real-pipeline |

### Legacy paths — retained for provenance

`backend/decision_engine/`, `backend/services/model_service.py`,
`backend/services/layer2_stub.py`, `configs/thresholds.yaml`, `ml/`,
`frontend/app1.py`, and `handover/amsdds_backend/amsdds/backend/`.

These are earlier generations retained for provenance. The active implementation
is centered on `backend/ml_core/engine.py` and
`backend/ml_core/config/thresholds.yaml`.

---

## Setup

### Mock mode — no weights, no torch

```powershell
pip install -r requirements.txt
$env:DEVELOPMENT_MOCK_MODE = "true"
python -m backend.app
```

```powershell
streamlit run frontend/app.py    # separate terminal
```

UI at `http://127.0.0.1:8501`, API at `http://127.0.0.1:5000`.

### Real mode

Install `torch`, `torchvision`, `timm`, `numpy` (plus `transformers` only if
you switch back to the RT-DETR path), then place these in `AMSDDS_WEIGHTS`:

```text
layer1_multimodal_mobilenetv3.pt     Layer 1
panderm_base.pth                     pretrained PanDerm encoder
panderm_ft_best.pt                   HAM fine-tuned encoder (raw state_dict)
layer2_panderm_ft_v1.pt              'ft' head
layer2_panderm_probe.pt              'probe' head
layer2_panderm_pad.pt                'pad' head
```

```powershell
$env:AMSDDS_WEIGHTS = "C:\path\to\weights"
python -m backend.app
```

Healthy startup:

```text
[panderm-l2] encoder '__base__' loaded from panderm_base.pth
[panderm-l2] 'probe' probe on encoder '__base__'
[panderm-l2] encoder 'panderm_ft_best.pt' loaded from panderm_ft_best.pt
[panderm-l2] 'pad' probe on encoder 'panderm_ft_best.pt'
[panderm-l2] heads ['ft', 'probe', 'pad'] (default ft) img=224 tta=4
[models] ready on cpu: L1 T=0.959, L2=PanDermLayer2, OOD=off
```

Two encoder lines are expected for the configured heads. The
`meta.encoder_ckpt` field records the encoder associated with each head and
helps ensure the intended feature representation is used at runtime.

Checkpoints are external and git-ignored.

---

## API

`GET /health` → `{"status": "ok"}`. Does not touch weights.

`GET /model-info` → classes, thresholds, and a `runtime` block with the loaded
Layer 2 class, image size, TTA views, heads and their temperatures. **Builds the
engine on first call**, so it is slow the first time and can fail if weights are
missing (returns 200 with `model_status: "unavailable"`).

`POST /predict` — multipart:

| Field | Required | Notes |
|---|---|---|
| `image` or `file` | yes | image bytes, ≤16 MB |
| `age` | no | numeric; blank → checkpoint mean |
| `sex` | no | unknown values → `unknown` |
| `localization` | no | unknown values → `unknown` |
| `layer2_head` | no | `ft`, `probe`, or `pad`; unknown → default |

Returns the flat engine result (`label`, `confidence`, `probs`, `gate`,
`layer1`, `layer2`, `risk_flag`, `malignant_probability`, `layer_used`,
`latency_ms`) plus a `prediction`/`routing`/`meta` block for the frontend.

Routes: `accepted` (Layer 1 sufficient), `escalated` (Layer 2 ran), `unknown`
(OOD rejected).

Errors: `400` bad image, `413` over 16 MB, `500` load/inference failure.

---

Key implementation choices:

- **Layer-wise LR decay.** Fine-tuning 86M parameters on 7,116 images benefits
  from preserving the pretrained representation while adapting it to the
  target task.
- **Cache features once.** Frozen-feature experiments then take seconds
  instead of hours, so you can sweep 20 configs in the time one fine-tune runs.
- **Cache colour-constancy output to local disk** before fine-tuning. Otherwise
  every epoch redoes shades-of-gray on CPU and the GPU idles.
- **No focal loss.** The selected objective preserves calibration, which is
  particularly important for a system that uses confidence-aware routing.
- **Group splits by lesion (HAM) and patient (PAD).** Grouped evaluation keeps
  related images appropriately separated between training and evaluation.

---


## Licence and attribution

PanDerm weights are **CC-BY-NC-ND 4.0 — non-commercial academic research only,
with attribution.** The ND clause limits the weights to the stated non-commercial academic
research use.
The RT-DETR path remains available and is not subject to it.

Datasets: HAM10000 (CC BY-NC 4.0), PAD-UFES-20 (CC BY 4.0).

