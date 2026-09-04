# backend/ml_core — authoritative ML inference engine

`ml_core.engine.Engine` (from the ML team's handover) is now the **single
source of truth** for all ML inference and routing:

- preprocessing (shades-of-gray colour constancy p6, then 299px for Layer 1 /
  Layer-2-backbone-specific resolution for Layer 2 — 224px for PanDerm, 512px
  for RT-DETR)
- **multimodal Layer 1** — MobileNetV3-Large, image + `age`/`sex`/`localization`
  metadata, temperature `0.959` (read from the checkpoint), 7 HAM10000 classes
- OOD check (Mahalanobis) — **currently disabled**, see below
- **gate** — escalate to Layer 2 if `confidence < 0.5` OR normalised
  `entropy > 0.3`
- **Layer 2** — selectable backbone via `layer2.backbone` in
  `config/thresholds.yaml` (see "Layer 2 backbone" below):
  - `panderm` (**current default**) — PanDerm_Base (BEiT ViT-B/16), fully
    fine-tuned end-to-end on HAM10000, single self-contained checkpoint, no
    selectable heads
  - `rtdetr` (fallback/reference) — RT-DETR R50vd encoder + GOA head; heads
    `ham` and `hampad`, default `hampad`
- **risk** — `malignant_probability = P(mel) + P(bcc) + P(akiec)`,
  `risk_flag = malignant_probability >= 0.26`, `risk_level` high / moderate
  (`>= 0.13`) / low, plus `uncertain` (`confidence < 0.5`) and `advisory`

## Layer 2 backbone: PanDerm (current) vs RT-DETR (fallback)

`config/thresholds.yaml`'s `layer2.backbone` flag selects which Layer 2
implementation `models.load_all()` builds. Both backbones' weights and code
are kept side by side during this transition; flip the flag and restart to
switch — no other code changes needed.

**PanDerm** (`backbone: panderm`, default). Source: `amsdds_v2/` (fully
fine-tuned PanDerm_Base, HAM10000-only — see
`amsdds_v2/notebooks/panderm.ipynb`). The trained checkpoint
(`amsdds_v2/outputs/layer2_panderm_ft_v1.pt`) is copied into the weights
directory as `layer2_panderm_ft_v1.pt` and is fully self-contained: BEiT
backbone weights, classification head, and calibration temperature (0.827) in
one file — unlike RT-DETR there is no separate encoder/head/feat_mu/feat_sd
split. `models.PanDermLayer2` builds a plain `timm.models.beit.Beit` from the
bundled `arch` dict and loads `model_state` with `strict=True`.

PanDerm ships **no selectable `ham`/`hampad`-style heads**. Two candidate
frozen-feature heads exist in `amsdds_v2/outputs/` (`layer2_panderm_probe.pt`,
HAM-only, frozen pretrained backbone; `layer2_panderm_pad.pt`, PAD-UFES-20
smartphone domain) but they were trained on features from **two different
backbones** — pretrained vs. HAM-fine-tuned (see the notebook's own note at
export time) — so, unlike RT-DETR's single shared frozen encoder, they cannot
be served behind one loaded model. Loading two full ViT-B/16 backbones just to
offer a head switch was judged not worth it; `layer2_head` is still accepted
by the API/engine for backward compatibility but is a no-op under
`PanDermLayer2` (always returns head name `"panderm"`).

**RT-DETR** (`backbone: rtdetr`, fallback/reference — do not delete). The
original handover implementation, untouched: `models.RTDetrEncoder` /
`models.Layer2`. Kept so results can still be reproduced/compared until
PanDerm is fully validated in production.

The old `backend/decision_engine/`, `backend/services/model_service.py` and
`backend/services/layer2_stub.py` are **retired** — they no longer make any ML
decision. They are left in the tree (unused) and can be deleted once this
integration has settled.

## Relationship to the handover

These files are a copy of
`handover/amsdds_backend/amsdds/backend/{engine,models,preprocess}.py` and
`.../config/thresholds.yaml`. The **only** deliberate changes are:

| file | change | why |
|---|---|---|
| `engine.py` | `from preprocess import …` / `from models import …` → `from .preprocess` / `from .models` | package imports |
| `models.py` | `load_all()` now honours `cfg["ood"]["enabled"]` (default `true`) | OOD on/off switch |
| `config/thresholds.yaml` | added `ood.enabled: false` (+ comment) | disable OOD |

No model architecture, preprocessing maths, inference maths, gate logic, risk
logic or Layer 2 behaviour has been modified. `MahalanobisOOD` is intact.

`mock_engine.MockEngine` is **new** and used only in `DEVELOPMENT_MOCK_MODE` /
the test suite. Real mode always uses `Engine`.

## OOD is intended to be temporarily DISABLED

`config/thresholds.yaml` documents `ood.enabled: false`, but the live value in
that file is currently `true` (pre-existing drift, unrelated to the PanDerm
Layer 2 swap — not corrected here since fixing it is an OOD-logic change, out
of scope for that work). Until it is corrected, every real image is likely to
be misclassified as OOD (`unknown`) per the reason below, before Layer 2 is
ever reached.

**Reason.** `ood_scores_mobilenet.npz` was generated from the *retired*
image-only 224px Layer 1, not the current multimodal 299px model. Its stored
class means / precision matrix describe a different 1280-D feature space, so
genuine HAM10000 images score a Mahalanobis distance of ~12 000 against a
threshold of ~3 600 and are wrongly returned as `unknown`
("Not a recognised skin lesion").

**Do not** change the OOD threshold or the Mahalanobis maths to work around
this. The implementation stays as-is.

**Re-enabling.** When the ML team delivers a corrected artifact fit on
`layer1_multimodal_mobilenetv3.pt` (same 299px preprocessing, same feature
extraction point), drop it in the weights dir and set `ood.enabled: true`.
No code change is required.

## Weights

**Never committed** (`.gitignore` blocks `*.pt` / `*.npz`):

```
layer1_multimodal_mobilenetv3.pt
layer2_panderm_ft_v1.pt           # current default Layer 2 (~343 MB) — copied from amsdds_v2/outputs/
layer2_rtdetr.pt                  # RT-DETR fallback/reference
layer2_rtdetr_goa_head.pt
layer2_rtdetr_goa_head_hampad.pt
ood_scores_mobilenet.npz          # present but unused while OOD is disabled
```

Default location: `handover/amsdds_backend/amsdds/weights/`. Override with the
`AMSDDS_WEIGHTS` environment variable (the backend resolves it to an absolute
path so behaviour never depends on the working directory).

RT-DETR Layer 2 also downloads the `PekingU/rtdetr_r50vd` skeleton from the
HuggingFace Hub on first construction (needs network or a warm HF cache).
PanDerm Layer 2 needs no Hub access — it is fully self-contained in
`layer2_panderm_ft_v1.pt` plus the `timm` package (BEiT implementation).
