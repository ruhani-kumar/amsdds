# Frontend + Backend + ML Engine Integration

Developer overview of the integration completed after the ML team handover.
**Not the final project README.** Describes the current repository state only.

---

## 1. Architecture

```
Streamlit frontend  (frontend/app.py)
        |   multipart POST /predict  (image + age/sex/localization/layer2_head)
        v
Flask backend       (backend/app.py + backend/routes/)
        |   validate image, read form fields, structured errors
        v
backend/context.py  ->  ONE cached ML engine
        |
        v
backend/ml_core/    (authoritative ML inference)
        |
  engine.Engine.predict():
        preprocess  (shades-of-gray p6, then 299px / 512px)
        v
    Layer 1  - multimodal MobileNetV3 (image + metadata), softmax(logits / 0.959)
        v
    OOD check  -> DISABLED (see section 8)
        v
    Gate:  confidence < 0.5  OR  normalized entropy > 0.3   -> escalate
        v
    Layer 2  - RT-DETR R50vd encoder + GOA head (HAM / HAMPAD, default HAMPAD)
        v
    Risk:  malignant_probability = P(mel)+P(bcc)+P(akiec)
           risk_flag = malignant_probability >= 0.26 ; risk_level ; uncertain ; advisory
        v
    flat result dict
        v
backend/services/response_builder.py  ->  JSON (flat fields + frontend compat block)
        v
Streamlit renders the response  (no ML logic in the frontend)
```

Key points:

- **Frontend** = presentation only. It sends inputs and displays whatever the
  backend returns.
- **Backend** owns API validation, form parsing, error handling, and
  orchestration (calling the one engine).
- **`backend/ml_core/`** is the single authoritative ML implementation:
  preprocessing, Layer 1, gate/escalation, Layer 2, head selection, malignant
  probability, risk level, uncertainty.
- Routing / gating / thresholds are **not** re-implemented in the frontend.
- The old `backend/decision_engine/` is **no longer used at runtime**.

---

## 2. File structure

```
backend/
├── app.py                     KEPT (unchanged)   Flask app factory, 16 MB limit, error handlers
├── config.py                  MODIFIED           loads ml_core thresholds.yaml, weights_dir, class_names
├── context.py                 MODIFIED           get_engine(): one lazily-built cached engine
├── decision_engine/           KEPT BUT UNUSED     old routing logic, no longer called at runtime
│   ├── __init__.py
│   └── engine.py
├── routes/
│   ├── health.py              KEPT (unchanged)   GET /health -> {"status": "ok"}
│   ├── model_info.py          MODIFIED           serves engine.model_info() + shell context
│   └── predict.py             MODIFIED           validates image, reads metadata, calls engine
├── services/
│   ├── response_builder.py    MODIFIED           engine result -> API JSON (+ frontend compat)
│   ├── model_service.py       KEPT BUT UNUSED    old RealLayer1 / MockLayer1 adapter
│   └── layer2_stub.py         KEPT BUT UNUSED    old deterministic Layer 2 stub
└── ml_core/                   ADDED              authoritative ML engine (copy of handover + 2 changes)
    ├── __init__.py            ADDED              build_engine(config) -> MockEngine | Engine
    ├── engine.py              ADDED              handover engine.py (relative imports only)
    ├── models.py              ADDED              handover models.py (+ ood.enabled gate in load_all)
    ├── preprocess.py          ADDED              handover preprocess.py (verbatim)
    ├── mock_engine.py         ADDED              deterministic MockEngine (tests / dev only)
    ├── README.md              ADDED              ml_core dev note (incl. OOD status)
    └── config/
        └── thresholds.yaml    ADDED             authoritative ML config (+ ood.enabled: false)

frontend/
├── app.py                     MODIFIED           Streamlit UI: metadata + head select + result rendering
└── app1.py                    (separate/experimental - not part of this integration)

configs/
└── thresholds.yaml            KEPT BUT UNUSED    old 0.85/0.35 config, superseded by ml_core/config

tests/
├── conftest.py                MODIFIED           builds new AppConfig, mock-mode client
├── test_api.py                MODIFIED           new flat API contract
├── test_decision_engine.py    KEPT (unchanged)   still passes; exercises the unused module
├── test_metadata.py           ADDED              age/sex/localization pass-through
├── test_layer2_head.py        ADDED              HAM / HAMPAD selection
├── test_response_builder.py   ADDED              unknown/OOD-shaped result must not crash
└── test_real_pipeline.py      ADDED              real-weights smoke test (auto-skips if no weights)

pytest.ini                     ADDED              testpaths = tests (skip handover/test_pipeline.py)
docs/FRONTEND_BACKEND_INTEGRATION.md   ADDED     this file
docs/INTEGRATION.md            STALE              describes the pre-handover design; needs updating
```

### Added / Modified / Kept-but-unused (summary)

| Status | Files |
|---|---|
| **ADDED** | `backend/ml_core/**`, `pytest.ini`, `tests/test_metadata.py`, `tests/test_layer2_head.py`, `tests/test_response_builder.py`, `tests/test_real_pipeline.py`, this doc |
| **MODIFIED** | `backend/config.py`, `backend/context.py`, `backend/routes/predict.py`, `backend/routes/model_info.py`, `backend/services/response_builder.py`, `frontend/app.py`, `requirements.txt`, `.gitignore`, `tests/conftest.py`, `tests/test_api.py` |
| **KEPT BUT UNUSED** | `backend/decision_engine/**`, `backend/services/model_service.py`, `backend/services/layer2_stub.py`, `configs/thresholds.yaml`, runtime paths in `ml/inference`, `ml/models`, `ml/preprocessing` |
| **UNCHANGED** | `backend/app.py`, `backend/routes/health.py` |

---

## 3. Frontend (`frontend/app.py`)

Streamlit view layer. Inputs:

| Input | Widget | Sent as (form field) |
|---|---|---|
| Age | text input | `age` (string; blank allowed) |
| Sex | selectbox `Male` / `Female` | `sex` = `"male"` / `"female"` |
| Lesion location | selectbox (HAM sites) | `localization` = lowercased site, or `""` |
| Layer 2 head | selectbox `HAMPAD` / `HAM` (default HAMPAD) | `layer2_head` = `"hampad"` / `"ham"` |
| Image | file uploader (jpg/png/bmp/tiff) | `image` (multipart file) |

The **Predict** button is disabled until an image is chosen and Sex + Location
are selected. On submit it `POST`s `multipart/form-data` to `/predict`.

Rendering:

- routing banner (`accepted` / `escalated` / `unknown`)
- mock-mode caption when `meta.mock_mode` is true
- final prediction: class, confidence, entropy, source
- confident / uncertain indicator
- **Risk Assessment**: `malignant_probability`, `risk_level`, `risk_flag`
- advisory text
- per-layer results (Layer 1 always; Layer 2 when escalated)
- **Raw JSON** expander

### The frontend does NOT compute

- confidence thresholds
- entropy thresholds
- escalation decisions
- malignant probability
- risk flag / risk level
- temperature scaling

It only displays values returned by the backend.

---

## 4. Backend

### `backend/app.py` (unchanged)
Flask app factory. `MAX_CONTENT_LENGTH = 16 MB`. Error handlers for
400 / 404 / 405 / 413 / 500 that return `{"success": false, "error": "..."}`
and never leak a traceback. Registers `health`, `model_info`, `predict`
blueprints. Runs on `127.0.0.1:5000` by default.

### `backend/routes/predict.py`
1. Extract the upload (`image` or `file`); empty / missing → `400`.
2. Decode + verify it is a real image (`PIL.Image.open().load()`); bad → `400`.
   Oversized → `413` (handled by `app.py`).
3. `get_engine(app)` (lazy, cached); failure to load → structured `500`.
4. Read `age`, `sex`, `localization`, `layer2_head` from `request.form`
   (blank → `None`).
5. `engine.predict(image, age=..., sex=..., localization=..., layer2_head=...)`;
   any inference exception → structured `500` (no traceback).
6. `build_predict_response(result, config, mock_mode)` → `200`.

Note: a Layer 2 / inference failure returns HTTP `500` (the handover engine has
no `escalated_failed` fallback).

### `backend/context.py`
`get_engine(app)` lazily builds **one** engine via
`backend.ml_core.build_engine(config)` and caches it on `app.extensions`.
Mock mode → `MockEngine`; real mode → `Engine`. No second decision system.

### `backend/services/response_builder.py`
`build_predict_response()` converts the engine's flat result into the API
response. Every field read with `.get()` (the engine's key set is
conditional). Output = handover flat fields **plus** a compatibility block
(`prediction`, `routing`, `meta`) and a `class` alias on `layer1` / `layer2`
so the existing frontend works unchanged. `ood` is included only if the engine
produced it (never while OOD is disabled).

### `backend/routes/model_info.py`
`GET /model-info` returns `config` context (classes, class names, thresholds,
`development_mock_mode`, `checkpoint_available`, `weights_dir`, `ood_enabled`)
plus the engine's own `model_info()` (device, Layer 1 temperature, Layer 2 head
temperatures, default head, OOD status). Stays usable if weights fail to load
(`model_status: "unavailable - engine failed to load"`).

---

## 5. ML core (`backend/ml_core/`)

Single authoritative ML inference implementation. Files `engine.py`,
`models.py`, `preprocess.py`, `config/thresholds.yaml` are a copy of
`handover/amsdds_backend/amsdds/backend/`. **Only two deliberate changes:**

| File | Change | Reason |
|---|---|---|
| `engine.py` | `from preprocess`/`from models` → `from .preprocess`/`from .models` | package imports |
| `models.py` | `load_all()` honours `cfg["ood"]["enabled"]` (default `true`) | OOD on/off switch |
| `config/thresholds.yaml` | added `ood.enabled: false` (+ comment) | disable OOD |

No model architecture, preprocessing, inference maths, gate logic, risk logic,
or Layer 2 behaviour was modified.

| File | Responsibility |
|---|---|
| `engine.py` | Full pipeline: preprocess → Layer 1 → (OOD) → gate → Layer 2 → risk/uncertainty → result dict. Also `model_info()`. |
| `models.py` | Loads all weights once. `Layer1` (multimodal MobileNetV3 + metadata encoding from checkpoint). `Layer2` (frozen RT-DETR encoder + one head per checkpoint). `MahalanobisOOD` (loaded only when `ood.enabled` and the npz exists). |
| `preprocess.py` | `shades_of_gray` colour constancy + per-model resize/crop/normalise (299px Layer 1, 512px Layer 2). |
| `mock_engine.py` | `MockEngine` — deterministic fake predictions matching the real `Engine` output shape. **Tests / `DEVELOPMENT_MOCK_MODE` only.** |
| `config/thresholds.yaml` | Authoritative ML config: classes, class names, weight filenames, image sizes, gate thresholds, risk threshold, OOD `enabled` flag, preprocessing. |
| `README.md` | ml_core dev note, incl. the OOD status. |

---

## 6. Current decision logic (handover rules — do not duplicate in the frontend)

**Layer 1**
- multimodal MobileNetV3-Large, 299px input
- image + metadata (age / sex / localization; missing fields → neutral vector)
- temperature `0.959` (read from the checkpoint, not the YAML)
- 7 HAM10000 classes: `akiec, bcc, bkl, df, mel, nv, vasc`

**Escalation (gate)** — escalate to Layer 2 when:
```
confidence < 0.5   OR   normalized_entropy > 0.3
```

**Layer 2**
- RT-DETR R50vd encoder (frozen) + GOA-tuned MLP head
- heads: `ham` (T ≈ 0.735), `hampad` (T ≈ 0.803)
- default head: `hampad`; selectable per request via `layer2_head`

**Malignant probability**
```
malignant_probability = P(mel) + P(bcc) + P(akiec)
```
Computed from the final layer's probabilities, independent of the argmax label.

**Risk flag**
```
risk_flag = malignant_probability >= 0.26
```

**Risk level** (implemented in `engine.py`)
```
"high"      if risk_flag                       (malignant_probability >= 0.26)
"moderate"  elif malignant_probability >= 0.13
"low"       otherwise
"unknown"   (OOD path only — not reachable while OOD is disabled)
```

**Uncertain**
```
uncertain = final_confidence < 0.5
```

---

## 7. Mock vs real mode

Controlled by the `DEVELOPMENT_MOCK_MODE` environment variable
(read in `backend/config.py`).

| Mode | Engine | Weights | Use |
|---|---|---|---|
| `DEVELOPMENT_MOCK_MODE=true` | `MockEngine` | none | tests, UI development, no torch/transformers needed |
| `DEVELOPMENT_MOCK_MODE=false` (default) | handover `Engine` | 4–5 local files + RT-DETR skeleton | real inference |

`MockEngine` returns deterministic predictions in the same response shape as
the real engine (scenario chosen by `MOCK_SCENARIO=A|B` or by image size
parity). It is **not** clinically meaningful — it only exercises the routing
and rendering paths.

### Real mode (PowerShell, from repo root)

```powershell
$env:DEVELOPMENT_MOCK_MODE = "false"
$env:AMSDDS_WEIGHTS = "$PWD\handover\amsdds_backend\amsdds\weights"   # optional; this is the default location

python -m backend.app          # http://127.0.0.1:5000
```

Then in a second terminal:

```powershell
streamlit run frontend/app.py  # http://localhost:8501
```

Model weights are local / external and **must not be committed**
(`.gitignore` blocks `*.pt`, `*.npz`, `*.npy`, `*.safetensors`,
`handover/**/weights/`).

---

## 8. OOD status — TEMPORARILY DISABLED

**OOD (out-of-distribution / "unknown lesion" detection) is currently disabled.**

```yaml
# backend/ml_core/config/thresholds.yaml
ood:
  enabled: false
```

### Why

`ood_scores_mobilenet.npz` was generated from the **retired image-only 224px
Layer 1**, not the current **multimodal 299px** Layer 1
(`layer1_multimodal_mobilenetv3.pt`). Its stored class means / precision matrix
describe a different 1280-D feature space, so genuine HAM10000 images score a
Mahalanobis distance around 12 000 against a threshold around 3 600 and would
be wrongly returned as `unknown`.

This is **not a model failure**:

- the current multimodal Layer 1 works
- Layer 2 (RT-DETR, HAM + HAMPAD) works
- the confidence / entropy gate works
- malignant probability + risk flag + risk level work

The **OOD artifact** is the only thing that is wrong.

### Re-enabling (later, ML team)

1. Regenerate the OOD statistics from `layer1_multimodal_mobilenetv3.pt` using
   the **same 299px preprocessing and the same 1280-D feature extraction
   point** the engine uses at inference.
2. Drop the corrected `ood_scores_mobilenet.npz` into the weights directory.
3. Set `ood.enabled: true`. No code change is required.

**Do not** work around this by changing the OOD threshold or the Mahalanobis
maths.

While disabled: `MahalanobisOOD` is not constructed, the engine's OOD block is
skipped, responses never contain an `ood` field, and `unknown` is always
`false`.

---

## 9. API contract

### `POST /predict` — `multipart/form-data`

| Field | Required | Notes |
|---|---|---|
| `image` (or `file`) | **yes** | jpg / png / bmp / tiff bytes |
| `age` | no | integer years as string; blank → neutral (mean age). Non-numeric → HTTP 500. |
| `sex` | no | `male` / `female`; anything else → `unknown` |
| `localization` | no | one of the HAM sites; unrecognised → `unknown` |
| `layer2_head` | no | `ham` / `hampad`; unrecognised → default (`hampad`) |

### Response (HTTP 200) — representative normal (escalated) case

```jsonc
{
  "success": true,

  "unknown": false,
  "escalated": true,
  "layer_used": "layer2:hampad",
  "label": "nv",
  "label_name": "Melanocytic nevus",
  "confidence": 0.674,
  "probs": { "akiec": 0.033, "bcc": 0.046, "bkl": 0.005, "df": 0.045,
             "mel": 0.119, "nv": 0.674, "vasc": 0.077 },
  "malignant_probability": 0.199,
  "risk_flag": false,
  "risk_level": "moderate",
  "advisory": "Low risk. Monitor for changes ...",
  "uncertain": false,
  "gate": { "conf": 0.690, "entropy": 0.559,
            "conf_threshold": 0.5, "entropy_threshold": 0.3,
            "reason": "high entropy" },
  "layer1": { "label": "nv", "class": "nv", "label_name": "...",
              "confidence": 0.690, "probs": { ... }, "entropy": 0.559 },
  "layer2": { "label": "nv", "class": "nv", "label_name": "...",
              "confidence": 0.674, "probs": { ... }, "entropy": 0.586,
              "head": "hampad" },
  "latency_ms": { "layer1": 92.2, "layer2": 685.7, "total": 777.9 },
  "metadata_used": true,

  // ---- compatibility block for the existing frontend ----
  "prediction": { "class": "nv", "confidence": 0.674, "entropy": 0.586 },
  "routing":    { "route": "escalated", "final_source": "layer2",
                  "uncertain": false, "notes": [] },
  "meta": { "mock_mode": false,
            "model": { "name": "MobileNetV3-Large multimodal ...",
                       "checkpoint_available": true },
            "thresholds": { "confidence": 0.5, "entropy": 0.3,
                            "uncertain_floor": 0.5, "risk": 0.26 } }
}
```

### Conditional fields

The engine's key set is **conditional**. On the `unknown` / OOD path (not
reachable while OOD is disabled) the response has **no**
`malignant_probability`, top-level `probs`, `gate`, `uncertain`, or `layer2`;
`label` / `confidence` are `null`; `risk_level` is `"unknown"`.
`layer2` is also `null` whenever the request is not escalated. `ood` appears
only when OOD is enabled. Consumers must use `.get()` / null checks.

### Error response

```json
{ "success": false, "error": "<message>" }
```
with HTTP `400` (bad / missing / undecodable image), `413` (> 16 MB), or
`500` (engine unavailable / inference failed). No traceback is returned.

### `GET /model-info`
Config context + `engine.model_info()` (device, temperatures, heads, OOD
status). Works even without weights.

### `GET /health`
`{"status": "ok"}` — never touches the engine.

---

## 10. Running locally (Windows PowerShell)

```powershell
# 1. Repo root
cd C:\path\to\adaptive-multi-layer-skin-disease-detection

# 2. Mock mode (no weights) ...
$env:DEVELOPMENT_MOCK_MODE = "true"
# ... or real mode:
# $env:DEVELOPMENT_MOCK_MODE = "false"

# 3. Weights location (real mode; optional - this is the default)
$env:AMSDDS_WEIGHTS = "$PWD\handover\amsdds_backend\amsdds\weights"

# 4. Start Flask
python -m backend.app                 # http://127.0.0.1:5000

# 5. Second terminal, repo root
# 6. Start Streamlit
streamlit run frontend/app.py         # http://localhost:8501
```

7. Open `http://localhost:8501`.
8. Upload a HAM10000 image (jpg/png).
9. Enter age, select Sex and Lesion Location.
10. Select the Layer 2 head (HAMPAD default).
11. Click **Predict**.

First **real** Layer 2 startup downloads the `PekingU/rtdetr_r50vd` skeleton
from the HuggingFace Hub — needs network access, or a warm
`~/.cache/huggingface`, on that first run.

---

## 11. Testing

```
python -m pytest -q      ->  33 passed
```

| Test file | Covers |
|---|---|
| `tests/test_api.py` | `/health`, `/model-info`, `/predict` validation + flat contract + structured 500 |
| `tests/test_metadata.py` | `age` / `sex` / `localization` pass-through, `metadata_used` |
| `tests/test_layer2_head.py` | HAMPAD default, explicit HAM / HAMPAD, unknown → default |
| `tests/test_response_builder.py` | `unknown`/OOD-shaped result does not crash the builder |
| `tests/test_decision_engine.py` | legacy `backend.decision_engine` unit tests (module unused at runtime; still green) |
| `tests/test_real_pipeline.py` | real-weights smoke: OOD disabled at runtime, both heads loaded, real prediction shape, head selection. **Auto-skips if weights are absent.** |

`pytest.ini` sets `testpaths = tests` so the handover's own
`test_pipeline.py` (a manual script that loads weights at import) is not
collected.

Mock mode exists partly so application/API tests run without loading torch,
transformers, or the model weights.

---

## 12. Status / TODO

### Completed
- [x] handover ML engine integrated as `backend/ml_core/`
- [x] one authoritative engine, lazily loaded and cached
- [x] real multimodal Layer 1 (299px, image + metadata, T = 0.959)
- [x] real Layer 2 (RT-DETR) with HAM / HAMPAD head selection (default HAMPAD)
- [x] confidence / entropy gate (0.5 / 0.3)
- [x] malignant probability + risk flag (0.26) + risk level
- [x] `age` / `sex` / `localization` metadata input end to end
- [x] API contract (flat fields + frontend compatibility block)
- [x] Streamlit UI integration
- [x] mock mode preserved for tests
- [x] 33 tests passing

### Pending
- [ ] corrected OOD artifact from `layer1_multimodal_mobilenetv3.pt` (299px)
- [ ] re-enable OOD (`ood.enabled: true`) after that artifact is validated
- [ ] delete the now-unused `backend/decision_engine/`,
      `backend/services/model_service.py`, `backend/services/layer2_stub.py`,
      `configs/thresholds.yaml` (optional cleanup)
- [ ] update / replace the stale `docs/INTEGRATION.md`
- [ ] frontend polish (numeric `age` field, `unknown`-state UI for when OOD
      is re-enabled)

**OOD is not currently working — it is intentionally disabled.**

---

## 13. Development rules

- Do **not** commit model weights (or `*.npz` / `*.safetensors`).
- Do **not** re-enable the current OOD artifact; do **not** "fix" it by
  changing the OOD threshold.
- Do **not** duplicate ML decision logic (gate, entropy, temperature,
  malignant probability, risk) in the frontend.
- Keep `backend/ml_core/` as the single authoritative ML engine — no parallel
  decision system.
- Keep `DEVELOPMENT_MOCK_MODE` / `MockEngine` working for tests and UI dev.
- Do **not** modify the trained model architecture or the handover inference
  maths without an explicit ML-side change.
