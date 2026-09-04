# Integration skeleton — Streamlit ▸ Flask ▸ Layer 1 ▸ Decision Engine ▸ Layer 2 stub

Minimal MVP wiring around the **existing** Layer 1 implementation
(`ml.inference.Layer1Model`). No ML code was rewritten. The real Layer 2
(RT-DETR) is a deterministic stub.

```
Streamlit (frontend/app.py)
   │  POST /predict  (multipart image)
   ▼
Flask (backend/app.py)
   │  validate → PIL.Image
   ▼
Layer 1  ml.inference.Layer1Model.predict_full()      ← owns colour constancy,
   │  (label, confidence, entropy, probs)                eval transform, MobileNetV3,
   ▼                                                     temperature scaling
Decision Engine  backend/decision_engine/engine.py
   │  confidence ≥ 0.85 AND entropy ≤ 0.35 ?
   ├── yes → route=accepted,  final_source=layer1
   └── no  → call Layer 2 stub (backend/services/layer2_stub.py)
              ├── ok   → route=escalated,        final_source=layer2
              │           uncertain = l2.confidence < 0.50
              └── raise → route=escalated_failed, final_source=layer1,
                          uncertain=true, Layer 1 result preserved
   ▼
Standardised JSON  (backend/services/response_builder.py)
   ▼
Streamlit renders routing.route / final_source / uncertain  (no threshold logic)
```

Thresholds come **only** from `configs/thresholds.yaml` (`conf_threshold 0.85`,
`entropy_threshold 0.35`, `uncertain_floor 0.5`, `temperature 0.9718`). They are
never hardcoded in the backend or frontend.

## Run it locally

```bash
pip install -r requirements.txt          # mock-mode deps only; torch not required

# Terminal 1 — backend
DEVELOPMENT_MOCK_MODE=true python -m backend.app          # http://127.0.0.1:5000

# Terminal 2 — frontend
streamlit run frontend/app.py                             # http://localhost:8501
```

Windows PowerShell equivalent for the backend:
```powershell
$env:DEVELOPMENT_MOCK_MODE = "true"; python -m backend.app
```

Or copy `.env.example` → `.env` and set `DEVELOPMENT_MOCK_MODE=true` there
(the backend auto-loads `.env`).

## Endpoints

| method | path | notes |
|---|---|---|
| GET | `/health` | `{"status":"ok"}`. Never touches the checkpoint. |
| GET | `/model-info` | classes, thresholds, temperature, mock flag, checkpoint availability. Works with no checkpoint. |
| POST | `/predict` | multipart field `image`. Returns the standardised JSON below. |

### `/predict` response contract

```jsonc
{
  "success": true,
  "prediction": { "class": "mel", "confidence": 0.90, "entropy": 0.20 },
  "routing":    { "route": "escalated", "final_source": "layer2",
                  "uncertain": false, "notes": ["..."] },
  "layer1":     { "class": "mel", "confidence": 0.65, "entropy": 0.45 },
  "layer2":     { "class": "mel", "confidence": 0.90, "entropy": 0.20 },
  "meta": {
    "mock_mode": true,
    "model": { "name": "MobileNetV3-Large (fine-tuned, HAM10000 7-class)",
               "checkpoint_available": false },
    "thresholds": { "confidence": 0.85, "entropy": 0.35, "uncertain_floor": 0.5 }
  }
}
```

`route` ∈ `accepted` | `escalated` | `escalated_failed`.
`layer2` is `null` for `accepted` and `escalated_failed`.

Errors: `{"success": false, "error": "<message>"}` with HTTP 400 (bad/missing
image), 500 (Layer 1 unavailable / inference failed). Stack traces are never
returned. A **Layer 2** failure is **not** a 500 — it becomes
`route=escalated_failed` with HTTP 200.

## Development mock mode

`DEVELOPMENT_MOCK_MODE=true` makes the backend serve deterministic fake Layer 1
predictions so the whole pipeline runs without `layer1_mobilenetv3_best.pt` or
torch. It is **backend-only** config — the frontend has no idea. Default is
`false` (real model).

| scenario | Layer 1 output | route |
|---|---|---|
| A | `nv`, conf 0.92, entropy 0.20 | accepted |
| B | `mel`, conf 0.65, entropy 0.45 | escalated → Layer 2 stub |

Scenario is chosen deterministically from image dimensions (`w*h` parity), or
pinned with `MOCK_SCENARIO=A` / `MOCK_SCENARIO=B`.

## Configuring the real checkpoint

```bash
export MODEL_CHECKPOINT_PATH=models/layer1_mobilenetv3_best.pt   # default; relative to repo root
export DEVELOPMENT_MOCK_MODE=false
pip install torch torchvision numpy scikit-learn pandas
```

Download `layer1_mobilenetv3_best.pt` from the Drive link in `README.md` and put
it at that path. It is **not** committed (`.gitignore` blocks `*.pt`). If it is
missing in real mode, `/predict` returns HTTP 500 with an actionable message and
`/model-info` reports `model_status: "unavailable - checkpoint not found"`.

## What remains to plug in when the checkpoint arrives

1. Drop `layer1_mobilenetv3_best.pt` at `MODEL_CHECKPOINT_PATH`.
2. `pip install torch torchvision numpy` (+ `scikit-learn`/`pandas` only if you
   also run training/eval code).
3. Set `DEVELOPMENT_MOCK_MODE=false`.

No code changes. `backend/services/model_service.py::RealLayer1` already calls
`ml.inference.Layer1Model(...).predict_full(pil_image)` and nothing else.

To swap the Layer 2 stub for real RT-DETR later: replace
`backend/services/layer2_stub.py::layer2_stub` with a function of the same
signature (`(layer1_result_dict) -> {"class","confidence","entropy"}`). The
Decision Engine and response contract stay as-is.

## Tests

```bash
python -m pytest -q          # 17 tests, no checkpoint / torch / GPU needed
```

`tests/test_decision_engine.py` — routing policy (7 cases incl. exact-threshold
boundary + Layer 2 failure fallback). `tests/test_api.py` — `/health`,
`/model-info`, `/predict` (missing image, bad file, mock success, both routing
paths, Layer 2 failure → `escalated_failed`).

## Files owned by this skeleton (do not confuse with Layer 1)

```
backend/            config, app factory, routes, decision engine, services
frontend/app.py     Streamlit view layer
tests/              decision engine + API tests
docs/INTEGRATION.md this file
.env.example        configuration template
```

`ml/` is the existing Layer 1 implementation — only `ml/inference/__init__.py`
and `ml/models/__init__.py` were touched, to fix broken re-exports so
`from ml.inference import Layer1Model` works as the handoff documents.
