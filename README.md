# adaptive-multi-layer-skin-disease-detection
Adaptive multi-layer skin disease detection using confidence-aware model routing.
# Drive link for model weights etc etc etc
https://drive.google.com/drive/folders/1DliZmfRP3U7tmsIn4AqsWncU0rjjLtTE?usp=sharing

## Running the integration skeleton

Streamlit ▸ Flask ▸ existing Layer 1 (`ml.inference.Layer1Model`) ▸ Decision
Engine ▸ Layer 2 stub. Full details in [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

```bash
pip install -r requirements.txt

# Terminal 1 — backend (mock mode: no checkpoint / torch needed)
DEVELOPMENT_MOCK_MODE=true python -m backend.app     # http://127.0.0.1:5000

# Terminal 2 — frontend
streamlit run frontend/app.py                        # http://localhost:8501

# tests
python -m pytest -q
```

- **Real model:** put `layer1_mobilenetv3_best.pt` (from the Drive link above) at
  `MODEL_CHECKPOINT_PATH` (default `models/layer1_mobilenetv3_best.pt`),
  `pip install torch torchvision numpy`, and set `DEVELOPMENT_MOCK_MODE=false`.
  The checkpoint is never committed (`.gitignore` blocks `*.pt`).
- Thresholds live only in `configs/thresholds.yaml`.
- Config via environment or a `.env` file — see `.env.example`.
