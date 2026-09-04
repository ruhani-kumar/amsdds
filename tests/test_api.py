"""API tests. Backend runs in DEVELOPMENT_MOCK_MODE - no weights / torch /
transformers. The handover engine is swapped for MockEngine."""
from __future__ import annotations

import dataclasses
import io

from backend.app import create_app


def _client_with_scenario(app_config, scenario):
    cfg = dataclasses.replace(app_config, mock_scenario=scenario)
    app = create_app(cfg)
    app.testing = True
    return app.test_client()


def _post_image(client, png_bytes, **form):
    data = {"image": (io.BytesIO(png_bytes), "lesion.png")}
    data.update(form)
    return client.post("/predict", data=data, content_type="multipart/form-data")


# 1. GET /health -> 200 (contract unchanged)
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


# 2. GET /model-info -> 200, mock mode, OOD disabled, handover thresholds
def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.get_json()
    assert body["development_mock_mode"] is True
    assert body["ood_enabled"] is False
    assert body["classes"] == ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    assert body["thresholds"]["confidence"] == 0.5
    assert body["thresholds"]["entropy"] == 0.3
    assert body["thresholds"]["risk"] == 0.26
    assert body["model_status"] == "mock"


# 3. POST /predict without an image -> 400
def test_predict_no_image(client):
    r = client.post("/predict")
    assert r.status_code == 400
    assert r.get_json()["success"] is False


# 4. POST /predict with a non-image file -> 400
def test_predict_invalid_file(client):
    data = {"image": (io.BytesIO(b"not an image at all"), "notes.txt")}
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["success"] is False


# 5. POST /predict in mock mode -> 200 + new flat contract + compat block
def test_predict_mock_ok(client, png_bytes):
    r = _post_image(client, png_bytes)
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True
    for key in (
        "unknown", "escalated", "layer_used", "label", "label_name",
        "confidence", "probs", "malignant_probability", "risk_flag",
        "risk_level", "advisory", "uncertain", "gate", "layer1",
        "latency_ms", "metadata_used",
    ):
        assert key in body, f"missing top-level field: {key}"
    assert body["unknown"] is False
    assert body["label"] in {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    assert 0.0 <= body["malignant_probability"] <= 1.0
    assert body["risk_level"] in {"low", "moderate", "high"}
    assert set(body["probs"]) == {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    # compat block for the existing Streamlit frontend
    assert body["prediction"]["class"] == body["label"]
    assert body["routing"]["route"] in {"accepted", "escalated"}
    assert body["meta"]["mock_mode"] is True
    assert body["meta"]["thresholds"]["confidence"] == 0.5
    # OOD disabled -> no `ood` block
    assert "ood" not in body


# 6. Forced scenarios exercise both routing paths.
def test_predict_scenario_a_accepted(app_config, png_bytes):
    client = _client_with_scenario(app_config, "A")
    body = _post_image(client, png_bytes).get_json()
    assert body["escalated"] is False
    assert body["layer_used"] == "layer1"
    assert body["layer2"] is None
    assert body["routing"]["route"] == "accepted"
    assert body["routing"]["final_source"] == "layer1"


def test_predict_scenario_b_escalated(app_config, png_bytes):
    client = _client_with_scenario(app_config, "B")
    body = _post_image(client, png_bytes).get_json()
    assert body["escalated"] is True
    assert body["layer_used"].startswith("layer2:")
    assert body["layer2"] is not None
    assert body["layer2"]["head"] in {"ham", "hampad"}
    assert body["routing"]["route"] == "escalated"
    assert body["routing"]["final_source"] == "layer2"


# 7. Engine inference failure -> structured HTTP 500, no internal detail leaked.
def test_predict_engine_error(app_config, png_bytes):
    app = create_app(app_config)
    app.testing = True

    class _Boom:
        is_mock = True

        def predict(self, *a, **k):
            raise RuntimeError("internal boom detail")

    app.extensions.setdefault("amsdds", {})["engine"] = _Boom()

    r = app.test_client().post(
        "/predict",
        data={"image": (io.BytesIO(png_bytes), "b.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 500
    body = r.get_json()
    assert body["success"] is False
    assert "error" in body
    assert "boom" not in body["error"]
