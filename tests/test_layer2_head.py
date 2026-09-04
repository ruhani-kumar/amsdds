"""layer2_head selection (mock mode, forced escalation scenario B)."""
from __future__ import annotations

import dataclasses
import io

import pytest

from backend.app import create_app


@pytest.fixture
def client_b(app_config):
    cfg = dataclasses.replace(app_config, mock_scenario="B")
    app = create_app(cfg)
    app.testing = True
    return app.test_client()


def _predict(client, png_bytes, **form):
    data = {"image": (io.BytesIO(png_bytes), "lesion.png")}
    data.update(form)
    return client.post(
        "/predict", data=data, content_type="multipart/form-data"
    ).get_json()


def test_default_head_is_hampad(client_b, png_bytes):
    body = _predict(client_b, png_bytes)
    assert body["escalated"] is True
    assert body["layer_used"] == "layer2:hampad"
    assert body["layer2"]["head"] == "hampad"


def test_explicit_ham(client_b, png_bytes):
    body = _predict(client_b, png_bytes, layer2_head="ham")
    assert body["layer_used"] == "layer2:ham"
    assert body["layer2"]["head"] == "ham"


def test_explicit_hampad(client_b, png_bytes):
    body = _predict(client_b, png_bytes, layer2_head="hampad")
    assert body["layer_used"] == "layer2:hampad"
    assert body["layer2"]["head"] == "hampad"


def test_unknown_head_falls_back_to_default(client_b, png_bytes):
    body = _predict(client_b, png_bytes, layer2_head="bogus")
    assert body["layer_used"] == "layer2:hampad"
    assert body["layer2"]["head"] == "hampad"
