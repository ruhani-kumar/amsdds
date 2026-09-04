"""Metadata pass-through to the engine (mock mode).

age / sex / localization -> metadata_used == True.
layer2_head alone is NOT metadata.
"""
from __future__ import annotations

import io

import pytest


def _predict(client, png_bytes, **form):
    data = {"image": (io.BytesIO(png_bytes), "lesion.png")}
    data.update(form)
    return client.post(
        "/predict", data=data, content_type="multipart/form-data"
    ).get_json()


def test_no_metadata(client, png_bytes):
    body = _predict(client, png_bytes)
    assert body["success"] is True
    assert body["metadata_used"] is False


@pytest.mark.parametrize(
    "form",
    [
        {"age": "45"},
        {"sex": "female"},
        {"localization": "back"},
        {"age": "62", "sex": "male", "localization": "face"},
    ],
)
def test_metadata_variants(client, png_bytes, form):
    body = _predict(client, png_bytes, **form)
    assert body["success"] is True
    assert body["metadata_used"] is True


def test_layer2_head_is_not_metadata(client, png_bytes):
    body = _predict(client, png_bytes, layer2_head="ham")
    assert body["success"] is True
    assert body["metadata_used"] is False
