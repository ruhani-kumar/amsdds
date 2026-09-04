from __future__ import annotations

import io

from flask import Blueprint, current_app, jsonify, request
from PIL import Image, UnidentifiedImageError

from backend.context import get_config, get_engine
from backend.services import build_predict_response, error_response

predict_bp = Blueprint("predict", __name__)

_IMAGE_FIELDS = ("image", "file")
_META_FIELDS = ("age", "sex", "localization", "layer2_head")


def _err(message: str, status: int):
    body, code = error_response(message, status)
    return jsonify(body), code


def _extract_upload():
    for name in _IMAGE_FIELDS:
        if name in request.files and request.files[name].filename:
            return request.files[name]
    return None


@predict_bp.post("/predict")
def predict():
    # 1-3. Validate input.
    upload = _extract_upload()
    if upload is None:
        return _err(
            "No image uploaded. Send a multipart form field named 'image'.", 400
        )

    raw = upload.read()
    if not raw:
        return _err("Uploaded file is empty.", 400)

    # 4. Decode to PIL, verifying it is a real image.
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return _err("Uploaded file could not be read as an image.", 400)

    config = get_config(current_app)

    # 5. ML engine (lazy-loaded, cached). Missing weights -> 500.
    try:
        engine = get_engine(current_app)
    except Exception:  # noqa: BLE001
        current_app.logger.exception("ML engine unavailable")
        return _err(
            "ML engine is unavailable (model weights missing or failed to load).",
            500,
        )

    # 6. Optional metadata + Layer 2 head selection (blank -> None).
    meta = {name: (request.form.get(name) or None) for name in _META_FIELDS}

    # 7. Full pipeline. Any inference failure -> structured 500 (no traceback).
    try:
        result = engine.predict(
            image,
            age=meta["age"],
            sex=meta["sex"],
            localization=meta["localization"],
            layer2_head=meta["layer2_head"],
        )
    except Exception:  # noqa: BLE001
        current_app.logger.exception("ML engine inference failed")
        return _err("Inference failed while processing the image.", 500)

    # 8. Standardised response.
    response = build_predict_response(
        result, config, mock_mode=getattr(engine, "is_mock", False)
    )
    return jsonify(response), 200
