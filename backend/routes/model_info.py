from flask import Blueprint, current_app, jsonify

from backend.config import LAYER1_MODEL_NAME
from backend.context import get_config, get_engine

model_info_bp = Blueprint("model_info", __name__)


@model_info_bp.get("/model-info")
def model_info():
    config = get_config(current_app)
    de = config.decision_engine

    body = {
        "layer1_model": LAYER1_MODEL_NAME,
        "classes": config.classes,
        "malignant_classes": config.malignant_classes,
        "class_names": config.class_names,
        "development_mock_mode": config.mock_mode,
        "checkpoint_available": config.checkpoint_exists,
        "weights_dir": str(config.weights_dir),
        "thresholds": {
            "confidence": de.conf_threshold,
            "entropy": de.entropy_threshold,
            "uncertain_floor": de.uncertain_floor,
            "risk": de.risk_threshold,
        },
        # OOD is temporarily disabled - see backend/ml_core/README.md.
        # Overwritten below with the engine's authoritative value when available.
        "ood_enabled": False,
    }

    try:
        engine = get_engine(current_app)
        info = engine.model_info()
        body["engine"] = info
        body["model_status"] = (
            "mock" if getattr(engine, "is_mock", False) else "available"
        )
        body["ood_enabled"] = bool(
            info.get("runtime", {}).get("ood_enabled", False)
        )
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("model-info: ML engine unavailable")
        body["model_status"] = "unavailable - engine failed to load"
        body["engine_error"] = str(exc)

    return jsonify(body)
