from __future__ import annotations

from backend.config import AppConfig

_LAYER1_NAME = (
    "MobileNetV3-Large multimodal (image + age/sex/localization, HAM10000 7-class)"
)


def _with_class_alias(layer: dict | None) -> dict | None:
    """Add a ``class`` key mirroring ``label`` (the current frontend reads
    ``class``; the handover engine emits ``label``)."""
    if not layer:
        return None
    out = dict(layer)
    if "label" in out and "class" not in out:
        out["class"] = out["label"]
    return out


def build_predict_response(
    result: dict, config: AppConfig, mock_mode: bool
) -> dict:
    unknown = bool(result.get("unknown", False))
    escalated = bool(result.get("escalated", False))
    layer_used = result.get("layer_used", "layer1")
    final_source = "layer2" if str(layer_used).startswith("layer2") else "layer1"
    route = "unknown" if unknown else ("escalated" if escalated else "accepted")

    # entropy for the compat `prediction` block: from the layer that produced
    # the final answer (layer2 if escalated, else layer1).
    final_layer = result.get("layer2") or result.get("layer1") or {}
    final_entropy = final_layer.get("entropy")

    de = config.decision_engine

    response = {
        "success": True,

        # ---- handover flat result (pass-through, conditional keys) ----
        "unknown": unknown,
        "escalated": escalated,
        "layer_used": layer_used,
        "label": result.get("label"),
        "label_name": result.get("label_name"),
        "confidence": result.get("confidence"),
        "probs": result.get("probs"),
        "malignant_probability": result.get("malignant_probability"),
        "risk_flag": result.get("risk_flag"),
        "risk_level": result.get("risk_level"),
        "advisory": result.get("advisory"),
        "uncertain": result.get("uncertain"),
        "gate": result.get("gate"),
        "layer1": _with_class_alias(result.get("layer1")),
        "layer2": _with_class_alias(result.get("layer2")),
        "latency_ms": result.get("latency_ms"),
        "metadata_used": result.get("metadata_used", False),

        # ---- compatibility block for the existing Streamlit frontend ----
        "prediction": {
            "class": result.get("label") or "unknown",
            "confidence": result.get("confidence") or 0.0,
            "entropy": final_entropy if final_entropy is not None else 0.0,
        },
        "routing": {
            "route": route,
            "final_source": final_source,
            "uncertain": bool(result.get("uncertain", False)),
            "notes": [],
        },
        "meta": {
            "mock_mode": mock_mode,
            "model": {
                "name": _LAYER1_NAME,
                "checkpoint_available": config.checkpoint_exists,
            },
            "thresholds": {
                "confidence": de.conf_threshold,
                "entropy": de.entropy_threshold,
                "uncertain_floor": de.uncertain_floor,
                "risk": de.risk_threshold,
            },
        },
    }

    # Only surface `ood` when the engine actually produced it (i.e. never while
    # OOD is disabled).
    if "ood" in result:
        response["ood"] = result["ood"]

    return response


def error_response(message: str, status: int) -> tuple[dict, int]:
    return {"success": False, "error": message}, status
