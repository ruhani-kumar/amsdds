"""response_builder must tolerate the handover engine's CONDITIONAL output.

An OOD / ``unknown`` result has no ``malignant_probability``, ``probs``,
``gate``, ``uncertain`` or ``layer2``. Building a response from it must not
raise (regression guard for the KeyError we hit in the handover test script).
"""
from __future__ import annotations

from pathlib import Path

from backend.config import AppConfig, DecisionEngineConfig
from backend.services.response_builder import build_predict_response

_CFG = AppConfig(
    checkpoint_path=Path("does-not-exist.pt"),
    thresholds_path=Path("does-not-exist.yaml"),
    weights_dir=Path("."),
    device="cpu",
    mock_mode=True,
    mock_scenario=None,
    decision_engine=DecisionEngineConfig(0.5, 0.3, 0.5, 0.26),
    classes=["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    malignant_classes=["mel", "bcc", "akiec"],
    class_names={},
    temperature=0.959,
)

# Exactly the shape Engine.predict() returns on the OOD early-return path.
_UNKNOWN = {
    "escalated": False,
    "unknown": True,
    "layer_used": "layer1",
    "metadata_used": False,
    "layer1": {
        "label": "nv", "label_name": "Melanocytic nevus",
        "confidence": 0.81, "probs": {}, "entropy": 0.36,
    },
    "ood": {"score": 12000.0, "threshold": 3600.0, "is_ood": True},
    "label": None,
    "label_name": "Not a recognised skin lesion",
    "confidence": None,
    "risk_level": "unknown",
    "risk_flag": False,
    "advisory": "Image not recognised as a skin lesion.",
    "latency_ms": {"layer1": 90.0, "total": 100.0},
}

_NORMAL_ACCEPTED = {
    "escalated": False,
    "unknown": False,
    "layer_used": "layer1",
    "metadata_used": True,
    "layer1": {
        "label": "nv", "label_name": "Melanocytic nevus",
        "confidence": 0.93, "probs": {"nv": 0.93}, "entropy": 0.12,
    },
    "gate": {
        "conf": 0.93, "entropy": 0.12, "conf_threshold": 0.5,
        "entropy_threshold": 0.3, "reason": "confident",
    },
    "label": "nv", "label_name": "Melanocytic nevus", "confidence": 0.93,
    "probs": {"nv": 0.93}, "malignant_probability": 0.02, "risk_flag": False,
    "risk_level": "low", "uncertain": False, "advisory": "Low risk.",
    "latency_ms": {"layer1": 88.0, "layer2": 0.0, "total": 90.0},
}


def test_unknown_shape_does_not_crash():
    r = build_predict_response(_UNKNOWN, _CFG, mock_mode=False)
    assert r["success"] is True
    assert r["unknown"] is True
    assert r["malignant_probability"] is None     # absent in source -> None, not KeyError
    assert r.get("gate") is None
    assert r["layer2"] is None
    assert r["routing"]["route"] == "unknown"
    assert r["prediction"]["confidence"] == 0.0   # compat block still serialisable
    assert "ood" in r


def test_normal_shape_passthrough():
    r = build_predict_response(_NORMAL_ACCEPTED, _CFG, mock_mode=True)
    assert r["success"] is True
    assert r["malignant_probability"] == 0.02
    assert r["risk_level"] == "low"
    assert r["routing"]["route"] == "accepted"
    assert r["layer1"]["class"] == "nv"           # `class` alias for the frontend
    assert "ood" not in r
