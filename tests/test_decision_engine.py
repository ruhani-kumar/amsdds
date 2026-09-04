"""Decision Engine unit tests. No Flask, no model, no torch."""
from __future__ import annotations

import pytest

from backend.decision_engine import DecisionEngine


def make_engine(config, layer2):
    return DecisionEngine(config=config, layer2=layer2)


def l2_confident(_l1):
    return {"class": "mel", "confidence": 0.90, "entropy": 0.20}


def l2_unconfident(_l1):
    return {"class": "mel", "confidence": 0.40, "entropy": 0.70}


def l2_broken(_l1):
    raise RuntimeError("layer 2 down")


# 1. High confidence + low entropy -> accepted
def test_accept_high_conf_low_entropy(engine_config):
    engine = make_engine(engine_config, l2_confident)
    r = engine.decide({"class": "nv", "confidence": 0.95, "entropy": 0.10})
    assert r.route == "accepted"
    assert r.final_source == "layer1"
    assert r.uncertain is False
    assert r.layer2 is None
    assert r.predicted_class == "nv"


# 2. Low confidence -> escalated
def test_escalate_low_conf(engine_config):
    engine = make_engine(engine_config, l2_confident)
    r = engine.decide({"class": "nv", "confidence": 0.62, "entropy": 0.20})
    assert r.route == "escalated"
    assert r.final_source == "layer2"
    assert r.layer2 == {"class": "mel", "confidence": 0.90, "entropy": 0.20}


# 3. High confidence + high entropy -> escalated
def test_escalate_high_entropy(engine_config):
    engine = make_engine(engine_config, l2_confident)
    r = engine.decide({"class": "nv", "confidence": 0.97, "entropy": 0.55})
    assert r.route == "escalated"
    assert r.final_source == "layer2"


# 4. Escalated, Layer 2 confidence >= floor -> uncertain False
def test_escalated_layer2_confident(engine_config):
    engine = make_engine(engine_config, l2_confident)
    r = engine.decide({"class": "nv", "confidence": 0.50, "entropy": 0.50})
    assert r.route == "escalated"
    assert r.uncertain is False


# 5. Escalated, Layer 2 confidence < floor -> uncertain True
def test_escalated_layer2_unconfident(engine_config):
    engine = make_engine(engine_config, l2_unconfident)
    r = engine.decide({"class": "nv", "confidence": 0.50, "entropy": 0.50})
    assert r.route == "escalated"
    assert r.uncertain is True
    assert r.predicted_class == "mel"


# 6. Layer 2 failure -> escalated_failed, Layer 1 preserved
def test_layer2_failure_falls_back(engine_config):
    engine = make_engine(engine_config, l2_broken)
    r = engine.decide({"class": "bkl", "confidence": 0.40, "entropy": 0.80})
    assert r.route == "escalated_failed"
    assert r.final_source == "layer1"
    assert r.uncertain is True
    assert r.layer2 is None
    assert r.predicted_class == "bkl"
    assert r.confidence == 0.40
    assert r.layer1 == {"class": "bkl", "confidence": 0.40, "entropy": 0.80}


# 7. Boundary: confidence exactly 0.85 and entropy exactly 0.35 -> accepted
def test_boundary_exact_thresholds_accept(engine_config):
    engine = make_engine(engine_config, l2_confident)
    r = engine.decide({"class": "nv", "confidence": 0.85, "entropy": 0.35})
    assert r.route == "accepted"
    assert r.final_source == "layer1"


@pytest.mark.parametrize(
    "conf,ent",
    [(0.8499, 0.35), (0.85, 0.3501)],
)
def test_boundary_just_outside_escalates(engine_config, conf, ent):
    engine = make_engine(engine_config, l2_confident)
    r = engine.decide({"class": "nv", "confidence": conf, "entropy": ent})
    assert r.route == "escalated"
