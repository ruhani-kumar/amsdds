"""Real handover pipeline smoke test.

SKIPPED unless the RT-DETR weight files are present (both backbones' weights
ship together during the PanDerm transition — see backend/ml_core/README.md).
Exercises the actual multimodal Layer 1 + whichever Layer 2 backbone is
selected by ``layer2.backbone`` in thresholds.yaml (default: PanDerm, a single
self-contained BEiT ViT-B/16 checkpoint — no torch/transformers Hub call
needed; the RT-DETR fallback additionally needs a warm HuggingFace cache for
``PekingU/rtdetr_r50vd``).

NOTE (pre-existing, unrelated to the PanDerm swap): ``ood.enabled`` in
thresholds.yaml is currently ``true`` even though the surrounding comments and
backend/ml_core/README.md document it as intentionally disabled (the shipped
ood_scores_mobilenet.npz was fit on a retired Layer 1 and rejects genuine
images). ``test_ood_is_disabled`` and the ``unknown`` assertion in
``test_real_prediction_shape`` will fail until that config flag is corrected;
this is a Layer 1/OOD config issue, not a Layer 2 one, and is intentionally
left as-is here.
"""
from __future__ import annotations

import os

import pytest
from PIL import Image

from backend.config import DEFAULT_THRESHOLDS_PATH, DEFAULT_WEIGHTS_DIR

_REQUIRED = [
    "layer1_multimodal_mobilenetv3.pt",
    "layer2_rtdetr.pt",
    "layer2_rtdetr_goa_head.pt",
    "layer2_rtdetr_goa_head_hampad.pt",
]
_WEIGHTS_DIR = os.environ.get("AMSDDS_WEIGHTS", str(DEFAULT_WEIGHTS_DIR))
_HAVE_WEIGHTS = all(
    os.path.isfile(os.path.join(_WEIGHTS_DIR, f)) for f in _REQUIRED
)

pytestmark = pytest.mark.skipif(
    not _HAVE_WEIGHTS, reason=f"real weights not present in {_WEIGHTS_DIR}"
)


@pytest.fixture(scope="module")
def engine():
    os.environ.setdefault("AMSDDS_WEIGHTS", _WEIGHTS_DIR)
    from backend.ml_core.engine import Engine

    return Engine(str(DEFAULT_THRESHOLDS_PATH))


def test_ood_is_disabled(engine):
    info = engine.model_info()
    assert info["runtime"]["ood_enabled"] is False
    assert info["runtime"]["ood_threshold"] is None


def test_layer2_heads_loaded(engine):
    # Layer 2 backbone is selectable (config: layer2.backbone). Current default
    # is PanDerm (single self-contained checkpoint, no ham/hampad heads); the
    # RT-DETR dual-head encoder is kept as a fallback/reference (see
    # backend/ml_core/README.md and models.PanDermLayer2's docstring).
    backbone = engine.cfg["layer2"]["backbone"]
    if backbone == "panderm":
        assert set(engine.l2.heads) == {"panderm"}
        assert engine.l2.default == "panderm"
    else:
        assert set(engine.l2.heads) == {"ham", "hampad"}
        assert engine.l2.default == "hampad"


def test_real_prediction_shape(engine):
    img_path = os.environ.get("AMSDDS_SMOKE_IMAGE")
    src = (
        img_path
        if img_path and os.path.isfile(img_path)
        else Image.new("RGB", (600, 450), (170, 120, 110))
    )
    out = engine.predict(src)

    assert out["unknown"] is False          # OOD disabled -> never "unknown"
    assert "gate" in out
    assert "malignant_probability" in out
    assert out["label"] in engine.classes
    assert out["risk_level"] in {"low", "moderate", "high"}
    assert set(out["probs"]) == set(engine.classes)
    assert isinstance(out["latency_ms"]["total"], float)


def test_layer2_head_selection(engine):
    src = Image.new("RGB", (600, 450), (60, 40, 40))  # low-info -> likely escalates
    candidate_heads = tuple(engine.l2.heads)   # {"panderm"} or {"ham", "hampad"}
    for head in candidate_heads:
        out = engine.predict(src, layer2_head=head)
        if out["escalated"]:
            assert out["layer2"]["head"] == head
            assert out["layer_used"] == f"layer2:{head}"
