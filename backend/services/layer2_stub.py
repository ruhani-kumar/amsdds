from __future__ import annotations

from typing import Mapping
_STUB_CLASS = "mel"
_STUB_CONFIDENCE = 0.90
_STUB_ENTROPY = 0.20


def layer2_stub(layer1_result: Mapping) -> dict:
    """Deterministic mock Layer 2 prediction."""
    return {
        "class": _STUB_CLASS,
        "confidence": _STUB_CONFIDENCE,
        "entropy": _STUB_ENTROPY,
        "source": "layer2_stub",
    }


def failing_layer2_stub(layer1_result: Mapping) -> dict:
    """Layer 2 that always fails - used to demo the escalated_failed path."""
    raise RuntimeError("Layer 2 (RT-DETR) is not available in this build.")
