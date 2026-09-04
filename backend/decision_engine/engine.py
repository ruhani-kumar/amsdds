from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from backend.config import DecisionEngineConfig

# A Layer 2 callable takes the Layer 1 result dict and returns a prediction dict
# with keys: "class", "confidence", "entropy".
Layer2Callable = Callable[[Mapping], Mapping]


@dataclass(frozen=True)
class Prediction:
    """A single model's output, normalised."""

    predicted_class: str
    confidence: float
    entropy: float

    @classmethod
    def from_mapping(cls, data: Mapping) -> "Prediction":
        # Accept either "class" or "predicted_class" as the label key.
        label = data.get("class", data.get("predicted_class"))
        if label is None:
            raise ValueError("prediction is missing a class label")
        return cls(
            predicted_class=str(label),
            confidence=float(data["confidence"]),
            entropy=float(data["entropy"]),
        )

    def as_dict(self) -> dict:
        return {
            "class": self.predicted_class,
            "confidence": self.confidence,
            "entropy": self.entropy,
        }


@dataclass
class DecisionResult:
    route: str            # "accepted" | "escalated" | "escalated_failed"
    final_source: str     # "layer1" | "layer2"
    uncertain: bool
    predicted_class: str
    confidence: float
    entropy: float
    layer1: dict
    layer2: dict | None = None
    notes: list[str] = field(default_factory=list)


class DecisionEngine:
    def __init__(self, config: DecisionEngineConfig, layer2: Layer2Callable):
        self.config = config
        self._layer2 = layer2

    def _accepts(self, p: Prediction) -> bool:
        return (
            p.confidence >= self.config.conf_threshold
            and p.entropy <= self.config.entropy_threshold
        )

    def decide(self, layer1_prediction: Mapping) -> DecisionResult:
        l1 = Prediction.from_mapping(layer1_prediction)
        l1_dict = l1.as_dict()

        if self._accepts(l1):
            return DecisionResult(
                route="accepted",
                final_source="layer1",
                uncertain=False,
                predicted_class=l1.predicted_class,
                confidence=l1.confidence,
                entropy=l1.entropy,
                layer1=l1_dict,
                layer2=None,
                notes=["Layer 1 fast path: confidence and entropy within thresholds."],
            )

        # Escalate.
        try:
            l2 = Prediction.from_mapping(self._layer2(l1_dict))
        except Exception:  # noqa: BLE001 - Layer 2 must never break the request
            return DecisionResult(
                route="escalated_failed",
                final_source="layer1",
                uncertain=True,
                predicted_class=l1.predicted_class,
                confidence=l1.confidence,
                entropy=l1.entropy,
                layer1=l1_dict,
                layer2=None,
                notes=[
                    "Escalation was attempted but Layer 2 failed.",
                    "Falling back to the Layer 1 result; flagged uncertain.",
                ],
            )

        uncertain = l2.confidence < self.config.uncertain_floor
        return DecisionResult(
            route="escalated",
            final_source="layer2",
            uncertain=uncertain,
            predicted_class=l2.predicted_class,
            confidence=l2.confidence,
            entropy=l2.entropy,
            layer1=l1_dict,
            layer2=l2.as_dict(),
            notes=(
                ["Escalated to Layer 2; Layer 2 result is authoritative."]
                + (
                    ["Layer 2 confidence below the uncertain floor."]
                    if uncertain
                    else []
                )
            ),
        )
