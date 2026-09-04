from __future__ import annotations

from typing import Protocol

from PIL import Image

from backend.config import AppConfig


class Layer1Unavailable(RuntimeError):
    """Raised when the real Layer 1 model cannot be loaded."""


CHECKPOINT_MISSING_MESSAGE = (
    "Layer 1 checkpoint not found at '{path}'.\n"
    "Set MODEL_CHECKPOINT_PATH to the location of layer1_mobilenetv3_best.pt, "
    "or enable DEVELOPMENT_MOCK_MODE=true to run the pipeline without weights."
)


class Layer1Interface(Protocol):
    is_mock: bool

    def predict_full(self, image: Image.Image): ...

class RealLayer1:
    is_mock = False

    def __init__(self, config: AppConfig):
        if not config.checkpoint_exists:
            raise Layer1Unavailable(
                CHECKPOINT_MISSING_MESSAGE.format(path=config.checkpoint_path)
            )
        try:
            # Lazy import: pulls in torch / torchvision only in real mode.
            from ml.inference import Layer1Model
        except Exception as exc:  # noqa: BLE001
            raise Layer1Unavailable(
                f"Could not import the Layer 1 wrapper (is torch installed?): {exc}"
            ) from exc

        try:
            self._model = Layer1Model(
                str(config.checkpoint_path), device=config.device
            )
        except Exception as exc:  # noqa: BLE001
            raise Layer1Unavailable(
                f"Failed to load Layer 1 checkpoint: {exc}"
            ) from exc

    def predict_full(self, image: Image.Image):
        return self._model.predict_full(image)

class MockLayer1:
    """Deterministic stand-in for Layer 1.

    Scenario A -> accepted fast path.   (nv, conf 0.92, entropy 0.20)
    Scenario B -> escalated to Layer 2. (mel, conf 0.65, entropy 0.45)

    If AppConfig.mock_scenario is set ("A"/"B") every image uses that scenario.
    Otherwise the scenario is chosen deterministically per image so a demo can
    show both UI paths by uploading two different images.
    """

    is_mock = True

    SCENARIOS = {
        "A": ("nv", 0.92, 0.20),
        "B": ("mel", 0.65, 0.45),
    }

    def __init__(self, config: AppConfig):
        self._classes = list(config.classes)
        self._forced = config.mock_scenario if config.mock_scenario in self.SCENARIOS else None

    def _scenario_for(self, image: Image.Image) -> str:
        if self._forced:
            return self._forced
        w, h = image.size
        return "A" if (w * h) % 2 == 0 else "B"

    def predict_full(self, image: Image.Image):
        label, confidence, entropy = self.SCENARIOS[self._scenario_for(image)]
        # Build a plausible probability vector: `confidence` on the predicted
        # class, remainder spread evenly. Only used for display / completeness.
        n = len(self._classes)
        rest = (1.0 - confidence) / max(n - 1, 1)
        probs = [confidence if c == label else rest for c in self._classes]
        return label, confidence, entropy, probs

def build_layer1(config: AppConfig) -> Layer1Interface:
    """Return the Layer 1 implementation for the current configuration.

    Never silently substitutes: mock mode is explicit config, and a missing
    checkpoint in real mode raises Layer1Unavailable with an actionable message.
    """
    if config.mock_mode:
        return MockLayer1(config)
    return RealLayer1(config)
