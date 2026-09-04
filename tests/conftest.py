"""Shared test fixtures.

Application / API tests run the backend in DEVELOPMENT_MOCK_MODE, so they need
no weights, no torch and no transformers - the handover engine is swapped for
backend.ml_core.mock_engine.MockEngine. The legacy backend.decision_engine
unit tests use plain dicts + injected Layer 2 callables.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import create_app  # noqa: E402
from backend.config import (  # noqa: E402
    AppConfig,
    DecisionEngineConfig,
    load_thresholds_file,
    DEFAULT_THRESHOLDS_PATH,
    DEFAULT_WEIGHTS_DIR,
)


def _make_config(**overrides) -> AppConfig:
    raw = load_thresholds_file(DEFAULT_THRESHOLDS_PATH)
    de = raw.get("decision_engine", {})
    base = dict(
        checkpoint_path=DEFAULT_WEIGHTS_DIR / "does-not-exist.pt",
        thresholds_path=DEFAULT_THRESHOLDS_PATH,
        weights_dir=DEFAULT_WEIGHTS_DIR,
        device="cpu",
        mock_mode=True,
        mock_scenario=None,
        decision_engine=DecisionEngineConfig(
            conf_threshold=float(de.get("conf_threshold", 0.5)),
            entropy_threshold=float(de.get("entropy_threshold", 0.3)),
            uncertain_floor=float(de.get("uncertain_floor", 0.5)),
            risk_threshold=float(de.get("risk_threshold", 0.26)),
        ),
        classes=list(raw.get("classes", [])),
        malignant_classes=list(raw.get("malignant_classes", [])),
        class_names=dict(raw.get("class_names", {})),
        temperature=float(raw.get("calibration", {}).get("temperature", 0.959)),
    )
    base.update(overrides)
    return AppConfig(**base)


@pytest.fixture
def engine_config() -> DecisionEngineConfig:
    """For the legacy backend.decision_engine unit tests (retired module)."""
    return DecisionEngineConfig(
        conf_threshold=0.85, entropy_threshold=0.35, uncertain_floor=0.5
    )


@pytest.fixture
def app_config() -> AppConfig:
    return _make_config()


@pytest.fixture
def client(app_config):
    app = create_app(app_config)
    app.testing = True
    return app.test_client()


@pytest.fixture
def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (180, 120, 90)).save(buf, format="PNG")
    return buf.getvalue()
