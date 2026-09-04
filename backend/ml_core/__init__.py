from __future__ import annotations

import os

__all__ = ["build_engine"]


def build_engine(config):
    """Return the ONE inference engine for this configuration.

    Mock mode -> MockEngine (no weights, no torch / transformers imported).
    Real mode -> the handover Engine, with ``AMSDDS_WEIGHTS`` pinned to an
    absolute path so behaviour never depends on the process working directory.

    Imports are deferred so that importing ``backend.ml_core`` (which happens
    for every request, mock or real) does not pull in torch.
    """
    if config.mock_mode:
        from .mock_engine import MockEngine

        return MockEngine(config)

    os.environ.setdefault("AMSDDS_WEIGHTS", str(config.weights_dir))
    from .engine import Engine

    return Engine(str(config.thresholds_path))
