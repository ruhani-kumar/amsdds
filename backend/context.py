from __future__ import annotations

from flask import Flask

from backend.config import AppConfig
from backend.ml_core import build_engine

_CONFIG_KEY = "APP_CONFIG"


def get_config(app: Flask) -> AppConfig:
    return app.config[_CONFIG_KEY]


def set_config(app: Flask, config: AppConfig) -> None:
    app.config[_CONFIG_KEY] = config


def get_engine(app: Flask):
    """Lazily build and cache the ONE handover ML engine on the app.

    Mock mode -> MockEngine (no weights). Real mode -> the handover Engine,
    which loads the 5 weight files + the RT-DETR skeleton on first use.

    There is deliberately no second decision system: preprocessing, OOD
    (currently disabled), the confidence/entropy gate, Layer 2 head selection
    and malignant-risk all live inside this one engine.
    """
    store = app.extensions.setdefault("amsdds", {})
    if "engine" not in store:
        store["engine"] = build_engine(get_config(app))
    return store["engine"]
