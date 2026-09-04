from __future__ import annotations

import io

_SCENARIOS = {
    # name: (label, layer1_confidence, layer1_entropy, escalate)
    "A": ("nv", 0.92, 0.15, False),   # fast path — accepted at Layer 1
    "B": ("mel", 0.61, 0.44, True),   # high entropy — escalates to Layer 2
}


class MockEngine:
    is_mock = True

    def __init__(self, config):
        self._classes = list(config.classes)
        self._names = dict(config.class_names)
        self._malignant = list(config.malignant_classes)
        de = config.decision_engine
        self._conf_thr = de.conf_threshold
        self._ent_thr = de.entropy_threshold
        self._floor = de.uncertain_floor
        self._risk_thr = de.risk_threshold
        self._forced = config.mock_scenario if config.mock_scenario in _SCENARIOS else None

    # -- scenario selection -------------------------------------------------
    def _scenario_for(self, src) -> str:
        if self._forced:
            return self._forced
        try:
            from PIL import Image

            if isinstance(src, Image.Image):
                w, h = src.size
            elif isinstance(src, (bytes, bytearray)):
                w, h = Image.open(io.BytesIO(src)).size
            else:
                w, h = Image.open(src).size
            return "A" if (w * h) % 2 == 0 else "B"
        except Exception:  # noqa: BLE001
            return "A"

    # -- helpers (mirror Engine._summary / _entropy shape) ----------------
    def _probs(self, label: str, confidence: float) -> dict:
        n = len(self._classes)
        rest = (1.0 - confidence) / max(n - 1, 1)
        return {c: float(confidence if c == label else rest) for c in self._classes}

    def _summary(self, label: str, confidence: float, entropy: float) -> dict:
        return {
            "label": label,
            "label_name": self._names.get(label, label),
            "confidence": float(confidence),
            "probs": self._probs(label, confidence),
            "entropy": float(entropy),
        }

    # -- API --------------------------------------------------------------
    def predict(self, src, age=None, sex=None, localization=None, layer2_head=None):
        label, conf1, ent1, escalate = _SCENARIOS[self._scenario_for(src)]

        out = {
            "escalated": False,
            "unknown": False,
            "layer_used": "layer1",
            "metadata_used": any(v not in (None, "") for v in (age, sex, localization)),
        }
        out["layer1"] = self._summary(label, conf1, ent1)
        out["gate"] = {
            "conf": float(conf1),
            "entropy": float(ent1),
            "conf_threshold": self._conf_thr,
            "entropy_threshold": self._ent_thr,
            "reason": ("low confidence" if conf1 < self._conf_thr
                       else "high entropy" if escalate else "confident"),
        }

        final = out["layer1"]
        if escalate:
            head = layer2_head if layer2_head in ("ham", "hampad") else "hampad"
            l2 = self._summary(label, max(conf1, 0.66), ent1 + 0.02)
            l2["head"] = head
            out["layer2"] = l2
            out["escalated"] = True
            out["layer_used"] = f"layer2:{head}"
            final = l2

        risk = float(sum(final["probs"][c] for c in self._malignant))
        risk_flag = risk >= self._risk_thr
        uncertain = final["confidence"] < self._floor
        out.update(
            label=final["label"],
            label_name=final["label_name"],
            confidence=final["confidence"],
            probs=final["probs"],
            malignant_probability=risk,
            risk_flag=risk_flag,
            risk_level=("high" if risk_flag else "moderate" if risk >= 0.13 else "low"),
            uncertain=uncertain,
            advisory=("High risk — consult a dermatologist promptly." if risk_flag
                      else "Low-confidence result; consider clinical review." if uncertain
                      else "Low risk. Monitor for changes and re-check if it grows, bleeds or changes colour."),
            latency_ms={"layer1": 0.0, "layer2": 0.0, "total": 0.0},
        )
        return out

    def model_info(self):
        return {
            "mock": True,
            "classes": self._classes,
            "class_names": self._names,
            "malignant_classes": self._malignant,
            "decision_engine": {
                "conf_threshold": self._conf_thr,
                "entropy_threshold": self._ent_thr,
                "uncertain_floor": self._floor,
                "risk_threshold": self._risk_thr,
            },
            "runtime": {
                "device": "mock",
                "layer1_temperature": None,
                "layer2_heads": {"ham": None, "hampad": None},
                "layer2_default_head": "hampad",
                "ood_enabled": False,
                "ood_threshold": None,
            },
        }
