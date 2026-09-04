import math, time
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from preprocess import Preprocessor, load_image
from models import load_all


class Engine:
    def __init__(self, cfg_path="config/thresholds.yaml"):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.classes = list(self.cfg["classes"])
        self.mal_idx = [self.classes.index(c) for c in self.cfg["malignant_classes"]]
        self.pre = Preprocessor(self.cfg)
        self.l1, self.l2, self.ood, self.device = load_all(self.cfg)
        de = self.cfg["decision_engine"]
        self.conf_thr, self.ent_thr = float(de["conf_threshold"]), float(de["entropy_threshold"])
        self.floor, self.risk_thr = float(de["uncertain_floor"]), float(de["risk_threshold"])
        self.log_c = math.log(len(self.classes))
        self._warmup()

    def _warmup(self):
        from PIL import Image
        img = Image.new("RGB", (600, 450), (180, 120, 100))
        for _ in range(2): self.predict(img)

    def _entropy(self, p): return float(-(p * np.log(p + 1e-12)).sum() / self.log_c)

    def _summary(self, p):
        i = int(p.argmax())
        return {"label": self.classes[i], "label_name": self.cfg["class_names"][self.classes[i]],
                "confidence": float(p[i]),
                "probs": {c: float(p[k]) for k, c in enumerate(self.classes)}}

    def predict(self, src, age=None, sex=None, localization=None, layer2_head=None):
        t0 = time.perf_counter()
        img = self.pre.normalise_colour(load_image(src))
        out = {"escalated": False, "unknown": False, "layer_used": "layer1",
               "metadata_used": any(v not in (None, "") for v in (age, sex, localization))}

        # Layer 1 (image + metadata; neutral vector if none supplied)
        meta = self.l1.encode_meta(age, sex, localization)
        logits1, feat = self.l1.run(self.pre.for_layer1(img), meta)
        p1 = F.softmax(logits1 / self.l1.temperature, 1)[0].numpy()
        conf1, ent1 = float(p1.max()), self._entropy(p1)
        out["layer1"] = {**self._summary(p1), "entropy": ent1}
        t1 = time.perf_counter()

        # OOD reject (image features only) — before the gate
        if self.ood is not None:
            s = self.ood.score(feat)
            out["ood"] = {"score": s, "threshold": self.ood.threshold, "is_ood": s > self.ood.threshold}
            if s > self.ood.threshold:
                out.update(unknown=True, label=None,
                           label_name="Not a recognised skin lesion", confidence=None,
                           risk_level="unknown", risk_flag=False,
                           advisory="Image not recognised as a skin lesion. Retake with the lesion centred and well lit.",
                           latency_ms={"layer1": (t1 - t0) * 1e3, "total": (time.perf_counter() - t0) * 1e3})
                return out

        # Gate
        escalate = conf1 < self.conf_thr or ent1 > self.ent_thr
        out["gate"] = {"conf": conf1, "entropy": ent1,
                       "conf_threshold": self.conf_thr, "entropy_threshold": self.ent_thr,
                       "reason": ("low confidence" if conf1 < self.conf_thr else
                                  "high entropy" if escalate else "confident")}
        p_final = p1

        # Layer 2 (selectable GOA head)
        if escalate:
            out["escalated"] = True
            logits2, T2, used = self.l2.run(self.pre.for_layer2(img), layer2_head)
            p2 = F.softmax(logits2 / T2, 1)[0].numpy()
            out["layer2"] = {**self._summary(p2), "entropy": self._entropy(p2), "head": used}
            p_final = p2
            out["layer_used"] = f"layer2:{used}"
        t2 = time.perf_counter()

        # Decision: label from argmax, risk flag INDEPENDENT of the label
        final = self._summary(p_final)
        risk = float(p_final[self.mal_idx].sum())
        risk_flag = risk >= self.risk_thr
        uncertain = final["confidence"] < self.floor
        out.update(label=final["label"], label_name=final["label_name"],
                   confidence=final["confidence"], probs=final["probs"],
                   malignant_probability=risk, risk_flag=risk_flag,
                   risk_level=("high" if risk_flag else "moderate" if risk >= 0.13 else "low"),
                   uncertain=uncertain,
                   advisory=("High risk — consult a dermatologist promptly." if risk_flag else
                             "Low-confidence result; consider clinical review." if uncertain else
                             "Low risk. Monitor for changes and re-check if it grows, bleeds or changes colour."),
                   latency_ms={"layer1": (t1 - t0) * 1e3,
                               "layer2": (t2 - t1) * 1e3 if escalate else 0.0,
                               "total": (time.perf_counter() - t0) * 1e3})
        return out

    def model_info(self):
        info = dict(self.cfg)
        info["runtime"] = {"device": self.device, "layer1_temperature": self.l1.temperature,
                           "layer2_heads": {k: v["T"] for k, v in self.l2.heads.items()},
                           "layer2_default_head": self.l2.default,
                           "ood_enabled": self.ood is not None,
                           "ood_threshold": self.ood.threshold if self.ood else None}
        return info


if __name__ == "__main__":
    import json, sys
    e = Engine()
    print(json.dumps(e.predict(sys.argv[1]), indent=2))
