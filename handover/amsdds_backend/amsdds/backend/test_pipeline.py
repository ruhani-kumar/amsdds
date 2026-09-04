import json, sys, time
import numpy as np
from PIL import Image
from engine import Engine

e = Engine()

if "--parity" in sys.argv:
    import os, pandas as pd
    i = sys.argv.index("--parity")
    cache, routing = sys.argv[i + 1], sys.argv[i + 2]
    n = int(sys.argv[i + 3]) if len(sys.argv) > i + 3 else 200
    r = pd.read_csv(routing)
    r = r.sample(min(n, len(r)), random_state=0)
    e.pre.cc = False                       # cached images are already colour-constant
    ok_esc = ok_lbl = tot = 0
    for row in r.itertuples():
        p = os.path.join(cache, f"{row.image_id}.jpg")
        if not os.path.exists(p): continue
        o = e.predict(Image.open(p))
        tot += 1
        ok_esc += int(o["escalated"] == bool(row.escalated))
        exp = row.pred if isinstance(row.pred, str) else e.classes[int(row.pred)]
        ok_lbl += int(o["layer1"]["label"] == exp)
    print(f"{tot} images | routing agreement {ok_esc/tot:.1%} | L1 label agreement {ok_lbl/tot:.1%}")
    print("both should be ~100%; lower = preprocessing/temperature skew")
    sys.exit()

img = Image.open(sys.argv[1]) if len(sys.argv) > 1 else Image.new("RGB", (600, 450), (190, 130, 110))

print("=== no metadata (neutral vector) ===")
o = e.predict(img); print(json.dumps({k: o[k] for k in
    ("label", "confidence", "escalated", "unknown", "malignant_probability",
     "risk_flag", "risk_level", "layer_used", "metadata_used")}, indent=2))

print("\n=== with metadata ===")
o = e.predict(img, age=62, sex="male", localization="face")
print(json.dumps({k: o[k] for k in ("label", "confidence", "escalated",
    "malignant_probability", "risk_flag", "metadata_used")}, indent=2))

print("\n=== both Layer 2 heads (forced escalation path) ===")
for h in e.l2.heads:
    o = e.predict(img, layer2_head=h)
    tag = o.get("layer2", {}).get("head", "not escalated")
    print(f"  head={h:7s} -> used={tag}  label={o['label']}  risk={o['malignant_probability']:.3f}")

print("\n=== latency (10 runs) ===")
ts = []
for _ in range(10):
    t = time.perf_counter(); e.predict(img); ts.append((time.perf_counter() - t) * 1e3)
print(f"  median {np.median(ts):.1f} ms  (escalated={o['escalated']})")
