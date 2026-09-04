# AMSDDS — Adaptive Multi-Stage Skin Disease Detection

Two-layer skin-lesion triage. A lightweight multimodal CNN answers the easy
~72% of images; a confidence + entropy gate escalates the rest to a frozen
RT-DETR encoder with a GOA-tuned head. A feature-space check rejects non-skin
input outright, and a malignant-probability risk flag fires **independently of
the predicted label** so a mislabelled melanoma still routes to referral.

---

## 1. Quick start

```bash
git clone <repo> && cd amsdds
bash scripts/download_weights.sh        # then put the 5 files in weights/
cd backend && pip install -r requirements.txt
python test_pipeline.py                 # smoke test, no image needed
python app.py                           # serves on :8000
```

`AMSDDS_WEIGHTS=/abs/path/to/weights` overrides the weights directory.

## 2. Weights (NOT in git — 5 files, ~176 MB)

| file | role | key facts |
|---|---|---|
| `layer1_multimodal_mobilenetv3.pt` | Layer 1 | 299px, image + 19-dim metadata, T=0.959 |
| `layer2_rtdetr.pt` | Layer 2 encoder | 512px, frozen RT-DETR r50vd backbone + hybrid encoder |
| `layer2_rtdetr_goa_head.pt` | L2 head "ham" | dermoscopy-only, T=0.735 |
| `layer2_rtdetr_goa_head_hampad.pt` | L2 head "hampad" | phone-capable, T=0.803 (**default**) |
| `ood_scores_mobilenet.npz` | OOD | Mahalanobis mus/prec/unk_thr on L1 image features |

Temperatures live **inside** the checkpoints, never in the config — the loader
reads them, so a head can never be paired with the wrong T.

## 3. How a request flows

1. **Preprocess** — shades-of-gray colour constancy on the original image, then
   resize/crop per model (299 for L1, 512 for L2). Must match training exactly.
2. **Layer 1** — multimodal MobileNetV3. Metadata (age / sex / localization) is
   optional; missing fields become a neutral vector (mean age, "unknown").
   Returns logits **and** the 1280-d image feature.
3. **OOD check** — Mahalanobis distance of that feature to the nearest class
   mean. Above `unk_thr` → return `unknown`, no label, no gate. Runs *before*
   the gate because a garbage image can still produce a confident softmax.
4. **Gate** — escalate if `confidence < 0.5` or `normalised entropy > 0.3`.
5. **Layer 2** — frozen encoder → pooled 768-d → standardise → GOA head →
   `softmax(logits / T)`. Head selectable per request.
6. **Decision** — label = argmax; `malignant_probability` = P(mel)+P(bcc)+P(akiec);
   `risk_flag` = that sum ≥ **0.26**, independent of the label.

## 4. API

### `POST /predict` — multipart
| field | required | notes |
|---|---|---|
| `image` | yes | jpg/png bytes |
| `age` | no | integer years |
| `sex` | no | `male` / `female` (anything else → unknown) |
| `localization` | no | one of the 15 HAM sites (see `/model-info`) |
| `layer2_head` | no | `ham` or `hampad` (default `hampad`) |

Response fields the frontend uses:

| field | meaning |
|---|---|
| `unknown` | true → show "not a recognised skin lesion", suppress all other output |
| `escalated` | true → play the "running advanced analysis" state |
| `layer_used` | `layer1` / `layer2:hampad` / `layer2:ham` |
| `label`, `label_name`, `confidence`, `probs` | prediction |
| `malignant_probability`, `risk_flag`, `risk_level` | **risk_level ∈ low / moderate / high** |
| `advisory` | ready-to-display sentence; matches risk_level |
| `uncertain` | final confidence below 0.5 |
| `gate` | conf/entropy + which rule fired (debug panel) |
| `layer1`, `layer2` | per-layer sub-results (for a "second opinion" view) |
| `latency_ms` | `{layer1, layer2, total}` |
| `metadata_used` | whether any metadata was supplied |

**Frontend rule:** drive the UI from `risk_flag` and `risk_level`, not from
`label`. The label can be wrong; the risk flag is the safety mechanism.

### `GET /model-info`
Returns `thresholds.yaml` plus runtime values (device, loaded heads and their
temperatures, OOD threshold). Never hardcode a threshold in the frontend.

### `GET /health`

## 5. Measured results

| metric | value |
|---|---|
| Layer 1 test accuracy / macro-F1 | 0.845 / 0.708 |
| Escalation rate | 28.3% |
| Fast-path accuracy | 93.7% |
| Dangerous misses: total / caught by gate / leaked | 75 / 54 / 21 |
| Layer 2 (hampad) HAM test acc / F1 | 0.857 / 0.736 |
| Layer 2 (hampad) PAD (phone) test acc | 0.638 |
| **Risk flag on phone images: malignant sensitivity** | **0.966** |
| Risk flag on phone images: benign specificity | 0.484 |
| OOD far-AUROC (non-skin) | 0.907 |

Risk threshold tuned on PAD **val** (sens 0.964), reported once on test.

**Stated limitations.** Training data is dermoscopy (HAM10000) plus smartphone
clinical images (PAD-UFES-20); PAD contains no `df`/`vasc`. Phone-image
melanoma *labelling* is weak (the frozen encoder is dermoscopy-trained) — this
is why the risk flag, not the label, carries safety. 21 malignant lesions pass
the gate confidently mislabelled; feature-space OOD scores do not separate them
(AUROC ≈ 0.5), which is the ceiling of single-model gating. No skin-tone
evaluation has been done; HAM is predominantly light-skinned.

## 6. Changing weights or thresholds

Everything is in `backend/config/thresholds.yaml`. Swap the default Layer 2
head with `layer2.default_head: ham|hampad`. **Do not** change a gate or OOD
threshold without re-deriving it — they are tied to the specific Layer 1
checkpoint.

**Pending (see `measured.pending`):** if the HAM+PAD adapted Layer 1 lands, its
image features move, so in order: re-fit OOD → re-sweep the gate → regenerate
routing → re-score the GOA head. Then update `layer1.file` and the five
affected numbers. No code changes.

## 7. Verifying a change didn't break serving

```bash
python test_pipeline.py --parity /path/to/ham_cache /path/to/test_routing_detail.csv 300
```
Routing and Layer 1 label agreement must be ~100%. Lower means preprocessing or
temperature drift — the classic silent failure.

## 8. Team split

- **Backend/API** (`app.py`, Docker, deploy) — the contract in §4 is frozen; build against it.
- **Frontend** — build from §4 alone; `test_pipeline.py` prints real JSON to mock against.
- **Models/config** (`engine.py`, `models.py`, weights, `thresholds.yaml`) — owns §6.
