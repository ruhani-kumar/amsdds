# AMSDDS — Layer 1 + Decision Engine thresholds, handoff

**Team Ignite (SHIH26-TID-139) · Smart Horizon 2026**
Owner of this work: Joan Sara Joe (Person E)
Covers blueprint Sections 6 (data), 7.1 (Layer 1), 8 (confidence/entropy/thresholds)

Every number below is measured on our own lesion-grouped splits. Nothing here
is copied from a paper or asserted from the proposal.

---

## 1. Headline results

**Layer 1 is MobileNetV3-Large, fine-tuned**, chosen over a DINOv2 ViT-S
alternative on measured test performance.

| | MobileNetV3 (fine-tuned) | DINOv2 ViT-S (frozen + MLP) |
|---|---|---|
| test accuracy | **0.8446** | 0.8119 |
| test macro-F1 | **0.7059** | 0.6387 |
| ECE raw | 0.0289 | 0.0251 |
| ECE calibrated | 0.0249 | 0.0183 |
| val macro-F1 | 0.7152 | 0.7174 |

**Adaptive routing, measured on the held-out test split:**

| metric | value |
|---|---|
| escalation rate | **29.6%** |
| compute saved vs always-on transformer | **70.4%** |
| fast-path accuracy (the 70% shipped without a second opinion) | **93.5%** |
| accuracy on the escalated pile | 62.9% |
| overall accuracy | 84.5% |
| malignant-called-benign cases caught by the gate | **52 of 82 (63%)** |
| malignant-called-benign cases leaked | 30 |

Blueprint Section 7.1 target was >85% top-1 validation accuracy. We report
**84.5% test accuracy** on a lesion-grouped held-out split — the harder and
more honest number — with **93.5% on the fast path**.

### Two findings worth putting in the report

**1. MobileNetV3 was never the problem — the recipe was.** Our first attempt
performed badly enough that we nearly abandoned the architecture. Three bugs:

- splits were not grouped by `lesion_id` (HAM10000 has multiple photos per
  lesion; random image splits leak near-duplicates into validation)
- class weights used raw inverse frequency, ~58:1 (`nv` 6705 vs `df` 115),
  which destabilised training
- only the last 2 backbone blocks were unfrozen, and the classifier's
  1280-unit hidden layer was discarded

After fixing all three, MobileNetV3 matched a modern self-supervised ViT on
validation and beat it on test.

**2. The proposal's 0.85 confidence threshold turned out to be correct — but
we derived it rather than assuming it.** The sweep independently selected
`conf=0.85, entropy=0.35`. We can show the search that produced it, which is
a materially stronger claim than asserting the number.

---

## 2. Data pipeline (blueprint Section 6)

**Source:** HAM10000 via `kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")`
**Classes (7):** `akiec, bcc, bkl, df, mel, nv, vasc`

### Split — grouped, stratified, seed 42

`StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)` grouped on
`lesion_id`. Fold 0 = test, fold 1 = val, remainder = train.

| split | n |
|---|---|
| train | 7116 |
| val | 1458 |
| test | 1441 |

| dx | train | val | test |
|---|---|---|---|
| akiec | 230 | 63 | 34 |
| bcc | 358 | 67 | 89 |
| bkl | 772 | 159 | 168 |
| df | 89 | 17 | 9 |
| mel | 782 | 185 | 146 |
| nv | 4796 | 935 | 974 |
| vasc | 89 | 32 | 21 |

Asserted in code: **no `lesion_id` appears in more than one split.**

### Preprocessing

- **Shades-of-Gray colour constancy** (power 6, Barata et al.) — illumination
  normalisation applied before everything else. This is our concrete answer to
  the "variations in lighting and image quality" challenge on slide 3. Applied
  at both train and inference; skipping it at inference is a silent
  train/serve skew bug.
- Resize shorter side to 256, cached to local disk as JPEG q95 (one-time,
  ~4 min). Without the cache the colour-constancy maths reruns every epoch and
  the CPU starves the GPU.
- Eval: `Resize(256)` → `CenterCrop(224)` → ImageNet mean/std
- Train augmentation (targets the lighting/quality failure modes specifically):
  `RandomResizedCrop(224, scale=0.65-1.0)`, H+V flip, `RandomRotation(30)`
  p=0.5, `ColorJitter(brightness .35, contrast .35, saturation .25, hue .03)`,
  `GaussianBlur` p=0.25

---

## 3. Layer 1 training recipe (blueprint Section 7.1)

- **Model:** `torchvision.mobilenet_v3_large`, `IMAGENET1K_V2` weights
- **Head:** `classifier[3]` → `Linear(1280, 7)`; the 1280 hidden layer is
  **kept** (the blueprint's `Linear(960, 7)` discards it)
- **Stage 1:** backbone frozen, classifier only, 5 epochs, AdamW lr 1e-3, cosine
- **Stage 2:** **entire backbone unfrozen**, 25 epochs, AdamW param groups
  (backbone 1e-4, classifier 5e-4), cosine, weight decay 1e-4
- **Loss:** class-weighted cross-entropy, **label smoothing 0.05**
- **Class weights:** sqrt-inverse frequency, normalised, **capped at 2.0**
- **Batch:** 32, AMP fp16, seed 42
- **Selection:** best val macro-F1 (reached S2 epoch 21, 0.7152)
- **Runtime:** ~72s/epoch, ~36 min total

### How the weight cap was chosen

Swept on the DINOv2 head first (cheap — cached features, ~45s per value),
then carried over:

| cap | val macro-F1 |
|---|---|
| 1.0 | 0.6908 |
| **2.0** | **0.7174** |
| 3.0 | 0.7036 |
| 5.0 | 0.7057 |
| 10.0 | 0.7015 |

Even cap 1.0 (no weighting at all) beat the original 10.0. The class weights
were doing net harm.

---

## 4. Calibration (blueprint Section 8)

Temperature scaling fitted by LBFGS on validation logits, applied at inference
as `softmax(logits / T)`.

| model | ECE raw | ECE calibrated | fitted T |
|---|---|---|---|
| **MobileNetV3 (production)** | 0.0289 | 0.0249 | **0.9718** |
| DINOv2 MLP head | 0.0251 | 0.0183 | 0.938 |
| DINOv2 *linear* head | 0.1317 | 0.0204 | >1 |

**Point worth making on slide 3.** The linear head was badly overconfident
(ECE 0.132) and temperature scaling cut that 6×. But both heads trained with
label smoothing and dropout started near 0.025 and fitted **T < 1** — i.e.
slightly *under*confident. Overconfidence is better fixed in the training
recipe than bolted on afterward. Report both numbers; the contrast is the
interesting part.

### Entropy

Shannon entropy over the calibrated softmax, normalised by `log(7)` so the
threshold is class-count-independent and lands in [0, 1].

---

## 5. Per-class test results — MobileNetV3

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| akiec | 0.533 | 0.471 | 0.500 | 34 |
| bcc | 0.819 | 0.764 | 0.791 | 89 |
| bkl | 0.698 | 0.714 | 0.706 | 168 |
| df | 0.800 | 0.444 | 0.571 | 9 |
| mel | 0.569 | 0.534 | 0.551 | 146 |
| nv | 0.917 | 0.938 | 0.927 | 974 |
| vasc | 1.000 | 0.810 | 0.895 | 21 |
| **macro avg** | 0.762 | 0.668 | 0.706 | 1441 |
| accuracy | | | **0.845** | 1441 |

Confusion matrix (rows = true, columns = predicted):

```
        akiec  bcc  bkl  df  mel   nv  vasc
akiec      16    5    5   0    6    2     0
bcc         3   68    5   0    6    7     0
bkl         8    2  120   0   12   26     0
df          1    2    0   4    0    2     0
mel         0    5   17   1   78   45     0
nv          2    1   25   0   32  914     0
vasc        0    0    0   0    3    1    17
```

### The number that justifies the architecture

Treating `mel`, `bcc`, `akiec` as malignant/concerning:

**82 of 269 malignant lesions (30%) are classified as benign by Layer 1.**
Of these, 63 are melanoma — 45 called `nv`, 17 called `bkl`, 1 called `df`.

This is roughly in line with published HAM10000 results, not a defect of our
training. It is the measured case for confidence-gated escalation: these are
the failures Layer 2 exists to catch.

`df` (9 test images) and `vasc` (21) are small enough that per-class metrics
are noise-dominated — read macro-F1 for those two.

### Caveat on the DINOv2 comparison

Both tied on val (~0.715) but DINOv2 dropped 0.079 on test while MobileNet
dropped 0.009. Part of that is selection pressure — the DINOv2 head was chosen
best-of-5 weight caps × 60 epochs on val, so its val score is partly fitted to
val. MobileNet held up better on a genuinely held-out split. Worth stating if
a judge asks why we trust the test gap.

---

## 6. Threshold sweep (blueprint Section 8.3)

110-point grid: confidence 0.50→0.95 and normalised entropy 0.10→0.60, both in
0.05 steps, **swept on validation only**. Test was touched once, afterward.

**Selection rule.** The blueprint's two conditions — fast-path accuracy ≥ 90%
and escalation rate in 20–35% — are treated as **constraints**, not objectives.
Within the feasible region we maximise the fraction of malignant-called-benign
cases the gate catches, tie-breaking toward less escalation. Escalation rate is
a compute budget; optimising it directly produces a gate that escalates cheap
easy cases while confidently shipping melanoma misses.

### Chosen thresholds

```yaml
conf_threshold: 0.85
entropy_threshold: 0.35
uncertain_floor: 0.50
temperature: 0.9718
```

### Reading the heatmap (`threshold_sweep.png`)

Four panels, chosen point starred. What the surface shows:

- **Entropy dominates below ~0.25.** At `entropy_thr = 0.10` the escalation
  rate is ~90% regardless of confidence — almost no prediction has normalised
  entropy that low. Those bottom rows catch ~90% of dangerous misses but
  escalate nearly everything, buying safety by abandoning the compute saving
  entirely. This is why entropy needed sweeping rather than guessing.
- **Confidence dominates above ~0.90.** The far-right column pushes fast-path
  accuracy to ~0.945 but escalation past 75%.
- **The feasible region is a narrow band**, and `0.85 / 0.35` sits at its
  corner — the most dangerous-miss capture available without breaking either
  constraint. Moving either threshold up buys a few more caught misses at a
  steep escalation cost.

### Results at the chosen thresholds (test set)

| metric | value |
|---|---|
| escalation rate | 0.2956 |
| compute saved vs always-on | 0.7044 |
| fast-path accuracy | 0.9350 |
| escalated-subset accuracy | 0.6291 |
| overall accuracy | 0.8446 |
| dangerous misses total | 82 |
| dangerous misses caught | 52 (63%) |
| dangerous misses leaked | 30 |

**62.9% is RT-DETR's bar.** If Layer 2 doesn't beat Layer 1's accuracy on the
escalated pile, escalation costs latency for nothing.

**Replace the proposal's literature-quoted 60–70% compute saving with our
measured 70.4%**, and note the consistency rather than citing theirs.

---

## 7. Saved artifacts

All in Google Drive: **`/content/drive/MyDrive/amsdds/`**

| file | what it is | who needs it |
|---|---|---|
| `splits.csv` | the canonical split — `image_id, lesion_id, dx, y, split` | **everyone**; load this, never re-split |
| `layer1_mobilenetv3_best.pt` | **production Layer 1** | Ruhani (A), Lakshmishree (C) |
| `metrics_mobilenetv3.json` | acc / macro-F1 / ECE raw+cal / val F1 / T | report, demo slide |
| `layer1_comparison.csv` | MobileNet vs DINOv2 side-by-side | report |
| `configs/thresholds.yaml` | CONF / ENTROPY / UNCERTAIN_FLOOR + measured routing numbers | **Lakshmishree (C)** — `/model-info` |
| `threshold_sweep.png` | 4-panel heatmap, chosen point starred | **demo + report** |
| `threshold_sweep.csv` | full 110-point grid | report appendix |
| `routing_results.json` | test-set routing metrics | demo |
| `test_routing_detail.csv` | per-image confidence, entropy, escalated, dangerous_miss | **demo image curation** |
| `layer2_train_subset.csv` | train images below the gate — **see the warning in §8** | Kashish (B) |
| `dinov2_feats.npz` | cached DINOv2 features (21348/1458/1441 × 768) | keep — needed for OOD work |
| `layer1_dinov2_head.pt` | DINOv2 head (baseline, not production) | archive |
| `ood_scores.npz` | energy + Mahalanobis scores, class means, precision matrix | OOD work |

**Not in Drive, rebuilds in ~4 min:** `/content/ham_cache/` — preprocessed
256px colour-constant JPEGs. Local disk deliberately; Drive is far too slow
for thousands of small reads.

### Checkpoint contents

```python
{
  "state": state_dict,       # val-best weights, not last epoch
  "classes": ["akiec","bcc","bkl","df","mel","nv","vasc"],
  "temperature": 0.9718,     # MUST be applied at inference
  "img_size": 224,
  "color_constancy": True,   # MUST be applied at inference
  "weight_cap": 2.0,
  "metrics": {...},
}
```

**Two silent-failure risks.** Colour constancy and temperature scaling both ran
during training. If either is skipped at inference you get plausible-looking
wrong answers, not a crash.

---

## 8. What goes to whom

**Ruhani (A) — ML core**
`layer1_mobilenetv3_best.pt`, `configs/thresholds.yaml`, `threshold_sweep.csv/png`.
Layer 1 and the Section 8.3 sweep are done, ahead of the Day 7 milestone. The
Decision Engine branches (Section 9) still need implementing against these
thresholds. **Ownership note:** Section 15 assigns Layer 1 and the sweep to
Person A; both were delivered by Person E. Worth agreeing explicitly who picks
up the Decision Engine rather than letting it drift.

**Kashish (B) — Layer 2**

⚠️ **`layer2_train_subset.csv` contains only 539 images (7.6% of train), but
the test escalation rate is 29.6%. Do not train on it as-is.** Those train
images were seen 30 times during fine-tuning, so Layer 1 is overconfident on
them and the gate barely fires. RT-DETR trained on 539 unrepresentative images
would not transfer to the ~430 test images it actually receives.

Two fixes:

- **Option A** — mine out-of-fold: 5-fold CV on train, score each fold with a
  model that never saw it. Correct, yields a realistic ~30% (~2100 images),
  costs ~3 hours of training.
- **Option B (recommended)** — pull **ISIC 2019** in, score with the current
  Layer 1, keep the bottom ~30% by confidence. This is what blueprint Section
  7.2 actually specifies, it is unseen data so confidence is honest, it gives
  more images, and it needs no retrain. Person E owns the ISIC pipeline.

Bar to beat either way: **62.9%**, Layer 1's accuracy on the escalated pile.

**Lakshmishree (C) — backend**
`configs/thresholds.yaml` for `/model-info`. `/predict` order matters:
colour constancy → resize/crop → normalise → model → `softmax(logits / T)` →
confidence + normalised entropy → decision engine.

```python
import torch
from torchvision.models import mobilenet_v3_large
c = torch.load("layer1_mobilenetv3_best.pt", map_location=dev, weights_only=False)
m = mobilenet_v3_large()
m.classifier[3] = torch.nn.Linear(1280, len(c["classes"]))
m.load_state_dict(c["state"]); m.eval().to(dev)
TEMP = c["temperature"]   # 0.9718
```

**Shreya (D) — frontend**
`test_routing_detail.csv`. For Scenario A filter `escalated == False` with high
confidence. For Scenario B filter `escalated == True & dangerous_miss == True` —
those are cases where the gate provably catches a malignant lesion Layer 1 got
wrong. Far stronger than a generic ambiguous image, and there are 52 to
choose from.

---

## 9. Status

**Done**
- [x] Dataset acquired, cleaned, lesion-grouped split, leakage-asserted
- [x] Preprocessing with colour constancy, cached
- [x] Layer 1 fine-tuned, checkpointed, evaluated
- [x] Baseline comparison against DINOv2 ViT-S
- [x] Temperature scaling + ECE before/after
- [x] Confidence + normalised entropy
- [x] Threshold sweep, chosen thresholds, heatmaps, `thresholds.yaml`
- [x] Routing metrics measured on test

**Next**
- [ ] ISIC 2019 acquisition + remap → honest Layer 2 training subset (Person E)
- [ ] Decision Engine branches, Section 9 (owner TBD)
- [ ] RT-DETR Layer 2 (Kashish)
- [ ] OOD detection — blueprint Section 9 defers it, slide 3 claims it. Cheap
      version: hold one class out of training entirely, then measure whether
      energy or Mahalanobis separates it from the six seen classes. Gives a
      real AUROC for "unknown disease detection". DINOv2 features already cached.

**Deferred per Section 20:** Supabase logging, Grad-CAM, OpenRouter explanations.

---

## 10. Honest limitations — state these, don't hide them

1. **30 dangerous misses leak through the gate.** Layer 1 is *confidently*
   wrong on 30 malignant lesions, so escalation never fires. This is the
   ceiling of softmax-based gating and the strongest argument for adding an
   OOD score (energy or Mahalanobis) as a second gate signal. Testable now
   with the cached features: do those 30 have anomalous feature-space
   distances even though their confidence is high?
2. **Skin tone.** Slide 10 claims "Skin Tone Inclusive Training." HAM10000 is
   overwhelmingly light-skinned and we cannot support that claim with it.
   Either bring in Fitzpatrick17k or DDI and report per-tone metrics, or soften
   the claim to a stated limitation with ITA-estimated tone bins.
3. **Melanoma recall is 0.534.** Nearly half of melanomas are missed by Layer 1.
   Frame as the motivation for escalation, with the measured 63% catch rate —
   not as a solved problem.
4. **`df` and `vasc` have 9 and 21 test images.** Noise-dominated.
5. **No bounding-box labels exist in HAM10000.** SSDLite is pretrained COCO,
   display-only, contributing nothing to classification. Say this out loud in
   the demo, not just in the blueprint assumptions.
6. **The escalated-pile accuracy of 62.9% is a bar, not a result.** If RT-DETR
   lands below it, the honest conclusion is that escalation didn't pay for
   itself — and we should report that rather than bury it.
