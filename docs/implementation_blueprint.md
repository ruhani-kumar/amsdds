# Adaptive Multi-Layer Skin Disease Detection System (AMSDDS)
## System Implementation Blueprint — Smart Horizon 2026 (Team ID SHIH26-TID-139, Team Ignite)

**Team:** Ruhani Kumar (Lead), Joan Sara Joe, Kashish Mittal, Lakshmishree K P, Shreya Jaganatha Gowda
**Window:** 16 days → 48-hour build/demo window at New Horizon College of Engineering (Sept 3–5, 2026)
**Core innovation to preserve:** Confidence-aware adaptive routing between MobileNetV3 (fast path) and RT-DETR (deep path), gated by softmax confidence + entropy.

**Stated assumptions (since the proposal underspecifies these, and the brief instructs us to decide rather than ask):**
- HAM10000/ISIC are classification-labeled, not bounding-box labeled. SSDLite is therefore used as a **pretrained, non-fine-tuned localizer** to draw a lesion bounding box for UI purposes only — it does not participate in the accept/escalate decision. All classification decisions come from the MobileNetV3 classification head (Layer 1) and the RT-DETR classification head (Layer 2).
- RT-DETR is used as a **classification backbone**, not a full object-detection re-train — its encoder-decoder is fine-tuned with a classification head on the same disease classes, because training a full RT-DETR detector from scratch is infeasible in 16 days.
- Team has 5 members. The brief's "four workstreams" instruction is reconciled by pairing the two ML sub-tracks (Layer 1 and Layer 2) as a single ML pod of 2 people running in parallel, giving 4 parallel workstreams total, mapped to 5 people (see Section 15).
- Demo environment is a single laptop/cloud VM, not a scaled production deployment. Docker/cloud auto-scale is documented but treated as post-MVP.

---

## 1. SYSTEM REQUIREMENTS

### 1.1 Functional Requirements

**Input & Preprocessing**
- FR-01: User can upload a skin lesion image (JPEG/PNG/TIFF) via web UI.
- FR-02: User can optionally activate a live camera feed and capture a frame.
- FR-03: System validates file type, size, and minimum resolution (224×224).
- FR-04: System rejects invalid/corrupt/non-image files with a clear error.
- FR-05: System runs an input sanity check (reject clearly non-skin images using a lightweight heuristic/classifier, see NFR-Error Handling).
- FR-06: System preprocesses image: resize (224×224 for MobileNetV3, 640×640 for RT-DETR), normalize (ImageNet mean/std), CLAHE contrast enhancement.

**Layer 1 — Fast Path**
- FR-07: System runs MobileNetV3-Large classification head on the preprocessed image.
- FR-08: System runs pretrained SSDLite to produce a lesion bounding box overlay (display-only).
- FR-09: System computes softmax class-probability distribution.
- FR-10: System computes top-1 confidence score.
- FR-11: System computes prediction entropy over the softmax distribution.

**Adaptive Decision Engine**
- FR-12: System applies the routing rule (confidence ≥ threshold AND entropy ≤ threshold → accept; else → escalate).
- FR-13: System logs the routing decision (accepted / escalated) and the metrics that produced it.
- FR-14: System escalates to RT-DETR when routing rule fails.

**Layer 2 — Escalation Path**
- FR-15: System runs RT-DETR classification head on the 640×640 preprocessed image when escalated.
- FR-16: System computes RT-DETR's confidence and entropy in the same format as Layer 1.
- FR-17: System merges Layer 2 output into the final result (Layer 2 result always wins over Layer 1 once escalation occurs).

**Output**
- FR-18: System maps the predicted class to a risk level (e.g., benign / monitor / high-risk) via a static lookup table.
- FR-19: System flags results as "uncertain" when even RT-DETR's confidence stays below a secondary floor threshold.
- FR-20: System returns a structured JSON result (see Section 10.4).
- FR-21: System displays disease label, confidence %, risk level, and which model produced the result (MobileNetV3 vs RT-DETR).
- FR-22: System visually animates the routing decision (fast analysis → confidence shown → escalating → advanced analysis → result) per Section 11.
- FR-23: System reports end-to-end processing latency in the UI.
- FR-24: System displays a non-diagnostic disclaimer ("not a substitute for professional medical advice") on every result screen.

**Operational**
- FR-25: System exposes a health-check endpoint for demo reliability checks.
- FR-26: System exposes a model-info endpoint (versions, thresholds, class list) for debugging during demo.

### 1.2 Non-Functional Requirements

| Category | Requirement |
|---|---|
| Latency | Fast path (Layer 1 only) ≤ 1.5s on CPU/laptop GPU; escalated path (Layer 1 + Layer 2) ≤ 4s. Both measured end-to-end from upload to rendered result. |
| Reliability | If RT-DETR fails to load or errors at inference time, system falls back to the Layer-1 result with an "uncertain, deep analysis unavailable" flag rather than crashing. |
| Usability | Zero-training UI: a first-time user must be able to get a result within 3 clicks (upload → analyze → result). |
| Scalability | Not a hard requirement for the hackathon; document the path (containerization, model serving) but do not implement autoscaling. |
| Security | No image is persisted to disk or database beyond the inference request lifecycle. All communication over HTTPS in any hosted demo. Basic input validation to reject obviously malicious/non-image payloads. |
| Privacy | No PII collected. No patient images stored. If Supabase logging is used (optional), only aggregate, non-identifiable metrics are stored — never the image or a derivative that could reconstruct it. |
| Modularity | Layer 1, Layer 2, Decision Engine, and API are separate modules with defined interfaces so any one can be swapped/retrained independently. |
| Maintainability | Config-driven thresholds (no hardcoded magic numbers in code — single `config.yaml`). Consistent naming for FR/module mapping so tasks map 1:1 to this blueprint. |
| Deployment | Runs locally via `docker-compose up` (backend + frontend) and/or a single cloud VM for the live demo. No Kubernetes, no multi-region requirement. |
| Error Handling | Every API failure mode returns a structured error object with an HTTP status code and a human-readable message (Section 10). |

---

## 2. FINAL SYSTEM COMPONENTS

**1. Frontend (Streamlit)**
- Purpose: Upload/capture image, trigger analysis, render adaptive-routing animation and final result.
- Inputs: User image (file or camera frame), API responses (JSON).
- Outputs: Rendered UI screens (Section 11).
- Technology: Streamlit (Python), calling the Flask API over HTTP.
- Dependencies: Backend API must be running and reachable.
- Team implements: All 6 screens, API client, loading/error states, routing-visualization component.
- Done when: A user can complete the full flow (upload → fast result OR escalation → final result) with no manual steps.

**2. API / Backend (Flask)**
- Purpose: Single entry point that orchestrates preprocessing → Layer 1 → Decision Engine → (optional) Layer 2 → response assembly.
- Inputs: HTTP POST with image bytes.
- Outputs: JSON prediction response (Section 10.4).
- Technology: Flask + Flask-RESTful conventions, Gunicorn for demo hosting.
- Dependencies: Preprocessing module, both model modules, Decision Engine, Result Generator.
- Team implements: `/predict`, `/health`, `/model-info` endpoints; error handling middleware; request validation.
- Done when: All 3 endpoints pass the API test suite (Section 18) including malformed-input cases.

**3. Image Preprocessing**
- Purpose: Convert raw upload into model-ready tensors for both layers.
- Inputs: Raw image bytes.
- Outputs: Two tensors — 224×224 normalized (Layer 1) and 640×640 normalized (Layer 2), plus a CLAHE-enhanced RGB array for the bounding-box overlay.
- Technology: OpenCV + TorchVision transforms.
- Dependencies: None (first stage).
- Team implements: `preprocess.py` with unit-testable pure functions (validate → resize → CLAHE → normalize).
- Done when: Function outputs match expected tensor shape/dtype for a fixed set of test images, deterministically.

**4. MobileNetV3 + SSDLite (Layer 1)**
- Purpose: Fast classification (primary) + display-only bounding box (secondary).
- Inputs: 224×224 normalized tensor (classifier), CLAHE RGB array (SSDLite).
- Outputs: Class-probability vector; bounding box coordinates + label (display only).
- Technology: TorchVision `mobilenet_v3_large` (pretrained, fine-tuned classifier head) + TorchVision pretrained `ssdlite320_mobilenet_v3_large` (COCO weights, unmodified).
- Dependencies: Preprocessing.
- Team implements: Fine-tuning script (offline, in Colab/Kaggle, not part of the live API), inference wrapper class `Layer1Model.predict(tensor) -> probs`.
- Done when: Fine-tuned checkpoint hits the accuracy target in Section 7 and the inference wrapper returns probabilities in < 300ms on CPU for a single image.

**5. Confidence Engine**
- Purpose: Turn a probability vector into a single top-1 confidence score.
- Inputs: Softmax probability vector.
- Outputs: `confidence: float`, `predicted_class: str`.
- Technology: Pure Python/NumPy, no external dependency.
- Dependencies: Layer 1 or Layer 2 output.
- Team implements: `confidence(probs) -> (label, score)`.
- Done when: Unit tests confirm correct top-1 extraction against hand-computed cases.

**6. Entropy / Uncertainty Engine**
- Purpose: Quantify how "spread out" the probability distribution is, as a second uncertainty signal independent of top-1 confidence.
- Inputs: Softmax probability vector.
- Outputs: `entropy: float` (normalized to [0,1]).
- Technology: Pure Python/NumPy (Section 8 formula).
- Dependencies: Layer 1 or Layer 2 output.
- Team implements: `entropy(probs) -> float`.
- Done when: Unit tests confirm entropy = 0 for a one-hot distribution and entropy = 1 for a uniform distribution over N classes.

**7. Adaptive Decision Engine**
- Purpose: The core innovation — decide accept vs. escalate, and assemble which layer's output is authoritative.
- Inputs: Layer 1 confidence, Layer 1 entropy, (conditionally) Layer 2 confidence/entropy.
- Outputs: `route: "accepted" | "escalated"`, `final_source: "layer1" | "layer2"`, `uncertain: bool`.
- Technology: Pure Python, config-driven thresholds.
- Dependencies: Confidence Engine, Entropy Engine, Layer 2 (conditionally invoked).
- Team implements: `decide(conf, entropy, layer2_fn) -> DecisionResult` (Section 9).
- Done when: All routing branches in Section 9 are covered by tests, including the fallback/failure branches.

**8. RT-DETR (Layer 2)**
- Purpose: Deep contextual re-classification for ambiguous/low-confidence cases.
- Inputs: 640×640 normalized tensor.
- Outputs: Class-probability vector (same schema as Layer 1, for a shared Confidence/Entropy Engine).
- Technology: Ultralytics RT-DETR-L (pretrained), fine-tuned classification head on ISIC subset.
- Dependencies: Preprocessing, invoked only by the Decision Engine.
- Team implements: Fine-tuning script (offline), inference wrapper `Layer2Model.predict(tensor) -> probs`.
- Done when: Fine-tuned checkpoint improves accuracy on the low-confidence validation subset vs. Layer 1 alone, and inference completes in < 3s on available GPU/CPU.

**9. Result Generator**
- Purpose: Assemble the final structured API response from the Decision Engine's output.
- Inputs: DecisionResult, class label, risk-level lookup table, timing data.
- Outputs: Final JSON response (Section 10.4).
- Technology: Pure Python.
- Dependencies: Decision Engine, static risk-level config.
- Team implements: `build_response(decision_result, timing) -> dict`.
- Done when: Output matches the schema exactly for both accepted and escalated cases.

**10. Explainability (Post-MVP, optional)**
- Purpose: Grad-CAM heatmap overlay on the lesion for interpretability.
- Inputs: Layer 1 or Layer 2 model + input tensor.
- Outputs: Heatmap image.
- Technology: `pytorch-grad-cam`.
- Dependencies: Trained models.
- Team implements: Only if time remains after Milestone 6 — marked OPTIONAL, not on the critical path.
- Done when: N/A unless attempted.

**11. Evaluation**
- Purpose: Produce the accuracy/latency/calibration numbers needed for the demo and report.
- Inputs: Held-out test split.
- Outputs: Metrics report (accuracy, macro-F1, AUC-ROC, ECE, latency, routing rate).
- Technology: scikit-learn, Matplotlib/Seaborn.
- Dependencies: Trained Layer 1 and Layer 2 checkpoints.
- Team implements: `evaluate.py` notebook/script producing the numbers used in Sections 18–19.
- Done when: A single script reproduces all reported numbers from a checkpoint + test split.

**12. Optional Database (Supabase)**
- Purpose: Store aggregate, non-identifiable inference logs for a "session stats" demo panel, if time permits.
- Inputs: Routing decisions, confidence/entropy values, timestamps (no images).
- Outputs: Row inserts, read-only dashboard query.
- Technology: Supabase (Postgres).
- Dependencies: Backend API (optional integration point).
- Team implements: Only if core pipeline (Sections 3–9, 2 above) is complete by Day 12 — explicitly marked OPTIONAL in Section 12/20.
- Done when: N/A unless attempted.

---

## 3. FINAL ARCHITECTURE

```
                    ┌───────────────────────┐
                    │   Streamlit Frontend   │
                    │  Upload / Camera / UI  │
                    └───────────┬────────────┘
                                │ HTTPS POST /predict (image)
                                ▼
                    ┌───────────────────────┐
                    │      Flask API         │
                    │  validate → orchestrate│
                    └───────────┬────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Preprocessing        │
                    │  OpenCV: resize, CLAHE,│
                    │  normalize (x2 sizes)  │
                    └───────────┬────────────┘
                                ▼
                    ┌───────────────────────┐
                    │  Layer 1: MobileNetV3  │
                    │  (classifier)          │
                    │  + SSDLite (bbox only) │
                    └───────────┬────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Confidence Engine      │
                    │ Entropy Engine         │
                    └───────────┬────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Adaptive Decision      │
                    │ Engine                 │
                    └─────┬─────────────┬────┘
                     ACCEPT│             │ESCALATE
                          │             ▼
                          │   ┌───────────────────────┐
                          │   │ Layer 2: RT-DETR       │
                          │   │ (classifier)           │
                          │   └───────────┬────────────┘
                          │               ▼
                          │   ┌───────────────────────┐
                          │   │ Confidence Engine      │
                          │   │ Entropy Engine (again) │
                          │   └───────────┬────────────┘
                          ▼               ▼
                    ┌───────────────────────┐
                    │   Result Generator     │
                    │ label + conf + risk +  │
                    │ uncertain + source     │
                    └───────────┬────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ JSON Response → UI     │
                    └───────────────────────┘
```

**Connections explained:**
- Frontend → API: one HTTP call per analysis (`POST /predict`, multipart image). The frontend never talks to the models directly — this keeps model logic swappable without touching the UI.
- API → Preprocessing: every request is preprocessed twice in parallel (224² and 640²) so Layer 2 doesn't have to wait on a second round-trip if escalation is needed.
- Preprocessing → Layer 1: always runs; this is the "fast path" that every image goes through.
- Layer 1 → Confidence/Entropy Engine: these are stateless pure functions, reused identically for Layer 2's output later — this is what keeps the two layers comparable.
- Confidence/Entropy → Decision Engine: the single choke point implementing the core innovation (Section 9). Nothing else in the system makes a routing decision.
- Decision Engine → Layer 2 (conditional): only invoked on escalation; this is what gives the ~60–70% average compute savings claimed in the proposal.
- Layer 2 output re-enters the same Confidence/Entropy Engine so the Result Generator has a uniform input shape regardless of which layer produced the final answer.
- Decision Engine / Layer 2 → Result Generator: assembles the response contract in Section 10.4, always including which layer produced the final result.
- Result Generator → Frontend: the frontend renders exclusively from this JSON — no business logic (thresholds, risk mapping) lives in the UI layer.

---

## 4. TECHNOLOGY STACK

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.10+ | Single language across ML, backend, and (via Streamlit) frontend — minimizes context-switching for a 5-person, 16-day team. |
| Frontend | Streamlit | Fastest path to a working, polished-looking UI without a separate JS build step; proposal already lists it as first choice. |
| Backend | Flask | Lightweight, matches proposal, trivial to containerize, no unnecessary framework overhead for a single `/predict` endpoint plus two utility endpoints. |
| ML Framework | PyTorch 2.x | Native support for both TorchVision (MobileNetV3, SSDLite) and Ultralytics RT-DETR; team's stated familiarity per proposal. |
| CNN | MobileNetV3-Large (TorchVision, ImageNet-pretrained) | Purpose-built for edge/low-latency inference; directly named in proposal. |
| Detection (display-only) | SSDLite320-MobileNetV3 (TorchVision, COCO-pretrained) | Provides a bounding-box overlay for the demo UI without requiring bbox-labeled training data we don't have. |
| Transformer | RT-DETR-L (Ultralytics, pretrained, fine-tuned classification head) | Matches proposal; best available real-time DETR variant with existing pretrained weights to fine-tune from. |
| Image Processing | OpenCV | Named in proposal; handles CLAHE, resizing, camera-feed capture. |
| Database | Supabase (Postgres) — optional, post-core | Only used for anonymous metric logging if time remains; not required for the core demo. |
| API | Flask REST (JSON over HTTPS) | Simple contract, easy to test with `curl`/Postman during integration. |
| Model Serving | Direct PyTorch inference in-process (ONNX export for Layer 1 only, as a stretch goal for the edge-deployment story) | Avoids standing up a separate serving process (Torchserve/Triton) that the team doesn't have time to operate; ONNX export for MobileNetV3 is cheap and demonstrates the edge-deployment claim from the proposal without needing it in the live demo path. |
| Deployment | Docker Compose (backend + frontend), single VM/laptop for live demo | Reproducible for judges without cloud dependency risk during the actual 48-hour presentation. |
| Version Control | GitHub (monorepo) | Named in proposal; single repo avoids cross-repo integration overhead for a 5-person team. |

---

## 5. AI TOOL STACK

**Decision: Google Antigravity is the single primary coding agent for this repository.** No other AI coding agent (Claude Code, GitHub Copilot, Bolt.new) writes into the same codebase, to avoid conflicting edits across 5 people. Claude (chat) is used only for planning/review/unblocking, never for direct commits.

| Tool | Role in project | When to use | Output |
|---|---|---|---|
| Claude (this chat) | Architecture, planning, blueprint review, unblocking design questions, converting this blueprint into task-level coding prompts later. | Before implementation starts, and whenever a design decision needs to be revisited mid-build. | Specs, task breakdowns, review notes — not code committed to the repo. |
| Google Antigravity | Primary implementation agent: ML fine-tuning scripts, Flask API, Streamlit frontend, integration glue. | For all repo code changes, from Day 3 onward. | Committed code across `ml/`, `backend/`, `frontend/`. |
| GitHub Copilot | Inline autocomplete only, for individual contributors typing code Antigravity didn't generate (e.g., quick test cases). Not a second "agent" driving structural changes. | Optional, individual-contributor convenience only. | Inline suggestions, no ownership of files. |
| Magic Patterns | UI design exploration for the Streamlit screens (Section 11) — mockups and visual direction, not shipped code. | Day 1–2, before frontend implementation starts, to lock a visual direction. | Static design references (screens/components) the frontend owner translates into Streamlit. |
| Bolt.new | Not used. Redundant with Antigravity as a second full-stack code generator; would create merge conflicts and split ownership. | N/A | N/A |
| Supabase | Optional persistence layer for anonymous session metrics (Section 12), nothing else. | Only after core pipeline (Milestones 1–5) is done, if time remains. | Postgres tables, no image data. |
| OpenRouter | Optional LLM explanation layer (Section 13) that turns the structured prediction into a plain-language summary. Never used for classification. | Only after core pipeline is done, if time remains. | Human-readable explanation string appended to the API response. |

---

## 6. DATASET REQUIREMENTS

### 6.1 Dataset

| Dataset | Source | Classes | Images | Used for |
|---|---|---|---|---|
| HAM10000 | ISIC / Kaggle | 7 (melanocytic nevi, melanoma, BCC, actinic keratosis, benign keratosis, dermatofibroma, vascular lesions) | 10,015 | Layer 1 (MobileNetV3) primary training set. |
| ISIC 2019 | ISIC Archive | 8 disease categories | 25,331 | Layer 2 (RT-DETR) fine-tuning set, prioritizing the classes with lowest Layer-1 confidence (ambiguous/rare cases). |
| ISIC 2020 | ISIC Archive | Binary (malignant/benign) + multi-class subset | 33,126 | Supplementary data for class balancing and threshold validation (Section 8), not full retraining given time budget. |

To keep the demo coherent, the **final class taxonomy is the 7 HAM10000 classes**; ISIC 2019/2020 images are re-mapped or filtered to this taxonomy rather than expanding it, since introducing new label sets mid-project would break the shared Confidence/Entropy Engine contract.

### 6.2 Data Pipeline

```
Raw Dataset (HAM10000 + ISIC subset)
        ↓
Validation (corrupt-file check, label sanity check)
        ↓
Cleaning (dedupe, remove unreadable images, remap ISIC labels → HAM10000 taxonomy)
        ↓
Train / Validation / Test split (stratified by class, 70 / 15 / 15)
        ↓
Preprocessing (resize, normalize, CLAHE)
        ↓
Augmentation (train split only)
        ↓
Training (Layer 1 first, Layer 2 second)
```

- **Image size:** 224×224 for MobileNetV3; 640×640 for RT-DETR.
- **Normalization:** ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`).
- **Augmentation (train only):** random horizontal/vertical flip, rotation ±30°, color jitter, Gaussian blur, Gaussian noise injection.
- **Class balancing:** class-weighted loss (inverse frequency) rather than oversampling, to avoid inflating the already-small rare classes with near-duplicate augmented copies.
- **Split:** 70% train / 15% validation / 15% test, stratified by class, fixed random seed committed to the repo (`configs/seed.yaml`) so results are reproducible across the team.
- **Format:** images stored as JPEG; labels in a single `labels.csv` (`image_id, class, split`).
- **Directory structure:**

```
data/
├── raw/
│   ├── ham10000/
│   └── isic_2019_2020/
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
├── labels.csv
└── class_map.json
```

---

## 7. ML MODEL REQUIREMENTS

### 7.1 Layer 1 — MobileNetV3-Large + SSDLite

- **Pretrained weights:** TorchVision ImageNet-pretrained `MobileNet_V3_Large_Weights.IMAGENET1K_V2`.
- **Architecture:** MobileNetV3-Large backbone; final classifier layer replaced with a `Linear(960, 7)` head for the 7 HAM10000 classes.
- **Input size:** 224×224×3.
- **Output classes:** 7 (Section 6.1 taxonomy).
- **Training/fine-tuning:** Two-stage transfer learning — (1) freeze backbone, train classifier head for 5 epochs; (2) unfreeze last 2 backbone blocks, fine-tune full network at a lower LR for remaining epochs.
- **Loss:** Class-weighted cross-entropy.
- **Optimizer:** AdamW.
- **Learning rate:** 1e-3 (head-only stage) → 1e-4 (fine-tune stage), cosine decay.
- **Batch size:** 32 (adjust down to 16 if GPU memory-constrained on Colab T4).
- **Epochs:** 15–20 total, with early stopping on validation macro-F1 (patience 3).
- **Augmentation:** As specified in Section 6.2, applied to train split only.
- **Checkpoint format:** `.pt` (state_dict) saved as `layer1_mobilenetv3_best.pt`, plus an ONNX export `layer1_mobilenetv3.onnx` as the stretch-goal edge artifact.
- **Evaluation metrics:** Top-1 accuracy, macro-F1, per-class recall (to catch rare-class collapse), Expected Calibration Error (ECE).
- **Target:** >85% top-1 validation accuracy (per proposal).
- **SSDLite:** used **as-is**, COCO-pretrained, no fine-tuning — its only job is producing a display bounding box, not a classification signal.

### 7.2 Layer 2 — RT-DETR-L

- **Model variant:** RT-DETR-L (Ultralytics implementation).
- **Pretrained weights:** COCO-pretrained checkpoint as the starting point; detection head discarded, a classification head (`Linear` layer on the pooled encoder output) attached for the 7-class taxonomy.
- **Input size:** 640×640×3.
- **Training/fine-tuning:** Fine-tune only on the subset of ISIC 2019 images that fall into low-Layer-1-confidence buckets (i.e., train Layer 2 specifically on the hard cases it will actually see in production, not the full dataset) — this both matches the escalation use case and keeps training time within the 16-day budget.
- **Dataset:** ISIC 2019 low-confidence subset (identified by running the trained Layer 1 model over ISIC 2019 and keeping the bottom ~30% by confidence), remapped to the shared taxonomy.
- **Output:** Same 7-class probability vector schema as Layer 1, so the Confidence/Entropy Engine works unchanged.
- **Checkpoint format:** `.pt`, saved as `layer2_rtdetr_best.pt`.
- **Evaluation metrics:** Top-1 accuracy and macro-F1 **on the low-confidence subset specifically** (this is the number that matters — it must beat Layer 1's accuracy on the same subset, or the escalation path isn't earning its latency cost), plus overall latency (ms/image).

---

## 8. CONFIDENCE + UNCERTAINTY REQUIREMENTS

### 8.1 Confidence

```
logits
   ↓ softmax
per-class probabilities  p = [p1, p2, ..., p7]
   ↓ argmax + max
top-1 class label, top-1 probability
   ↓
confidence = top-1 probability   (range [0, 1])
```

### 8.2 Entropy

Shannon entropy over the softmax distribution, **normalized** by `log(num_classes)` so the value is comparable across the 7-class taxonomy regardless of how many classes exist:

```
H(p) = -Σ pᵢ log(pᵢ)        (unnormalized, range [0, log(7)])
H_norm(p) = H(p) / log(7)   (normalized, range [0, 1])
```

We use `H_norm` everywhere in the Decision Engine so the entropy threshold is a clean, class-count-independent number.

### 8.3 Threshold selection (do not hardcode 85% blindly)

The proposal specifies an 85% confidence threshold and an entropy component, but a hackathon-appropriate way to justify (not just assert) both numbers is:

1. Run the trained Layer 1 model over the **validation split** and record `(confidence, entropy, correct/incorrect)` for every sample.
2. Sweep the confidence threshold from 0.5 to 0.95 in 0.05 steps and the entropy threshold from 0.1 to 0.6 in 0.05 steps.
3. For each `(conf_threshold, entropy_threshold)` pair, compute: accuracy of the accepted (fast-path) subset, % of validation set escalated, and accuracy of the escalated subset if Layer 2 were applied.
4. Pick the pair that (a) keeps fast-path accuracy ≥ 90% (i.e., the accepted set is genuinely reliable) while (b) keeping the escalation rate in the 20–35% range (so the ~60–70% average-compute-savings claim in the proposal is actually achievable, not accidental).
5. Report the chosen thresholds and the sweep chart in the evaluation deliverable — this is the "validation" the brief asks for, and it directly produces one of the demo's headline numbers.

**Calibration / temperature scaling:** Optional stretch goal. If ECE on the validation set is high (models are typically overconfident, per Guo et al. cited in the proposal), fit a single temperature scalar `T` post-hoc on the validation logits (`softmax(logits / T)`) and re-run the threshold sweep in step 2 with calibrated probabilities. Only attempt this after Milestone 3 is otherwise complete — it is a refinement, not a blocker.

---

## 9. ADAPTIVE DECISION ENGINE

```
Input image
     ↓
Preprocess (both resolutions)
     ↓
Layer 1: MobileNetV3 → probabilities
     ↓
confidence = top-1(probabilities)
entropy = H_norm(probabilities)
     ↓
Decision:

IF confidence >= CONF_THRESHOLD AND entropy <= ENTROPY_THRESHOLD:
    → route = "accepted", final_source = "layer1"

ELSE IF Layer 2 available:
    → run Layer 2 (RT-DETR) → probabilities_2
    → confidence_2 = top-1(probabilities_2)
    → entropy_2 = H_norm(probabilities_2)
    → route = "escalated", final_source = "layer2"

    IF confidence_2 < UNCERTAIN_FLOOR:
        → uncertain = True   (flag as "uncertain/unknown", do NOT force a class)
    ELSE:
        → uncertain = False

ELSE (Layer 2 unavailable / failed to load / errored):
    → route = "escalated_failed"
    → final_source = "layer1"
    → uncertain = True   (fallback: report Layer 1's result but explicitly mark low-confidence)
```

**Exact conditions (config-driven, not hardcoded):**
- `CONF_THRESHOLD` — from Section 8.3 sweep (proposal's starting point: 0.85).
- `ENTROPY_THRESHOLD` — from Section 8.3 sweep.
- `UNCERTAIN_FLOOR` — a lower bound (e.g., 0.5) applied to Layer 2's own confidence; if even the deep model isn't confident, the system should say "uncertain" rather than assert a diagnosis. This is the concrete implementation of the proposal's "closed-set mitigation" claim.

**Other defined states:**
- **Uncertain result:** `confidence_2 < UNCERTAIN_FLOOR` after escalation — returned with `uncertain: true` and no forced class commitment (UI shows "possible conditions" ranked list instead of a single label).
- **Unknown/OOD result:** Same code path as "uncertain" for the hackathon scope — the team does not implement a separate OOD detector (e.g., Mahalanobis distance) given the time budget; this is noted as a Future Enhancement (Section 24 of the proposal), not part of the MVP decision logic.
- **Model failure:** Any exception raised during Layer 1 or Layer 2 inference is caught at the API layer; Layer 1 failure → HTTP 500 (nothing can be returned); Layer 2 failure during escalation → fallback branch above (never a 500 to the user just because the deep model failed).
- **Invalid input:** Caught before any model runs (FR-03/FR-04/FR-05); returns HTTP 400, never reaches the Decision Engine.
- **Fallback behavior:** Documented above — always degrade to "best available answer + explicit uncertainty flag," never a silent wrong answer and never a hard crash.

---

## 10. API REQUIREMENTS

### 10.1 `POST /predict`
- **Purpose:** Run the full adaptive pipeline on a single image.
- **Request:** `multipart/form-data`, field `image` (JPEG/PNG/TIFF).
- **Response:** JSON, schema in 10.4.
- **Status codes:** `200` success (accepted or escalated, including uncertain), `400` invalid input (bad format, too small, not an image), `422` input rejected by sanity check (non-skin image), `500` internal model failure on Layer 1 (unrecoverable), `503` Layer 2 unavailable (still returns `200` with fallback per Section 9 — `503` is reserved for the whole service being unavailable, not a single-layer degradation).
- **Errors:** `{ "error": "invalid_image", "message": "..." }` shape, consistent across all 4xx/5xx cases.
- **Dependencies:** Preprocessing, Layer 1, Decision Engine, (conditionally) Layer 2, Result Generator.

### 10.2 `GET /health`
- **Purpose:** Liveness check for the demo (also lets the frontend show a "backend offline" state instead of hanging).
- **Request:** None.
- **Response:** `{ "status": "ok", "layer1_loaded": true, "layer2_loaded": true }`.
- **Status codes:** `200` if the API process is up (even if a model failed to load — that's reported in the body, not the status code).
- **Errors:** N/A (this endpoint itself should not fail).
- **Dependencies:** None beyond process being alive.

### 10.3 `GET /model-info`
- **Purpose:** Debug/demo transparency — show judges the active thresholds and class list without reading code.
- **Request:** None.
- **Response:** `{ "classes": [...7 names...], "conf_threshold": 0.85, "entropy_threshold": 0.35, "uncertain_floor": 0.5, "layer1_version": "...", "layer2_version": "..." }`.
- **Status codes:** `200`.
- **Errors:** N/A.
- **Dependencies:** Config loader.

### 10.4 Prediction response schema

```json
{
  "prediction": "melanoma",
  "confidence": 0.913,
  "entropy": 0.21,
  "model_used": "layer2_rtdetr",
  "escalated": true,
  "risk_level": "high",
  "uncertain": false,
  "layer1": {
    "prediction": "melanocytic_nevus",
    "confidence": 0.61,
    "entropy": 0.44
  },
  "bounding_box": { "x": 112, "y": 84, "w": 96, "h": 101 },
  "processing_time_ms": 1840,
  "disclaimer": "This result is not a medical diagnosis. Consult a dermatologist."
}
```

`layer1` block is always included (even when `escalated: false`, where it duplicates the top-level fields) so the frontend routing animation always has both numbers to display.

---

## 11. FRONTEND REQUIREMENTS

**Screen 1 — Landing / Home**
- Components: Title, one-line description of the system, "Start Screening" button, disclaimer footer.
- Buttons: "Start Screening."
- Info displayed: What the tool does, that it is not a diagnosis.
- API interaction: None.
- Loading/error state: N/A.
- Animation: Subtle fade-in only — first impressions, keep it clean.

**Screen 2 — Image Upload**
- Components: Drag-and-drop upload zone, "Use Camera" toggle, image preview thumbnail, "Analyze" button.
- Buttons: Upload, Use Camera, Analyze, Clear/Retake.
- Info displayed: Selected image preview, format/size validation errors inline.
- API interaction: None yet (validation is client-side pre-check; server-side validation happens on submit).
- Loading state: Disabled "Analyze" button until a valid image is present.
- Error state: Inline red text under the upload zone for invalid files.

**Screen 3 — Analysis / Processing**
- Components: Progress indicator, live status text.
- Buttons: None (or a "Cancel" if feasible).
- Info displayed: "Fast Analysis..." status text while waiting on `/predict`.
- API interaction: `POST /predict` fired on entry; screen transitions automatically based on response.
- Loading state: Spinner/progress bar.
- Error state: If the API call fails outright (network/500), show retry button.
- Animation: This is the screen that carries the routing visualization:

```
FAST ANALYSIS
      ↓
Confidence: 62%
      ↓
Low confidence
      ↓
ESCALATING...
      ↓
ADVANCED ANALYSIS
      ↓
Final Result
```
Implemented as a sequential reveal driven by the single `/predict` response (not multiple round-trips) — the animation timing is client-side pacing over the already-returned data, not a live multi-step API conversation, to keep the backend simple.

**Screen 4 — Fast-path result** (shown when `escalated: false`)
- Components: Result card (disease label, confidence bar, risk badge), bounding-box overlay on the image, "Analyze Another" button.
- Buttons: Analyze Another, (optional) "Show details."
- Info displayed: `prediction`, `confidence`, `risk_level`, `model_used: layer1`, `processing_time_ms`.
- API interaction: None (renders from the already-fetched response).
- Loading/error state: N/A.
- Animation: Confidence bar fills to the reported value.

**Screen 5 — Escalation state** (transient, part of Screen 3's sequence when `escalated: true`)
- Components: "Escalating to advanced analysis..." message, Layer 1's low-confidence number shown briefly for contrast.
- Info displayed: `layer1.confidence`, `layer1.entropy`.
- Animation: This is what makes the core innovation visible — must not be skipped or instant, even though the underlying API call already returned; pace it (~1–2s) so judges can read it.

**Screen 6 — Advanced result** (shown when `escalated: true`)
- Components: Result card (disease label, confidence bar, risk badge), "Advanced Analysis" badge distinguishing it from Screen 4, side-by-side comparison of Layer 1 vs Layer 2 confidence, bounding-box overlay, "Analyze Another" button.
- Info displayed: `prediction`, `confidence`, `risk_level`, `model_used: layer2_rtdetr`, `layer1` sub-object for comparison, `uncertain` flag (if true, show a ranked-possibilities list instead of a single confident label).
- API interaction: None (renders from the already-fetched response).
- Error state: If `uncertain: true`, replace the single-label card with an "inconclusive — please consult a dermatologist" state rather than forcing a class.

---

## 12. DATABASE / SUPABASE REQUIREMENTS

**Decision: Supabase is EXCLUDED from the core MVP.** The proposal's own security section states no patient images should be persisted, and the core pipeline (Sections 3–10) has no functional dependency on a database — every response is generated and consumed within a single request. Adding Supabase to the critical path would introduce a dependency (network call, schema migrations, auth) with no benefit to the demo's central claim (adaptive routing).

**If time remains after Milestone 6 (optional, post-MVP):**
- **Why:** A "session stats" panel (e.g., "12 images analyzed this session, 4 escalated") is a nice-to-have that shows the system working over multiple runs, useful for Q&A but not for the core demo scenarios.
- **Tables:** `inference_logs (id, timestamp, predicted_class, confidence, entropy, escalated, model_used, processing_time_ms)`.
- **Fields:** As above — no image data, no user identifiers.
- **Relationships:** Single flat table, no joins needed.
- **RLS:** Public insert (from backend service role only, not client-side), public read for the demo dashboard, no per-user isolation needed since there is no user concept.
- **What is stored:** Aggregate metrics only, as listed above.
- **What is NEVER stored:** Raw images, image derivatives/embeddings that could reconstruct the image, any patient-identifying information.

---

## 13. OPENROUTER / LLM REQUIREMENTS

**Decision: OPTIONAL — POST-MVP.**

**If added:**
- **Exact feature:** After the Result Generator produces the structured JSON, an optional call to an LLM (via OpenRouter) converts the structured result into one short, plain-language paragraph (e.g., "This looks like it could be melanoma, based on visual features the system detected with high confidence. This is not a diagnosis — please see a dermatologist promptly.").
- **Where it sits in architecture:** A thin post-processing step after the Result Generator, before the response is returned to the frontend — it never touches raw pixels and never influences `prediction`/`confidence`/`risk_level`.
- **Input:** The structured JSON from Section 10.4 (text only).
- **Output:** A single `explanation: string` field appended to the response.
- **Model requirement:** Any small, fast instruction-tuned model available via OpenRouter (cost/latency, not reasoning depth, is the constraint here).
- **Fallback:** If the OpenRouter call fails or times out, the `explanation` field is simply omitted — the rest of the response (which drives the actual UI) is unaffected. This must be implemented as fire-and-forget with a short timeout so it can never slow down or break the core demo path.
- The LLM never performs classification and is never on the critical path for Milestones 1–6.

---

## 14. FOLDER STRUCTURE

```
amsdds/
│
├── frontend/
│   ├── app.py                  # Streamlit entrypoint, screen router
│   ├── screens/                # One file per screen (Section 11)
│   ├── components/             # Reusable UI pieces (confidence bar, routing animation)
│   └── api_client.py           # Wraps calls to the Flask API
│
├── backend/
│   ├── app.py                  # Flask app, route registration
│   ├── routes/
│   │   ├── predict.py
│   │   ├── health.py
│   │   └── model_info.py
│   ├── pipeline/
│   │   ├── preprocessing.py
│   │   ├── confidence.py
│   │   ├── entropy.py
│   │   ├── decision_engine.py
│   │   └── result_generator.py
│   └── config.py               # Loads configs/thresholds.yaml
│
├── ml/
│   ├── preprocessing/          # Shared with backend/pipeline/preprocessing.py via package import
│   ├── models/
│   │   ├── layer1_mobilenetv3.py
│   │   └── layer2_rtdetr.py
│   ├── inference/
│   │   ├── layer1_wrapper.py
│   │   └── layer2_wrapper.py
│   ├── uncertainty/             # Threshold sweep script (Section 8.3)
│   └── evaluation/              # evaluate.py, metrics reports
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_confidence_entropy.py
│   ├── test_decision_engine.py
│   └── test_api.py
│
├── notebooks/                   # Colab/Kaggle training notebooks (Layer 1, Layer 2, threshold sweep)
├── scripts/                     # Dataset download/cleaning scripts
├── configs/
│   ├── thresholds.yaml          # CONF_THRESHOLD, ENTROPY_THRESHOLD, UNCERTAIN_FLOOR
│   ├── classes.json             # 7-class taxonomy + risk-level mapping
│   └── seed.yaml
├── assets/                      # Demo test images (Section 19)
├── docs/                        # This blueprint, report, references
│
├── requirements.txt
├── docker-compose.yml
├── .env.example
└── README.md
```

- `frontend/` — owner: frontend engineer (Section 15, Person D).
- `backend/` — owner: backend engineer (Person C).
- `ml/` — owner: ML pod (Persons A & B).
- `tests/` — shared, each owner writes tests for their own module.
- `configs/` — shared source of truth; **thresholds live here, never hardcoded in Python** (ties back to NFR-Maintainability).

---

## 15. FIVE-PERSON TEAM DIVISION

Four parallel workstreams, five people — the two ML roles (A and B) run in parallel on the same pod but own distinct files, so neither blocks the other; everyone else runs independently of the ML training timeline by working against a stub API from Day 2.

**Person A — Ruhani Kumar (Team Lead) — ML Core: Layer 1 + Confidence/Entropy/Decision Engine**
- Responsibility: The core innovation itself.
- Tasks: Fine-tune MobileNetV3 (Section 7.1), implement Confidence/Entropy Engines (Section 8), implement the Adaptive Decision Engine (Section 9), run the threshold sweep (Section 8.3).
- Deliverables: `layer1_mobilenetv3_best.pt`, `ml/inference/layer1_wrapper.py`, `backend/pipeline/confidence.py`, `backend/pipeline/entropy.py`, `backend/pipeline/decision_engine.py`, threshold sweep report.
- Files owned: `ml/models/layer1_mobilenetv3.py`, `ml/inference/layer1_wrapper.py`, `backend/pipeline/confidence.py`, `entropy.py`, `decision_engine.py`, `configs/thresholds.yaml`.
- Dependencies: Needs cleaned/split dataset from Person E by Day 4. Also owns overall integration sign-off since this is the innovation the demo hinges on.

**Person B — Kashish Mittal — ML Escalation: SSDLite integration + Layer 2 (RT-DETR)**
- Responsibility: The escalation path.
- Tasks: Wire up pretrained SSDLite for bounding boxes (Section 7.1), identify the low-confidence ISIC subset (needs Person A's trained Layer 1 checkpoint), fine-tune RT-DETR classification head (Section 7.2).
- Deliverables: `layer2_rtdetr_best.pt`, `ml/inference/layer2_wrapper.py`, SSDLite bbox integration.
- Files owned: `ml/models/layer2_rtdetr.py`, `ml/inference/layer2_wrapper.py`, bounding-box utility in `backend/pipeline/preprocessing.py` (shared, coordinate with Person E).
- Dependencies: Needs Person A's Layer 1 checkpoint by Day 7 to mine the hard-example subset; needs Person E's ISIC pipeline by Day 4.

**Person C — Lakshmishree K P — Backend / API**
- Responsibility: Flask API and orchestration.
- Tasks: Build `/predict`, `/health`, `/model-info` (Section 10) against a stub Layer 1/Layer 2 (hardcoded fake responses) starting Day 2, swap in real model wrappers as Persons A/B deliver them, implement error handling and the fallback branch (Section 9).
- Deliverables: Working Flask API, API test suite.
- Files owned: `backend/app.py`, `backend/routes/*`, `backend/config.py`.
- Dependencies: Needs the JSON schema (Section 10.4) locked by Day 1 (no code dependency — can build against a mock from Day 2), needs real wrappers from A/B by Day 9.

**Person D — Shreya Jaganatha Gowda — Frontend**
- Responsibility: Streamlit UI, including the routing-visualization animation.
- Tasks: Build all 6 screens (Section 11) against a mocked API response from Day 2, use Magic Patterns for visual direction on Day 1–2, wire to the real API once Person C's endpoints are live.
- Deliverables: Working Streamlit app.
- Files owned: `frontend/*`.
- Dependencies: Needs the JSON schema locked by Day 1 (same as Person C — this is why the schema is finalized before any model code is written); needs live API by Day 10.

**Person E — Joan Sara Joe — Data Pipeline + Testing/Deployment**
- Responsibility: Everything the ML pod depends on, plus quality gates for the whole team.
- Tasks: Dataset acquisition and cleaning (Section 6), build the shared `preprocessing.py` module (used by both backend and ml/), write the test suite skeleton (Section 18) that other owners fill in for their modules, own Docker/deployment setup, own the demo test-image set (Section 19).
- Deliverables: Cleaned/split dataset, `preprocessing.py`, `docker-compose.yml`, curated demo images, test suite scaffolding.
- Files owned: `scripts/*`, `ml/preprocessing/`, `backend/pipeline/preprocessing.py`, `docker-compose.yml`, `assets/`, `tests/` scaffolding.
- Dependencies: None upstream — this is the Day-1 critical path everyone else waits on for real data; must be substantially done by Day 4.

**Integration points:**
- Day 1: JSON schema (Section 10.4) and thresholds config format locked by the whole team — this is what lets C and D start immediately without waiting on A/B.
- Day 4: Person E delivers cleaned/split dataset — unblocks A and B's real training.
- Day 9–10: A/B's real model wrappers replace C's stub in the Flask API — first true end-to-end run.
- Day 11: D's frontend connects to the real (not mocked) API.
- Day 12+: Whole-team integration testing (Section 18), then polish.

---

## 16. 16-DAY IMPLEMENTATION PLAN

| Day | Main Goal | Ruhani (A) | Kashish (B) | Lakshmishree (C) | Shreya (D) | Joan (E) | Deliverable |
|---|---|---|---|---|---|---|---|
| 1 | Lock contracts | Finalize JSON schema & thresholds config format with team | Same | Same | Same, Magic Patterns exploration starts | Set up repo structure (Section 14), start dataset download | Locked API schema, repo skeleton |
| 2 | Stubs unblock non-ML work | Set up training notebook skeleton | Set up RT-DETR notebook skeleton | Build `/predict` against hardcoded mock response | Build Screen 1–2 against mock API | Continue dataset cleaning, write `preprocessing.py` v1 | Working mock API + first 2 screens |
| 3 | Preprocessing ready | Review preprocessing output, start EDA on HAM10000 | Review preprocessing output | Build `/health`, `/model-info` | Build Screen 3 (loading state) | Finish `preprocess.py`, unit tests | `preprocessing.py` complete, tested |
| 4 | Dataset split delivered | Begin Layer 1 head-only training | Wait on Layer 1 checkpoint; prep ISIC 2019 remap script | Wire preprocessing into backend | Build Screen 4 (fast-path result) | Deliver clean train/val/test split | Dataset split + first training run started |
| 5 | Layer 1 fine-tuning | Continue Layer 1 fine-tune (stage 2) | Finish ISIC remap script | Add error handling (Section 9 fallback) | Build Screen 5 (escalation animation) | Docker Compose skeleton | Layer 1 mid-training checkpoint |
| 6 | Layer 1 near-final | Finish Layer 1 training, run eval metrics | Wire SSDLite (pretrained, no training needed) | Integrate real Layer 1 wrapper (early version) into API | Build Screen 6 (advanced result) | Curate first-pass demo test images | Layer 1 checkpoint + eval report |
| 7 | Threshold sweep + hard-example mining | Run threshold sweep (Section 8.3), pick thresholds | Mine low-confidence ISIC subset using Layer 1 | Update `/model-info` with real thresholds | Wire routing animation to real `escalated` flag | Expand test suite (preprocessing, confidence/entropy) | Chosen CONF/ENTROPY thresholds committed to `configs/` |
| 8 | Decision Engine live | Implement & unit-test Decision Engine | Start RT-DETR fine-tuning | Integrate Decision Engine into `/predict` | Polish Screens 4–6 visuals | CI-style test run across team's modules so far | Decision Engine merged, first true accept/escalate branch working |
| 9 | Layer 2 fine-tuning continues | Support B on eval subset definition | Continue RT-DETR fine-tuning | First full backend integration test (Layer 1 only path) | Connect frontend to real (non-mock) `/predict` for accepted cases | Fix bugs surfaced by integration test | End-to-end fast-path demo works |
| 10 | Layer 2 checkpoint | Idle/buffer — support integration | Finish RT-DETR fine-tune, run eval | Integrate Layer 2 wrapper + fallback logic (Section 9) | Connect escalation path in frontend | Full-team integration test #1 | End-to-end escalation-path demo works |
| 11 | Full pipeline integration | Bug triage across pipeline | Bug triage across pipeline | Bug triage, latency profiling | Bug triage, animation timing polish | Own integration test log, track open issues | First fully working end-to-end pipeline (both paths) |
| 12 | Evaluation | Produce final accuracy/F1/AUC/ECE report | Produce Layer 2-specific hard-subset report | Latency measurement across both paths | UI polish pass | Run evaluation script (Section 11 of components), package numbers | Evaluation report (Sections 18/19 inputs) |
| 13 | Optional features (if on schedule) | Support OpenRouter integration if time allows | Support Supabase logging if time allows | Implement OpenRouter/Supabase hooks (optional, Sections 12/13) | Session-stats panel (optional) | Finalize demo test-image set (Section 19) | Optional features gated on core being done |
| 14 | Testing | Run adaptive pipeline test matrix (Section 18) | Same | Same | Same | Own test execution, log results | All test cases in Section 18 passing |
| 15 | Polish + demo rehearsal | Rehearse demo scenarios A & B | Same | Same | Same, fix any last UI issues | Same, verify Docker demo path works offline | Rehearsed demo, fallback plan ready |
| 16 | Final demo prep & documentation | Finalize report/docs | Same | Same | Same | Own README + docs folder | Submission-ready repo + demo |

---

## 17. MILESTONES

**Milestone 1 — Image → model → prediction**
- Deadline: Day 6.
- Requirements: Preprocessing + Layer 1 wrapper + a raw (unrouted) prediction returned from a script (not yet the API).
- Definition of done: Given a test image, the script prints a class label and raw probabilities.

**Milestone 2 — Prediction → confidence + entropy**
- Deadline: Day 7.
- Requirements: Confidence and Entropy Engines implemented and unit-tested; threshold sweep run.
- Definition of done: Given Milestone 1's output, the system prints confidence, entropy, and the chosen thresholds from the sweep.

**Milestone 3 — Adaptive routing**
- Deadline: Day 8.
- Requirements: Decision Engine implemented, integrated into the API's `/predict`, all routing branches (Section 9) covered.
- Definition of done: `/predict` returns `route: "accepted"` for a known easy test image and `route: "escalated"` for a known ambiguous test image, without Layer 2 needing to exist yet (stub Layer 2 returning a placeholder is acceptable at this milestone).

**Milestone 4 — RT-DETR escalation**
- Deadline: Day 10.
- Requirements: Real Layer 2 wrapper integrated, replacing the placeholder from Milestone 3.
- Definition of done: An escalated request returns a real RT-DETR-derived prediction, and its accuracy on the low-confidence subset beats Layer 1 alone on the same subset.

**Milestone 5 — Backend + frontend integration**
- Deadline: Day 11.
- Requirements: Frontend fully connected to the real API for both accepted and escalated paths; routing animation reflects real data.
- Definition of done: A user can complete the full flow in the browser for both an easy and an ambiguous test image, with no manual intervention.

**Milestone 6 — Complete demo**
- Deadline: Day 15.
- Requirements: Both demo scenarios (Section 19) rehearsed and reliable; evaluation numbers finalized; fallback plan tested.
- Definition of done: The team can run Scenario A and Scenario B back-to-back, live, with the metrics shown in Section 19.

---

## 18. TESTING REQUIREMENTS

**ML**
- Top-1 accuracy, macro-F1, per-class recall (Layer 1 and Layer 2, on held-out test split).
- Confusion matrix (Layer 1 and Layer 2).
- Calibration (ECE), before and after temperature scaling if attempted.
- Entropy distribution histogram, correct vs. incorrect predictions.
- Routing rate: % of test set escalated at chosen thresholds.
- Latency: p50/p95 inference time per layer, on the actual demo hardware.

**Backend**
- Valid image → 200 with correct schema.
- Invalid image (corrupt bytes) → 400.
- Unsupported format (e.g., `.bmp`) → 400.
- Low-resolution image (< 224×224) → 400.
- Model failure (simulate Layer 1 exception) → 500.
- Layer 2 failure during escalation (simulate exception) → 200 with fallback/`uncertain: true`, not a crash.
- API-level failure (Layer 2 process not started) → verify `/health` correctly reports `layer2_loaded: false`.

**Adaptive Pipeline**
- High confidence, low entropy → CNN result returned, `escalated: false`.
- Low confidence → RT-DETR invoked, `escalated: true`.
- High entropy (even with borderline confidence) → RT-DETR invoked.
- Invalid input → rejected before reaching Layer 1 (400/422, decision engine never invoked — verify via logs).
- Unknown/suspicious image (deliberately out-of-taxonomy test image, e.g., a photo of a wall) → `uncertain: true` after escalation, not a confidently wrong class.
- Transformer failure → fallback branch verified (Section 9's third case).

**Frontend**
- Upload flow (valid file).
- Loading state renders and clears correctly.
- Routing animation displays the correct branch (accepted vs. escalated) matching the API response.
- Result screen renders all required fields (Section 11) for both fast-path and escalated cases.
- Error states (network failure, backend down) render a retry option instead of hanging.
- Responsive layout check on at least one non-desktop viewport width, since judges may view on a laptop projected or a tablet.

---

## 19. DEMO REQUIREMENTS

**Scenario A — Easy Case**
```
Image (clear, well-lit, unambiguous lesion)
      ↓
MobileNetV3
      ↓
High confidence (≥ chosen threshold)
      ↓
Immediate result
```

**Scenario B — Ambiguous Case (the core innovation — most important)**
```
Image (borderline lesion, or deliberately harder lighting/crop)
      ↓
MobileNetV3
      ↓
Low confidence / high entropy
      ↓
Automatic escalation
      ↓
RT-DETR
      ↓
Final result
```

**Test images to prepare:**
- 2–3 curated "easy" images from the HAM10000 test split with known-correct, high-confidence Layer 1 predictions (Scenario A, with a backup in case one fails live).
- 2–3 curated "ambiguous" images from the mined low-confidence subset (Section 15, Person B's task) where Layer 2 is verified to correct or refine Layer 1's answer (Scenario B, with a backup).
- 1 deliberately out-of-scope image (e.g., a non-skin photo) to demonstrate the `uncertain`/rejection path live, if time in the demo slot allows.

**Expected behavior:** Exactly as specified in Sections 9 and 11 — no manual switch, the routing must be visibly automatic.

**Metrics to display during/after the demo:** Validation accuracy (Layer 1), accuracy on the hard subset (Layer 2 vs. Layer 1), chosen thresholds and how they were derived (Section 8.3), average latency for both paths, and the escalation-rate/compute-savings figure computed from the actual test run (framed honestly relative to the proposal's ~60–70% claim — report the team's own measured number, not the literature figure, once available).

**Latency to measure:** End-to-end (browser click to rendered result) for both scenarios, captured and displayed on Screen 4/6 (`processing_time_ms`).

**Fallback / offline-demo strategy:** Run the full stack locally via `docker-compose up` on the presenting laptop rather than depending on venue Wi-Fi/cloud hosting; keep the curated test images bundled in `assets/` so the demo never depends on a live upload working perfectly; if live camera capture is flaky on the venue's hardware, fall back to the pre-uploaded curated images without breaking the narrative.

---

## 20. FINAL IMPLEMENTATION CHECKLIST

- [ ] Repository (Section 14 structure created, `.gitignore` for model checkpoints/data)
- [ ] Environment (`requirements.txt`, `.env.example`, Colab/Kaggle notebook environment)
- [ ] Dataset (HAM10000 + ISIC 2019/2020 acquired, cleaned, split, `labels.csv`)
- [ ] Preprocessing (`preprocess.py`, CLAHE, dual-resolution resize, normalization, unit-tested)
- [ ] MobileNetV3 (fine-tuned, checkpoint saved, >85% val accuracy)
- [ ] SSDLite (pretrained weights wired in, bbox overlay working)
- [ ] Confidence Engine (implemented, unit-tested)
- [ ] Entropy Engine (implemented, unit-tested)
- [ ] Calibration (threshold sweep run and documented; temperature scaling if time allows)
- [ ] Decision Engine (all branches from Section 9 implemented and tested)
- [ ] RT-DETR (fine-tuned classification head, checkpoint saved, beats Layer 1 on hard subset)
- [ ] API (`/predict`, `/health`, `/model-info`, error handling, schema-conformant responses)
- [ ] Frontend (all 6 screens, connected to real API)
- [ ] Routing visualization (Screen 3/5 animation reflecting real `escalated` data)
- [ ] Testing (ML, backend, pipeline, frontend test matrices from Section 18 passing)
- [ ] Evaluation (accuracy/F1/AUC/ECE/latency/routing-rate report produced)
- [ ] Deployment (`docker-compose.yml` working locally, offline-demo path verified)
- [ ] Demo (Scenarios A & B rehearsed with curated test images, backups ready)
- [ ] Documentation (README, this blueprint, references, final report)
- [ ] Disclaimer present on every result screen (non-diagnostic, consult-a-dermatologist language)
- [ ] Config-driven thresholds verified — no hardcoded magic numbers in committed code
- [ ] Optional features (Supabase logging, OpenRouter explanation) explicitly deferred unless core checklist above is 100% complete by Day 12
