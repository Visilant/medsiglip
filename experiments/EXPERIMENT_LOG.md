# Experiment Log — MedSigLIP Fine-Tuning

> **Model:** google/medsiglip-448 (400M vision + 400M text encoder, 448x448, 64 text tokens)
> **Dataset:** Visilant ophthalmology slit-lamp images (96,927 clean / 86,672 train / 9,631 val)
> **Hardware:** 2x A100-40GB
> **Tracking:** [W&B project](https://wandb.ai/adi-visilant-visilant-inc/medsiglip-visilant)

---

## Status Dashboard

| ID | Phase | Strategy | Mode | LR | Epochs | Batch (eff) | GPU | Status | Val Loss | ZS Lens Acc | ZS Corneal Acc | Notes |
|----|-------|----------|------|----|--------|-------------|-----|--------|----------|-------------|----------------|-------|
| C1 | 1 | Full fine-tune | full | 1e-5 | 2 | 128 | 0+1 | pending | — | — | — | Baseline contrastive |
| C2 | 1 | LoRA r=16 | lora | 2e-4 | 3 | 64 | 1 | **running** | — | — | — | α=32, dropout=0.05 |
| C3 | 1 | Partial freeze | partial_freeze | 5e-5 | 3 | 64 | 0 | **running** | — | — | — | Last 6 vision layers |
| C4a | 1 | Caption: clinical | [best_mode] | [best_lr] | 1 | 64 | 0 | blocked(C1-C3) | — | — | — | Ablation on caption style |
| C4b | 1 | Caption: label_only | [best_mode] | [best_lr] | 1 | 64 | 0 | blocked(C1-C3) | — | — | — | |
| C4c | 1 | Caption: sentence | [best_mode] | [best_lr] | 1 | 64 | 0 | blocked(C1-C3) | — | — | — | |
| C5a | 1 | Data: 10K samples | [best_mode] | [best_lr] | 1 | 64 | 1 | blocked(C1-C3) | — | — | — | Data efficiency curve |
| C5b | 1 | Data: 30K samples | [best_mode] | [best_lr] | 1 | 64 | 1 | blocked(C1-C3) | — | — | — | |
| C5c | 1 | Data: 50K samples | [best_mode] | [best_lr] | 1 | 64 | 1 | blocked(C1-C3) | — | — | — | |
| C5d | 1 | Data: full | [best_mode] | [best_lr] | 1 | 64 | 1 | blocked(C1-C3) | — | — | — | |
| P1 | 2 | Linear probe | frozen | 1e-3 | 10 | 64/gpu | 0+1 | blocked(Phase 1) | — | — | — | Per-task (lens, corneal) |
| P2 | 2 | MLP head | frozen | 1e-4 | 15 | 64/gpu | 0+1 | blocked(Phase 1) | — | — | — | 1152→512→256→K |
| P3 | 2 | Full FT + head | full | 5e-5 | 3 | [tbd] | 0+1 | blocked(Phase 1) | — | — | — | |
| P4 | 2 | LoRA + head | lora | 2e-4 | 5 | [tbd] | 0+1 | blocked(Phase 1) | — | — | — | |

### Phase 1 Decision Gate
**Criteria:** Select the strategy (C1/C2/C3) with the best zero-shot macro-F1 on val set across both tasks. Use that mode + LR for C4/C5 ablations. Best Phase 1 checkpoint feeds Phase 2.

### Phase 2 Decision Gate
**Criteria:** Compare P1-P4 on held-out test set (628 images). Primary metric: macro-F1. Clinical metric: sensitivity at 95% specificity for mature_cataract, aphakia (lens), Active corneal infection (corneal).

---

## Phase 1: Contrastive Fine-Tuning

### C1 — Full Fine-Tune (Baseline)
- **Hypothesis:** Full parameter update gives strongest representation but may overfit on 87K images
- **Config:** `mode=full, lr=1e-5, epochs=2, batch_eff=128 (per_device=8, grad_accum=8), warmup=500`
- **Trainable params:** ~800M (both encoders)
- **Status:** pending
- **Results:**
  - Val loss: —
  - Zero-shot lens_status: acc=— | macro_f1=— | auroc=—
  - Zero-shot corneal: acc=— | macro_f1=— | auroc=—
- **W&B:** —
- **Checkpoint:** —
- **Notes:** —

### C2 — LoRA r=16
- **Hypothesis:** Parameter-efficient fine-tuning preserves pretrained features while adapting to domain; lower risk of catastrophic forgetting
- **Config:** `mode=lora, r=16, alpha=32, dropout=0.05, lr=2e-4, epochs=3, batch_eff=64 (per_device=16, grad_accum=4), warmup=300`
- **Trainable params:** ~6.3M (LoRA adapters on q/k/v/out projections)
- **Status:** **running** (GPU 1, launched 2026-03-07 ~05:30 UTC)
- **Results:**
  - Val loss: —
  - Zero-shot lens_status: acc=— | macro_f1=— | auroc=—
  - Zero-shot corneal: acc=— | macro_f1=— | auroc=—
- **W&B:** [contrastive-C2](https://wandb.ai/adi-visilant-visilant-inc/huggingface/runs/cgyif47k)
- **Checkpoint:** `experiments/outputs/C2/`
- **Notes:** Encountered truncated image error at eval step 200; mitigated via LOAD_TRUNCATED_IMAGES=True

### C3 — Partial Freeze
- **Hypothesis:** Freezing early vision layers + text encoder focuses capacity on high-level ophthalmology features while keeping compute manageable
- **Config:** `mode=partial_freeze, lr=5e-5, epochs=3, batch_eff=64 (per_device=4, grad_accum=16), warmup=400`
- **Trainable params:** ~120M (last 6 vision layers, layers 21-27 of 27)
- **Status:** **running** (GPU 0, launched 2026-03-07 ~05:30 UTC)
- **Results:**
  - Val loss: —
  - Zero-shot lens_status: acc=— | macro_f1=— | auroc=—
  - Zero-shot corneal: acc=— | macro_f1=— | auroc=—
- **W&B:** [contrastive-C3](https://wandb.ai/adi-visilant-visilant-inc/huggingface/runs/e9oikrlf)
- **Checkpoint:** `experiments/outputs/C3/`
- **Notes:** —

### C4 — Caption Style Ablation (after C1-C3 gate)
- **Hypothesis:** Rich clinical captions (demographics, VA, illumination) improve alignment vs. label-only text
- **Variants:**
  - **C4a (clinical):** "Slit lamp photo, {image_type} illumination, {dilation}. {age}yo {gender}. Lens: {lens}. Cornea: {cornea}. VA: {va}."
  - **C4b (label_only):** "Lens: {lens}. Cornea: {cornea}."
  - **C4c (sentence):** "Slit lamp photograph showing {lens} and {cornea}."
- **Config:** best mode/LR from C1-C3, 1 epoch, batch_eff=64
- **Status:** blocked (awaiting Phase 1 gate)
- **Results:** —

### C5 — Data Size Ablation (after C1-C3 gate)
- **Hypothesis:** MedSigLIP shows strong data efficiency; performance plateaus before using all 87K samples
- **Variants:** 10K / 30K / 50K / full (86,672)
- **Config:** best mode/LR from C1-C3, 1 epoch, batch_eff=64
- **Status:** blocked (awaiting Phase 1 gate)
- **Results:** —

---

## Phase 2: Classification

### P1 — Linear Probe (Frozen Backbone)
- **Hypothesis:** Frozen contrastive features + linear head establishes classification baseline; reveals embedding quality directly
- **Config:** `head=linear, backbone=frozen, lr=1e-3, epochs=10, batch=64/gpu`
- **Tasks:** lens_status (7-class), corneal (4-class) — trained separately
- **Status:** blocked (awaiting best Phase 1 checkpoint)
- **Results:** —

### P2 — MLP Head (Frozen Backbone)
- **Hypothesis:** Non-linear head captures class boundaries that linear probe misses, especially for rare classes
- **Config:** `head=mlp (1152→512→256→K), backbone=frozen, lr=1e-4, epochs=15`
- **Status:** blocked
- **Results:** —

### P3 — Full Fine-Tune + Head
- **Hypothesis:** End-to-end fine-tuning yields best raw accuracy but risks forgetting contrastive features
- **Config:** `head=linear, backbone=full, lr=5e-5, epochs=3`
- **Status:** blocked
- **Results:** —

### P4 — LoRA + Head
- **Hypothesis:** LoRA preserves contrastive features while allowing task-specific adaptation; best trade-off
- **Config:** `head=linear, backbone=lora r=16, lr=2e-4, epochs=5`
- **Status:** blocked
- **Results:** —

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-07 | Bin corneal to 4 classes | Rare conditions (<500 each) collapsed into "Rare"; insufficient data for fine-grained classification |
| 2026-03-07 | Drop "Other" lens status | Only 50 samples; too noisy and ambiguous to learn |
| 2026-03-07 | Clamp class weights [0.5, 10.0] | Prevents extreme gradients from aphakia (410 samples, weight ~250x without clamping) |
| 2026-03-07 | Set CUDA_VISIBLE_DEVICES before torch import | PyTorch's CUDA init grabs all visible GPUs; OOM when 2 jobs share GPU memory |
| 2026-03-07 | Enable LOAD_TRUNCATED_IMAGES | 15,131 corrupt JPEGs (13%); filtering is primary defense, but flag prevents crashes from edge cases |
| 2026-03-07 | Use per-device batch 16 for LoRA, 4 for partial freeze | A100-40GB VRAM constraints; grad_accum compensates to reach effective batch 64 |

---

## Timeline

| Date | Event |
|------|-------|
| 2026-03-07 | Project initialized. Environment setup. Data analysis complete |
| 2026-03-07 | Discovered 15,131 corrupt images. Created bad_images.json, regenerated splits |
| 2026-03-07 | Fixed OOM: CUDA_VISIBLE_DEVICES must be set before torch import |
| 2026-03-07 | Launched C2 (GPU1) + C3 (GPU0) — 3rd attempt after OOM + corrupt image fixes |
| | C2+C3 complete. Phase 1 gate decision (fill in) |
| | C1 launched (fill in) |
| | Phase 1 winner selected. C4+C5 ablations started |
| | Phase 2 experiments started |
