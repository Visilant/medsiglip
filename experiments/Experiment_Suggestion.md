# MIG-Based Experiment Plan

**Hardware:** 2× NVIDIA A100-SXM4-40GB, MIG-capable
**Dataset:** 86,672 train / 9,631 val / 628 test images
**Model:** MedSigLIP-448 (~400M vision + ~400M text encoder)

---

## VRAM Estimates (fp16 training)

| Experiment Type | Trainable Params | Est. VRAM | Min MIG Profile |
|---|---|---|---|
| Frozen backbone classifier (bs=32) | ~260K (linear) / ~1M (MLP) | ~2 GB | `1g.5gb` (4.75 GB) |
| LoRA classifier (bs=8) | ~3M | ~3.5 GB | `1g.5gb` (4.75 GB) |
| Full FT classifier (bs=4, grad_ckpt) | ~400M | ~8 GB | `2g.10gb` (9.75 GB) |
| LoRA contrastive (bs=8) | ~6M | ~7 GB | `2g.10gb` (9.75 GB) |
| Partial freeze contrastive (bs=8) | ~450M | ~9 GB | `2g.10gb` (9.75 GB) |
| Full contrastive (bs=8) | ~800M | ~16 GB | `3g.20gb` (19.62 GB) |

---

## Wave 1: Phase 1 + Phase 2 Baselines (concurrent, different GPUs)

Run contrastive pre-training on GPU 0 while Phase 2 baselines churn through GPU 1.

### GPU 0 — Phase 1 Contrastive (3 experiments)

**MIG: `1× 3g.20gb` + `2× 2g.10gb`** = 7 slices

| Slice | Experiment | Est. VRAM | Batch | Epochs | SMs |
|---|---|---|---|---|---|
| 3g.20gb | C1: full contrastive | ~16 GB | 8 (×8 accum) | 2 | 42 |
| 2g.10gb | C2: LoRA contrastive | ~7 GB | 8 (×4 accum) | 3 | 28 |
| 2g.10gb | C3: partial freeze | ~9 GB | 8 (×4 accum) | 3 | 28 |

All 3 contrastive modes run simultaneously. C1 gets the largest slice (most memory-hungry). Estimated ~5-8 hours.

### GPU 1 — Phase 2 Baselines, Round 1 (4 experiments)

**MIG: `4× 1g.10gb`** = 4 slices (3 spare)

| Slice | Experiment | Est. VRAM | Batch | Epochs |
|---|---|---|---|---|
| 1g.10gb | P1_lens: frozen + linear | ~2 GB | 32 | 10 |
| 1g.10gb | P1_corneal: frozen + linear | ~2 GB | 32 | 10 |
| 1g.10gb | P2_lens: frozen + MLP | ~2 GB | 32 | 15 |
| 1g.10gb | P2_corneal: frozen + MLP | ~2 GB | 32 | 15 |

These are lightweight (only head params update, no backbone gradients). Estimated ~45-90 min each on 14 SMs. They use base MedSigLIP weights — no contrastive checkpoint.

### GPU 1 — Resize after Round 1 completes (~1-2h in)

**Reconfigure to: `2× 2g.10gb` + `3× 1g.5gb`** = 7 slices

| Slice | Experiment | Est. VRAM | Batch | Epochs |
|---|---|---|---|---|
| 2g.10gb | P3_lens: full FT + linear | ~8 GB | 4 (×8 accum) | 3 |
| 2g.10gb | P3_corneal: full FT + linear | ~8 GB | 4 (×8 accum) | 3 |
| 1g.5gb | P4_lens: LoRA + linear | ~3.5 GB | 8 (×4 accum) | 5 |
| 1g.5gb | P4_corneal: LoRA + linear | ~3.5 GB | 8 (×4 accum) | 5 |
| 1g.5gb | P_focal_lens: frozen + linear + focal | ~2 GB | 32 | 10 |

5 more experiments. P3 needs `gradient_checkpointing=True` and reduced batch to fit in 10 GB. Estimated ~2-4 hours.

**Wave 1 total: 12 experiments, ~6-8 hours wall clock.**

---

## Wave 2: Phase 2 with Best Contrastive Checkpoint (9 experiments)

After Wave 1, compare C1/C2/C3 zero-shot metrics → pick best checkpoint.

### GPU 0 — Resize to `4× 1g.10gb`

| Slice | Experiment | Batch | Epochs |
|---|---|---|---|
| 1g.10gb | P1_lens_ckpt: frozen + linear | 32 | 10 |
| 1g.10gb | P1_corneal_ckpt: frozen + linear | 32 | 10 |
| 1g.10gb | P2_lens_ckpt: frozen + MLP | 32 | 15 |
| 1g.10gb | P2_corneal_ckpt: frozen + MLP | 32 | 15 |

### GPU 1 — Resize to `2× 2g.10gb` + `3× 1g.5gb`

| Slice | Experiment | Batch | Epochs |
|---|---|---|---|
| 2g.10gb | P3_lens_ckpt: full FT + linear | 4 | 3 |
| 2g.10gb | P3_corneal_ckpt: full FT + linear | 4 | 3 |
| 1g.5gb | P4_lens_ckpt: LoRA + linear | 8 | 5 |
| 1g.5gb | P4_corneal_ckpt: LoRA + linear | 8 | 5 |
| 1g.5gb | P_focal_corneal: frozen + linear + focal | 32 | 10 |

**Wave 2 total: 9 experiments, ~3-4 hours. Direct comparison with Wave 1 baselines shows contrastive pre-training value.**

---

## Wave 3: Ablations (8 experiments)

After best contrastive mode is known from Wave 1.

### Both GPUs — `4× 1g.10gb` each

These are all LoRA/partial-freeze with 1 epoch — memory-light.

| GPU | Slice | Experiment | Data |
|---|---|---|---|
| 0 | 1g.10gb | C4_clinical | Full |
| 0 | 1g.10gb | C4_label_only | Full |
| 0 | 1g.10gb | C4_sentence | Full |
| 0 | 1g.10gb | C5_10k | 10K samples |
| 1 | 1g.10gb | C5_30k | 30K samples |
| 1 | 1g.10gb | C5_50k | 50K samples |
| 1 | 1g.10gb | C5_full | Full |
| 1 | 1g.10gb | WeightedSampler ablation | Full |

**Wave 3 total: 8 experiments, ~2-3 hours (1 epoch each).**

---

## Summary

| Wave | Experiments | MIG Config | Wall Clock |
|---|---|---|---|
| 1 | 12 (C1-C3 + P1-P4 baselines + focal) | GPU0: 3g+2g+2g / GPU1: dynamic | ~6-8h |
| 2 | 9 (P1-P4 with checkpoint + focal) | GPU0: 4×10gb / GPU1: 2g+2g+5gb×3 | ~3-4h |
| 3 | 8 (C4+C5+sampler ablations) | Both: 4×10gb | ~2-3h |
| **Total** | **29 experiments** | | **~12-15h** |

Without MIG (sequential, 2 GPUs): these 29 experiments would take ~40-60 hours.

---

## MIG Setup Commands

```bash
# Enable MIG mode (requires no running GPU processes, needs sudo)
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi -i 1 -mig 1

# --- Wave 1: GPU 0 (Phase 1 contrastive) ---
sudo nvidia-smi mig -cgi 3g.20gb,2g.10gb,2g.10gb -i 0
sudo nvidia-smi mig -cci -i 0

# --- Wave 1: GPU 1 Round 1 (Phase 2 frozen baselines) ---
sudo nvidia-smi mig -cgi 1g.10gb,1g.10gb,1g.10gb,1g.10gb -i 1
sudo nvidia-smi mig -cci -i 1

# List MIG instances to get UUIDs
nvidia-smi -L

# Each experiment targets a specific MIG UUID:
# CUDA_VISIBLE_DEVICES=MIG-<uuid> python -m training.train_classifier ...

# --- Wave 1: GPU 1 Resize for Round 2 (after frozen baselines complete) ---
# Destroy existing instances first
sudo nvidia-smi mig -dci -i 1
sudo nvidia-smi mig -dgi -i 1
# Create new layout
sudo nvidia-smi mig -cgi 2g.10gb,2g.10gb,1g.5gb,1g.5gb,1g.5gb -i 1
sudo nvidia-smi mig -cci -i 1

# --- Wave 2: Resize both GPUs ---
# GPU 0
sudo nvidia-smi mig -dci -i 0
sudo nvidia-smi mig -dgi -i 0
sudo nvidia-smi mig -cgi 1g.10gb,1g.10gb,1g.10gb,1g.10gb -i 0
sudo nvidia-smi mig -cci -i 0
# GPU 1
sudo nvidia-smi mig -dci -i 1
sudo nvidia-smi mig -dgi -i 1
sudo nvidia-smi mig -cgi 2g.10gb,2g.10gb,1g.5gb,1g.5gb,1g.5gb -i 1
sudo nvidia-smi mig -cci -i 1

# --- Wave 3: Both GPUs 4×10gb ---
# (repeat destroy + create pattern)
sudo nvidia-smi mig -dci -i 0 && sudo nvidia-smi mig -dgi -i 0
sudo nvidia-smi mig -cgi 1g.10gb,1g.10gb,1g.10gb,1g.10gb -i 0
sudo nvidia-smi mig -cci -i 0
sudo nvidia-smi mig -dci -i 1 && sudo nvidia-smi mig -dgi -i 1
sudo nvidia-smi mig -cgi 1g.10gb,1g.10gb,1g.10gb,1g.10gb -i 1
sudo nvidia-smi mig -cci -i 1

# --- Disable MIG when done ---
sudo nvidia-smi mig -dci -i 0 && sudo nvidia-smi mig -dgi -i 0
sudo nvidia-smi mig -dci -i 1 && sudo nvidia-smi mig -dgi -i 1
sudo nvidia-smi -i 0 -mig 0
sudo nvidia-smi -i 1 -mig 0
```

---

## Code Changes Required Before Starting

1. **Add `--gradient_checkpointing` flag** — critical for P3 full-FT to fit in 10 GB MIG slices
2. **Wire up `--use_weighted_sampler`** — currently a dead flag, needed for the sampler ablation
3. **Add `--save_total_limit`** — prevent disk blowup from 29 experiments saving checkpoints
4. **Make `dataloader_num_workers` a CLI arg** — reduce from 4 to 1-2 under MIG (8 experiments × 4 workers = 32 processes will oversubscribe CPU)
5. **Create a MIG launcher script** — discovers MIG UUIDs, assigns experiments to slices, launches with correct `CUDA_VISIBLE_DEVICES=MIG-<uuid>`
6. **Vectorize `get_sample_weights()`** — `iterrows()` on 86K rows is very slow
7. **Fix `FocalLoss` re-instantiation** — created every batch in `compute_loss()`, should be in `__init__`

---

## Code Review Notes

### Issues Found

- **`--use_weighted_sampler` is a dead flag** (`train_classifier.py:89`): parsed but never used, `WeightedRandomSampler` imported but never instantiated
- **No gradient checkpointing** in either training script: critical for fitting full-FT experiments on smaller MIG slices
- **`FocalLoss` re-instantiated every batch** (`train_classifier.py:127`): should be created once in `ClassificationTrainer.__init__`
- **No `save_total_limit`**: 29 experiments with epoch checkpoints will consume significant disk
- **`dataloader_num_workers=4` hardcoded**: 8 experiments × 4 workers = 32 processes will oversubscribe CPU under MIG
- **`get_sample_weights()` uses `iterrows()`** (`dataset.py:198`): very slow on 86K rows, should be vectorized with `pandas.map()`

### Things That Look Good

- Early `CUDA_VISIBLE_DEVICES` binding before torch import — correct and necessary
- `BadImageTracker` with thread-safety — solid for parallel data loading
- Fallback image loading (tries 11 neighbors) — robust against sporadic failures
- Image transform matches MedSigLIP expectations (`Resize(448) → ToTensor → Normalize(0.5)`)
- Collator None-filtering — handles partial batches from failed image loads
- `ClassificationTrainer` NaN guard (zero loss to skip batch) — correct gradient behavior
- `remove_unused_columns=False` — necessary since HF Trainer can't introspect custom datasets
- Stratified splits on composite key (`lens_status|corneal_binned`) — both tasks get balanced representation
- Class weight clamping `[0.5, 10.0]` — prevents extreme gradients from rare classes
