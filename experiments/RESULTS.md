# Results & Analysis

> **Last updated:** 2026-03-08
> **Status:** Awaiting C2 + C3 completion

---

## Pretrained Baseline (No Fine-Tuning)

Zero-shot performance of `google/medsiglip-448` without any fine-tuning.

| Task | Accuracy | Macro F1 | Macro AUROC |
|------|----------|----------|-------------|
| Lens Status (7-class) | — | — | — |
| Corneal (4-class) | — | — | — |

> TODO: Run baseline eval before comparing fine-tuned models

---

## Phase 1: Contrastive Fine-Tuning Results

### Summary Comparison

| ID | Mode | Trainable Params | Val Loss | ZS Lens F1 | ZS Corneal F1 | ZS Avg F1 | Train Time |
|----|------|-----------------|----------|------------|---------------|-----------|------------|
| baseline | none | 0 | — | — | — | — | — |
| C1 | full | ~800M | — | — | — | — | — |
| C2 | lora | ~6.3M | — | — | — | — | — |
| C3 | partial_freeze | ~120M | — | — | — | — | — |

### Per-Class Zero-Shot F1 (Val Set)

#### Lens Status

| Class | Baseline | C1 | C2 | C3 |
|-------|----------|----|----|-----|
| clear_crystalline_lens | — | — | — | — |
| immature_cataract | — | — | — | — |
| early_lens_changes | — | — | — | — |
| PCIOL | — | — | — | — |
| not_able_to_visualize_lens | — | — | — | — |
| mature_cataract | — | — | — | — |
| aphakia | — | — | — | — |

#### Corneal Abnormality

| Class | Baseline | C1 | C2 | C3 |
|-------|----------|----|----|-----|
| Normal | — | — | — | — |
| Active corneal infection | — | — | — | — |
| Inactive corneal opacity | — | — | — | — |
| Rare | — | — | — | — |

### Phase 1 Gate Decision
- **Winner:** TBD
- **Rationale:** TBD
- **Selected for C4/C5:** mode=TBD, lr=TBD

---

## C4: Caption Style Ablation

| Style | ZS Lens F1 | ZS Corneal F1 | ZS Avg F1 | Delta vs Clinical |
|-------|------------|---------------|-----------|-------------------|
| clinical | — | — | — | baseline |
| label_only | — | — | — | — |
| sentence | — | — | — | — |

---

## C5: Data Size Ablation

| Samples | ZS Lens F1 | ZS Corneal F1 | ZS Avg F1 | % of Full Performance |
|---------|------------|---------------|-----------|----------------------|
| 10,000 | — | — | — | — |
| 30,000 | — | — | — | — |
| 50,000 | — | — | — | — |
| 86,672 (full) | — | — | — | 100% |

---

## Phase 2: Classification Results

### Summary Comparison

| ID | Head | Backbone | Task | Test Acc | Test F1 | Test AUROC | Sens@95Spec (critical) |
|----|------|----------|------|----------|---------|------------|----------------------|
| P1_lens | linear | frozen | lens_status | — | — | — | — |
| P1_corneal | linear | frozen | corneal | — | — | — | — |
| P2_lens | mlp | frozen | lens_status | — | — | — | — |
| P2_corneal | mlp | frozen | corneal | — | — | — | — |
| P3_lens | linear | full | lens_status | — | — | — | — |
| P3_corneal | linear | full | corneal | — | — | — | — |
| P4_lens | linear | lora | lens_status | — | — | — | — |
| P4_corneal | linear | lora | corneal | — | — | — | — |

### Clinical Metrics (Sensitivity @ 95% Specificity)

#### Lens Status — Critical Classes

| Class | P1 | P2 | P3 | P4 |
|-------|----|----|----|----|
| mature_cataract | — | — | — | — |
| aphakia | — | — | — | — |

#### Corneal — Critical Classes

| Class | P1 | P2 | P3 | P4 |
|-------|----|----|----|----|
| Active corneal infection | — | — | — | — |

---

## Confusion Matrices

> Paste or link confusion matrix images/tables here after experiments complete.

---

## Key Findings

> To be filled as results come in:
>
> 1. **Fine-tuning strategy:** [which mode won and why]
> 2. **Caption impact:** [did rich captions help?]
> 3. **Data efficiency:** [where did performance plateau?]
> 4. **Classification head:** [frozen vs fine-tuned backbone, linear vs MLP]
> 5. **Clinical viability:** [sensitivity/specificity for actionable classes]

---

## Failure Analysis

> Document any unexpected results, debugging insights, or negative results here.
> Negative results are valuable — they prevent repeating failed approaches.

| Experiment | Expected | Actual | Root Cause | Action |
|-----------|----------|--------|-----------|--------|
| — | — | — | — | — |
