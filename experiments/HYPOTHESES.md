# Hypotheses & Experimental Design

---

## Research Questions

1. **RQ1:** What fine-tuning strategy (full / LoRA / partial freeze) maximizes contrastive alignment for ophthalmology images with limited data (~87K images)?
2. **RQ2:** Do rich clinical captions improve zero-shot classification over label-only text?
3. **RQ3:** How data-efficient is MedSigLIP? At what sample size does performance plateau?
4. **RQ4:** Does a classification head outperform zero-shot, and which head architecture (linear / MLP) works best?
5. **RQ5:** Is end-to-end fine-tuning worth it for classification, or does frozen backbone + head suffice?

---

## Hypotheses by Experiment

### Phase 1: Contrastive Fine-Tuning

| ID | Hypothesis | Null | Success Criterion |
|----|-----------|------|-------------------|
| **C1** | Full fine-tuning gives the strongest embeddings but may overfit, especially on rare classes | No improvement over pretrained model | ZS macro-F1 > pretrained baseline on both tasks |
| **C2** | LoRA achieves 90%+ of full fine-tune performance with ~1% of trainable params, and better stability | LoRA < 80% of C1 performance | ZS macro-F1 within 5% of C1; lower val loss variance |
| **C3** | Partial freeze of last 6 vision layers outperforms LoRA by directly updating attention patterns while keeping text encoder stable | Partial freeze ≈ LoRA | ZS macro-F1 > C2 by >2% |
| **C4** | Clinical captions (with demographics, VA, illumination) improve contrastive alignment by >3% over label-only captions | Caption style has <1% effect | clinical > label_only on ZS macro-F1 |
| **C5** | Performance curves show >90% of full-data performance at 30K samples (data efficiency) | Linear scaling to 87K | Diminishing returns visible on log-scale plot |

### Phase 2: Classification

| ID | Hypothesis | Null | Success Criterion |
|----|-----------|------|-------------------|
| **P1** | Linear probe on frozen fine-tuned backbone achieves >75% macro-F1 on both tasks | Linear probe < 60% | macro-F1 > 75% lens, > 70% corneal |
| **P2** | MLP head improves over linear by >3% on rare classes (mature_cataract, aphakia, Rare corneal) | MLP ≈ linear overall | Per-class F1 improvement on classes with <5K samples |
| **P3** | End-to-end fine-tuning gives marginal improvement (<2%) over frozen backbone + MLP | E2E >> frozen by >5% | Macro-F1 gap between P3 and P2 < 2% |
| **P4** | LoRA backbone + head matches full fine-tune (P3) with 10x less GPU memory | LoRA << full by >3% | Macro-F1 within 2% of P3 |

---

## Experimental Controls

| Variable | Controlled By |
|----------|--------------|
| Random seed | Fixed at 42 across all experiments |
| Data splits | Same `splits.json` for all experiments |
| Image preprocessing | Same transform: Resize(448) → ToTensor → Normalize(0.5) |
| Tokenizer | Same SiglipTokenizer, max 64 tokens |
| Evaluation | Same zero-shot protocol (val set) for Phase 1; same test set for Phase 2 |
| Hardware | Same A100-40GB GPUs, fp16 mixed precision |

---

## Decision Framework

### Phase 1 Gate (after C1, C2, C3)

```
IF best ZS macro-F1 (avg of lens + corneal) > pretrained baseline:
    winner = argmax(ZS_macro_F1_avg) among {C1, C2, C3}
    IF C2 within 2% of C1 AND C2 val_loss more stable:
        winner = C2  (prefer parameter efficiency)
    Proceed to C4/C5 ablations with winner's mode + LR
ELSE:
    Investigate: check per-class breakdown, loss curves, increase epochs
```

### Phase 2 Gate (after P1-P4)

```
primary_metric = macro_F1 on test set
clinical_metric = sensitivity @ 95% specificity for:
    - lens: mature_cataract, aphakia
    - corneal: Active corneal infection

Select model that maximizes primary_metric
Break ties with clinical_metric
Report confidence intervals via bootstrap (n=1000)
```

---

## Ablation Design

### C4: Caption Style (controlled experiment)
- **Fixed:** Mode, LR, epochs, data, seed — all from best C1-C3
- **Varied:** `caption_style ∈ {clinical, label_only, sentence}`
- **Comparison:** ZS macro-F1 on val set per task
- **Expected outcome:** clinical > sentence > label_only (richer context = better alignment)

### C5: Data Size (scaling curve)
- **Fixed:** Mode, LR, caption_style=clinical, seed
- **Varied:** `max_samples ∈ {10000, 30000, 50000, 86672}`
- **Analysis:** Plot macro-F1 vs. log(samples). Fit power law if possible.
- **Expected outcome:** Diminishing returns after ~30K; MedSigLIP is data-efficient

### Focal Loss Ablation (Phase 2)
- **Fixed:** Best P1/P2 config
- **Varied:** Loss function: CrossEntropy vs. Focal(gamma=2.0)
- **Expected outcome:** Focal loss improves rare-class recall without hurting majority-class precision
