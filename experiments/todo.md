# TODO — MedSigLIP Fine-Tuning

> See also: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | [RUNBOOK.md](RUNBOOK.md) | [RESULTS.md](RESULTS.md) | [HYPOTHESES.md](HYPOTHESES.md)

---

## Currently Running (3rd launch, 2026-03-07 ~05:30 UTC)

| Job | GPU | Log | W&B | Est. Duration |
|-----|-----|-----|-----|---------------|
| **C2** — LoRA r=16, lr=2e-4, 3 epochs, batch 64 eff | GPU 1 (25GB) | `outputs/C2.log` | [C2 run](https://wandb.ai/adi-visilant-visilant-inc/huggingface/runs/cgyif47k) | ~2.8h (4065 steps @ ~2.5s/step) |
| **C3** — Partial freeze, lr=5e-5, 3 epochs, batch 64 eff | GPU 0 (24GB) | `outputs/C3.log` | [C3 run](https://wandb.ai/adi-visilant-visilant-inc/huggingface/runs/e9oikrlf) | ~4.3h (4065 steps @ ~3.8s/step) |

### Monitor
```bash
ps aux | grep train_contrastive | grep -v grep
nvidia-smi
tail -f experiments/outputs/C2.log
tail -f experiments/outputs/C3.log
```

---

## Issues Fixed

1. **OOM (1st launch):** CUDA_VISIBLE_DEVICES set inside Python after torch import → Fixed: set as shell env var
2. **Corrupt images (2nd launch):** 15,131 corrupt .jpg files → Fixed: filtered in `load_and_filter_data()`, bad images being moved to `/home/adi/visilant_data_bad/`

---

## Checklist

### Pre-Experiment (Critical)
- [x] Environment setup (uv venv, all deps installed)
- [x] Data pipeline (loading, filtering, splits, captions)
- [x] Smoke tests (data, model forward pass, GPU)
- [x] Fixed CUDA_VISIBLE_DEVICES bug
- [x] Found and filtered 15,131 corrupt images
- [x] Regenerated splits: 86,672 train / 9,631 val
- [ ] **Run pretrained baseline evaluation** (needed for comparison)
- [ ] Verify bad images fully moved to `visilant_data_bad/`

### Phase 1: Contrastive
- [ ] C2 — LoRA contrastive (running)
- [ ] C3 — Partial freeze contrastive (running)
- [ ] C1 — Full fine-tune contrastive
- [ ] **Phase 1 gate decision** — pick best mode (see HYPOTHESES.md)
- [ ] C4 — Caption style ablation (3 variants)
- [ ] C5 — Data size ablation (4 sizes)
- [ ] Update RESULTS.md with Phase 1 findings

### Phase 2: Classification
- [ ] P1 — Linear probe (lens_status + corneal)
- [ ] P2 — MLP head (lens_status + corneal)
- [ ] P3 — Full FT classification
- [ ] P4 — LoRA classification
- [ ] Focal loss ablation
- [ ] Test set evaluation (628 held-out images)
- [ ] Update RESULTS.md with Phase 2 findings
- [ ] Clinical metrics: sensitivity @ 95% specificity

### Post-Experiment
- [ ] Final model selection and rationale
- [ ] Export best model for serving integration
- [ ] Update EXPERIMENT_LOG.md with final status

---

## Next Steps (after C2 + C3 finish)

### 1. Check results
```bash
cat experiments/outputs/C2/zeroshot_lens_status.json | python -m json.tool
cat experiments/outputs/C2/zeroshot_corneal.json | python -m json.tool
cat experiments/outputs/C3/zeroshot_lens_status.json | python -m json.tool
cat experiments/outputs/C3/zeroshot_corneal.json | python -m json.tool
```

### 2. Run C1 (full fine-tune) on GPU 1
```bash
PYTHONPATH=experiments CUDA_VISIBLE_DEVICES=1 experiments/.venv/bin/python -m training.train_contrastive \
    --experiment_id C1 --mode full --learning_rate 1e-5 --num_epochs 2 \
    --per_device_batch_size 8 --gradient_accumulation_steps 8 \
    --warmup_steps 500 --gpu "1" --config experiments/config/base.yaml \
    > experiments/outputs/C1.log 2>&1 &
```

### 3. Phase 1 gate → pick best mode → run C4 + C5

### 4. Phase 2 classification with best checkpoint

---

## Launch Pattern (always use this)
```bash
PYTHONPATH=experiments CUDA_VISIBLE_DEVICES=<gpu_id> experiments/.venv/bin/python -m training.<script> \
    --experiment_id <ID> --config experiments/config/base.yaml [flags...] \
    > experiments/outputs/<ID>.log 2>&1 &
```
