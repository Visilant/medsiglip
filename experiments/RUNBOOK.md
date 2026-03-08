# Runbook — MedSigLIP Experiments

Operational manual for launching, monitoring, and debugging experiments.

---

## Environment Setup

```bash
cd /home/adi/medsiglip
bash experiments/setup.sh          # installs uv, syncs deps, checks GPU + model + W&B
source experiments/.venv/bin/activate  # only if running interactively
```

**Critical:** Never install into the base env at `/home/adi/dashboard_visilant/backend/.venv`.

---

## Launching Experiments

### Golden Pattern (always follow this)

```bash
PYTHONPATH=experiments \
CUDA_VISIBLE_DEVICES=<gpu_id> \
experiments/.venv/bin/python -m training.<script> \
    --experiment_id <ID> \
    --config experiments/config/base.yaml \
    [flags...] \
    > experiments/outputs/<ID>.log 2>&1 &
```

**Why CUDA_VISIBLE_DEVICES as env var?** PyTorch's CUDA initialization grabs all visible GPUs on `import torch`. Setting it inside Python (after import) is too late and causes OOM when multiple jobs run.

### Phase 1 — Contrastive

```bash
# C1: Full fine-tune
PYTHONPATH=experiments CUDA_VISIBLE_DEVICES=1 experiments/.venv/bin/python -m training.train_contrastive \
    --experiment_id C1 --mode full --learning_rate 1e-5 --num_epochs 2 \
    --per_device_batch_size 8 --gradient_accumulation_steps 8 \
    --warmup_steps 500 --gpu "1" --config experiments/config/base.yaml \
    > experiments/outputs/C1.log 2>&1 &

# C2: LoRA
PYTHONPATH=experiments CUDA_VISIBLE_DEVICES=1 experiments/.venv/bin/python -m training.train_contrastive \
    --experiment_id C2 --mode lora --lora_r 16 --lora_alpha 32 \
    --learning_rate 2e-4 --num_epochs 3 \
    --per_device_batch_size 16 --gradient_accumulation_steps 4 \
    --warmup_steps 300 --gpu "1" --config experiments/config/base.yaml \
    > experiments/outputs/C2.log 2>&1 &

# C3: Partial freeze
PYTHONPATH=experiments CUDA_VISIBLE_DEVICES=0 experiments/.venv/bin/python -m training.train_contrastive \
    --experiment_id C3 --mode partial_freeze \
    --learning_rate 5e-5 --num_epochs 3 \
    --per_device_batch_size 4 --gradient_accumulation_steps 16 \
    --warmup_steps 400 --gpu "0" --config experiments/config/base.yaml \
    > experiments/outputs/C3.log 2>&1 &
```

### Phase 1 — Ablations (use best mode/LR from C1-C3)

```bash
# C4: Caption ablation (run 3 variants)
for style in clinical label_only sentence; do
    PYTHONPATH=experiments CUDA_VISIBLE_DEVICES=0 experiments/.venv/bin/python -m training.train_contrastive \
        --experiment_id C4_${style} --mode <best_mode> --learning_rate <best_lr> \
        --num_epochs 1 --caption_style $style \
        --per_device_batch_size <best_batch> --gradient_accumulation_steps <best_accum> \
        --gpu "0" --config experiments/config/base.yaml \
        > experiments/outputs/C4_${style}.log 2>&1
done

# C5: Data size ablation
for n in 10000 30000 50000; do
    PYTHONPATH=experiments CUDA_VISIBLE_DEVICES=1 experiments/.venv/bin/python -m training.train_contrastive \
        --experiment_id C5_${n} --mode <best_mode> --learning_rate <best_lr> \
        --num_epochs 1 --max_samples $n \
        --per_device_batch_size <best_batch> --gradient_accumulation_steps <best_accum> \
        --gpu "1" --config experiments/config/base.yaml \
        > experiments/outputs/C5_${n}.log 2>&1
done
```

### Phase 2 — Classification

```bash
# P1: Linear probe (both tasks)
for task in lens_status corneal; do
    PYTHONPATH=experiments CUDA_VISIBLE_DEVICES=0 experiments/.venv/bin/python -m training.train_classifier \
        --experiment_id P1_${task} --task $task --head_type linear --backbone_mode frozen \
        --learning_rate 1e-3 --num_epochs 10 --per_device_batch_size 64 \
        --checkpoint_path experiments/outputs/<best_C>/final_model \
        --gpu "0" --config experiments/config/base.yaml \
        > experiments/outputs/P1_${task}.log 2>&1
done
```

### Using Shell Scripts

```bash
bash experiments/scripts/run_phase1.sh C2          # run single experiment
bash experiments/scripts/run_phase1.sh all          # run all Phase 1
bash experiments/scripts/run_phase2.sh P1 lens_status 0  # specific task + GPU
```

---

## Monitoring

### Quick Status

```bash
# Check running jobs
ps aux | grep train_ | grep -v grep

# GPU utilization + memory
nvidia-smi
watch -n 5 nvidia-smi   # continuous

# Tail logs
tail -f experiments/outputs/C2.log
tail -f experiments/outputs/C3.log

# Check W&B (if browser available)
# https://wandb.ai/adi-visilant-visilant-inc/medsiglip-visilant
```

### Check Results After Completion

```bash
# Zero-shot metrics (Phase 1)
cat experiments/outputs/<ID>/zeroshot_lens_status.json | python -m json.tool
cat experiments/outputs/<ID>/zeroshot_corneal.json | python -m json.tool

# Classification metrics (Phase 2)
cat experiments/outputs/<ID>/final_metrics.json | python -m json.tool
cat experiments/outputs/<ID>/classification_report.txt

# Experiment config snapshot
cat experiments/outputs/<ID>/experiment_config.json | python -m json.tool
```

### GPU Memory Reference

| Mode | per_device_batch | VRAM Usage (A100-40GB) |
|------|-----------------|----------------------|
| full | 8 | ~38 GB |
| lora r=16 | 16 | ~25 GB |
| partial_freeze | 4 | ~24 GB |
| classifier (frozen) | 64 | ~15 GB |
| classifier (full) | 8 | ~38 GB |
| classifier (lora) | 16 | ~25 GB |

---

## Troubleshooting

### OOM (CUDA Out of Memory)
- **Symptom:** `torch.cuda.OutOfMemoryError`
- **Root cause:** CUDA_VISIBLE_DEVICES not set as shell env var before Python starts
- **Fix:** Always use `CUDA_VISIBLE_DEVICES=<id>` as prefix to the command, never `os.environ` after torch import
- **If still OOM:** Reduce `per_device_batch_size`, increase `gradient_accumulation_steps`

### Corrupt Image Errors
- **Symptom:** `PIL.UnidentifiedImageError` or `OSError: image file is truncated`
- **Root cause:** 15,131 corrupt JPEG files in dataset
- **Defense 1:** Filtered out via `bad_images.json` in `load_and_filter_data()`
- **Defense 2:** `PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True` in `dataset.py`
- **If new corruptions found:** Run `experiments/data/classify_bad.py` to scan, update `bad_images.json`

### W&B Issues
- **Login:** `experiments/.venv/bin/wandb login`
- **Offline mode:** `WANDB_MODE=offline` prefix to command
- **Disable:** `WANDB_DISABLED=true`

### Stale Processes
```bash
# Kill by experiment ID
ps aux | grep "experiment_id C2" | grep -v grep | awk '{print $2}' | xargs kill
# Or kill all training
pkill -f train_contrastive
pkill -f train_classifier
```

---

## Output Directory Structure

After an experiment completes, its output directory contains:

```
experiments/outputs/<ID>/
├── experiment_config.json        # Full config snapshot (reproducibility)
├── checkpoint-<step>/            # Intermediate checkpoints (if save_steps set)
│   ├── pytorch_model.bin
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── trainer_state.json
├── final_model/                  # Final model weights
│   ├── config.json
│   ├── model.safetensors
│   └── adapter_config.json      # (LoRA only)
├── zeroshot_lens_status.json     # Phase 1: zero-shot eval results
├── zeroshot_corneal.json         # Phase 1: zero-shot eval results
├── final_metrics.json            # Phase 2: classification metrics
├── classification_report.txt     # Phase 2: sklearn report
└── training_args.bin             # HuggingFace Trainer args
```
