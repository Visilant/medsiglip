#!/usr/bin/env bash
# Phase 1: Contrastive fine-tuning experiments
# Run from the experiments/ directory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EXP_DIR")"
cd "$PROJECT_DIR"

export PYTHONPATH="$EXP_DIR:$PYTHONPATH"
PYTHON="$EXP_DIR/.venv/bin/python"
CONFIG="$EXP_DIR/config/base.yaml"

# --- C1: Full fine-tune, both encoders ---
run_c1() {
    echo "=== C1: Full fine-tune ==="
    $PYTHON -m training.train_contrastive \
        --experiment_id C1 \
        --config $CONFIG \
        --mode full \
        --learning_rate 1e-5 \
        --num_epochs 2 \
        --per_device_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --warmup_steps 500 \
        --caption_style clinical \
        --gpu "0,1"
}

# --- C2: LoRA r=16, both encoders ---
run_c2() {
    echo "=== C2: LoRA fine-tune ==="
    $PYTHON -m training.train_contrastive \
        --experiment_id C2 \
        --config $CONFIG \
        --mode lora \
        --learning_rate 2e-4 \
        --num_epochs 3 \
        --per_device_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --warmup_steps 300 \
        --lora_r 16 \
        --lora_alpha 32 \
        --caption_style clinical \
        --gpu "0,1"
}

# --- C3: Partial freeze (last 6 vision layers + full text) ---
run_c3() {
    echo "=== C3: Partial freeze ==="
    $PYTHON -m training.train_contrastive \
        --experiment_id C3 \
        --config $CONFIG \
        --mode partial_freeze \
        --learning_rate 5e-5 \
        --num_epochs 3 \
        --per_device_batch_size 12 \
        --gradient_accumulation_steps 5 \
        --warmup_steps 400 \
        --caption_style clinical \
        --gpu "0,1"
}

# --- C4: Caption ablation (best arch, 3 variants, 1 epoch each) ---
run_c4() {
    BEST_MODE="${1:-lora}"  # Pass best mode from C1-C3 results
    BEST_LR="${2:-2e-4}"
    echo "=== C4: Caption ablation ==="
    for style in clinical label_only sentence; do
        echo "  Caption style: $style"
        $PYTHON -m training.train_contrastive \
            --experiment_id "C4_${style}" \
            --config $CONFIG \
            --mode "$BEST_MODE" \
            --learning_rate "$BEST_LR" \
            --num_epochs 1 \
            --per_device_batch_size 16 \
            --gradient_accumulation_steps 4 \
            --warmup_steps 300 \
            --caption_style "$style" \
            --gpu "0"
    done
}

# --- C5: Data size ablation (best arch, 4 sizes, 1 epoch each) ---
run_c5() {
    BEST_MODE="${1:-lora}"
    BEST_LR="${2:-2e-4}"
    echo "=== C5: Data size ablation ==="
    for size in 10000 30000 50000; do
        echo "  Data size: $size"
        $PYTHON -m training.train_contrastive \
            --experiment_id "C5_${size}" \
            --config $CONFIG \
            --mode "$BEST_MODE" \
            --learning_rate "$BEST_LR" \
            --num_epochs 1 \
            --per_device_batch_size 16 \
            --gradient_accumulation_steps 4 \
            --warmup_steps 300 \
            --caption_style clinical \
            --max_samples "$size" \
            --gpu "1"
    done
    # Full dataset (no limit)
    $PYTHON -m training.train_contrastive \
        --experiment_id "C5_full" \
        --config $CONFIG \
        --mode "$BEST_MODE" \
        --learning_rate "$BEST_LR" \
        --num_epochs 1 \
        --per_device_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --warmup_steps 300 \
        --caption_style clinical \
        --gpu "1"
}

# --- Main: run selected experiment ---
case "${1:-all}" in
    C1) run_c1 ;;
    C2) run_c2 ;;
    C3) run_c3 ;;
    C4) run_c4 "${2:-lora}" "${3:-2e-4}" ;;
    C5) run_c5 "${2:-lora}" "${3:-2e-4}" ;;
    all)
        run_c1
        run_c2
        run_c3
        ;;
    *)
        echo "Usage: $0 {C1|C2|C3|C4|C5|all} [best_mode] [best_lr]"
        exit 1
        ;;
esac
