#!/usr/bin/env bash
# Phase 2: Classification experiments
# Run from the experiments/ directory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EXP_DIR")"
cd "$PROJECT_DIR"

export PYTHONPATH="$EXP_DIR:$PYTHONPATH"
PYTHON="$EXP_DIR/.venv/bin/python"
CONFIG="$EXP_DIR/config/base.yaml"

# Best Phase 1 checkpoint (update after Phase 1 completes)
CHECKPOINT="${CHECKPOINT:-}"

# --- P1: Linear probe (frozen backbone) ---
run_p1() {
    echo "=== P1: Linear probe ==="
    for task in lens_status corneal; do
        echo "  Task: $task"
        $PYTHON -m training.train_classifier \
            --experiment_id P1 \
            --config $CONFIG \
            --task "$task" \
            --head_type linear \
            --backbone_mode frozen \
            --checkpoint_path "$CHECKPOINT" \
            --learning_rate 1e-3 \
            --num_epochs 10 \
            --per_device_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --warmup_steps 100 \
            --gpu "0,1"
    done
}

# --- P2: MLP head (frozen backbone) ---
run_p2() {
    echo "=== P2: MLP head ==="
    for task in lens_status corneal; do
        echo "  Task: $task"
        $PYTHON -m training.train_classifier \
            --experiment_id P2 \
            --config $CONFIG \
            --task "$task" \
            --head_type mlp \
            --backbone_mode frozen \
            --checkpoint_path "$CHECKPOINT" \
            --learning_rate 1e-4 \
            --num_epochs 15 \
            --per_device_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --warmup_steps 100 \
            --gpu "0,1"
    done
}

# --- P3: Full fine-tune vision + linear head ---
run_p3() {
    TASK="${1:-lens_status}"
    GPU="${2:-0,1}"
    echo "=== P3: Full fine-tune, task=$TASK ==="
    $PYTHON -m training.train_classifier \
        --experiment_id P3 \
        --config $CONFIG \
        --task "$TASK" \
        --head_type linear \
        --backbone_mode full \
        --checkpoint_path "$CHECKPOINT" \
        --learning_rate 5e-5 \
        --num_epochs 3 \
        --per_device_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --warmup_steps 200 \
        --gpu "$GPU"
}

# --- P4: LoRA vision + linear head ---
run_p4() {
    TASK="${1:-lens_status}"
    GPU="${2:-0,1}"
    echo "=== P4: LoRA fine-tune, task=$TASK ==="
    $PYTHON -m training.train_classifier \
        --experiment_id P4 \
        --config $CONFIG \
        --task "$TASK" \
        --head_type linear \
        --backbone_mode lora \
        --checkpoint_path "$CHECKPOINT" \
        --learning_rate 2e-4 \
        --num_epochs 5 \
        --per_device_batch_size 16 \
        --gradient_accumulation_steps 2 \
        --warmup_steps 150 \
        --gpu "$GPU"
}

# --- Ablation: Focal loss vs CE ---
run_focal_ablation() {
    TASK="${1:-lens_status}"
    echo "=== Focal loss ablation, task=$TASK ==="
    $PYTHON -m training.train_classifier \
        --experiment_id P_focal \
        --config $CONFIG \
        --task "$TASK" \
        --head_type linear \
        --backbone_mode frozen \
        --checkpoint_path "$CHECKPOINT" \
        --learning_rate 1e-3 \
        --num_epochs 10 \
        --per_device_batch_size 64 \
        --use_focal_loss \
        --focal_gamma 2.0 \
        --gpu "1"
}

# --- Main ---
case "${1:-all}" in
    P1) run_p1 ;;
    P2) run_p2 ;;
    P3) run_p3 "${2:-lens_status}" "${3:-0,1}" ;;
    P4) run_p4 "${2:-lens_status}" "${3:-0,1}" ;;
    focal) run_focal_ablation "${2:-lens_status}" ;;
    all)
        run_p1
        run_p2
        ;;
    *)
        echo "Usage: $0 {P1|P2|P3|P4|focal|all} [task] [gpu]"
        exit 1
        ;;
esac
