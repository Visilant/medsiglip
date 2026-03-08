"""Phase 1: Contrastive fine-tuning entry point.

Usage:
    .venv/bin/python -m training.train_contrastive \
        --experiment_id C1 \
        --mode full \
        --learning_rate 1e-5 \
        --num_epochs 2 \
        --per_device_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --warmup_steps 500 \
        --caption_style clinical
"""

import argparse
import json
import logging
import os
import sys

# CUDA_VISIBLE_DEVICES must be set before torch import.
# Parse --gpu early from sys.argv to set it in time.
_gpu_arg = None
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu_arg = sys.argv[i + 1]
        break
if _gpu_arg is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg

import numpy as np
import torch
import yaml
from transformers import AutoProcessor, Trainer, TrainingArguments

from data.caption_builder import CaptionProcessor
from data.collator import contrastive_collate_fn
from data.dataset import VisilantContrastiveDataset
from data.splits import create_stratified_split, load_and_filter_data, load_splits, save_runtime_bad_images
from models.contrastive import load_contrastive_model
from training.callbacks import NaNLossCallback
from training.evaluate import retrieval_evaluate, zero_shot_evaluate

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Contrastive fine-tuning")
    parser.add_argument("--experiment_id", type=str, required=True, help="Experiment ID (e.g., C1, C2)")
    parser.add_argument("--config", type=str, default="experiments/config/base.yaml")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "lora", "partial_freeze"])
    parser.add_argument("--caption_style", type=str, default="clinical", choices=["clinical", "label_only", "sentence"])
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples (for ablation)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--gpu", type=str, default=None, help="GPU device(s), e.g. '0' or '0,1'")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # GPU already set via early sys.argv parsing above

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join(config["paths"]["output_base"], args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment config
    exp_config = {**vars(args), "config_values": config}
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    print(f"=== Experiment {args.experiment_id}: {args.mode} contrastive, caption={args.caption_style} ===")

    # Load and filter data
    df = load_and_filter_data(
        csv_path=config["data"]["csv_path"],
        image_dir=config["data"]["image_dir"],
        image_extension=config["data"]["image_extension"],
        drop_lens_status=config["labels"]["lens_status_drop"],
        corneal_map=config["labels"]["corneal_abnormality_map"],
    )

    # Create or load splits
    splits_path = config["data"]["splits_path"]
    if os.path.exists(splits_path):
        print(f"Loading existing splits from {splits_path}")
        splits = load_splits(splits_path)
    else:
        print("Creating new stratified splits...")
        splits = create_stratified_split(
            df=df,
            test_csv_path=config["data"]["test_csv_path"],
            val_fraction=config["data"]["val_fraction"],
            seed=config["training"]["seed"],
            save_path=splits_path,
        )

    train_df = df.loc[df.index.isin(splits["train_indices"])].copy()
    val_df = df.loc[df.index.isin(splits["val_indices"])].copy()

    # Limit samples for ablation
    if args.max_samples and args.max_samples < len(train_df):
        train_df = train_df.sample(n=args.max_samples, random_state=config["training"]["seed"])
        print(f"Limited training to {len(train_df)} samples")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Caption processor
    caption_proc = CaptionProcessor(
        model_id=config["model"]["model_id"],
        max_tokens=config["model"]["max_text_tokens"],
        caption_style=args.caption_style,
    )

    # Datasets
    train_dataset = VisilantContrastiveDataset(train_df, config["data"]["image_dir"], caption_proc)
    val_dataset = VisilantContrastiveDataset(val_df, config["data"]["image_dir"], caption_proc)

    # Model
    model = load_contrastive_model(
        model_id=config["model"]["model_id"],
        mode=args.mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        fp16=config["training"]["fp16"],
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        report_to="wandb",
        run_name=f"contrastive-{args.experiment_id}",
        seed=config["training"]["seed"],
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=contrastive_collate_fn,
        callbacks=[NaNLossCallback()],
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Persist runtime-discovered bad images
    n_bad = save_runtime_bad_images()
    if n_bad:
        logger.info("Persisted %d runtime-discovered bad images to bad_images.json", n_bad)

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final"))
    print(f"Model saved to {output_dir}/final")

    # Run zero-shot evaluation
    print("\nRunning zero-shot evaluation on val set...")
    processor = AutoProcessor.from_pretrained(config["model"]["model_id"])

    for task, classes in [
        ("lens_status", config["labels"]["lens_status_classes"]),
        ("corneal", config["labels"]["corneal_classes"]),
    ]:
        label_col = "mapped_lens_status" if task == "lens_status" else "corneal_binned"
        prompts = [f"slit lamp showing {c.replace('_', ' ')}" for c in classes]

        from data.dataset import VisilantClassificationDataset
        from data.collator import classification_collate_fn
        eval_ds = VisilantClassificationDataset(
            val_df, config["data"]["image_dir"], label_col, classes,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_ds, batch_size=64, shuffle=False,
            collate_fn=classification_collate_fn,
            num_workers=config["training"]["dataloader_num_workers"],
        )

        zs_metrics = zero_shot_evaluate(
            model=model if not hasattr(model, "base_model") else model.base_model,
            processor=processor,
            dataloader=eval_loader,
            class_prompts=prompts,
            class_names=classes,
            label_column=label_col,
        )
        print(f"\n{task} zero-shot results:")
        print(f"  Accuracy: {zs_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {zs_metrics['macro_f1']:.4f}")
        if not np.isnan(zs_metrics.get("macro_auroc", float("nan"))):
            print(f"  Macro AUROC: {zs_metrics['macro_auroc']:.4f}")

        with open(os.path.join(output_dir, f"zeroshot_{task}.json"), "w") as f:
            # Convert non-serializable values
            serializable = {k: v for k, v in zs_metrics.items() if k != "classification_report"}
            serializable["classification_report"] = zs_metrics.get("classification_report", "")
            json.dump(serializable, f, indent=2, default=str)

    print(f"\n=== Experiment {args.experiment_id} complete ===")


if __name__ == "__main__":
    main()
