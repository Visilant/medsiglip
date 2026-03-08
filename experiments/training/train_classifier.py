"""Phase 2: Classification fine-tuning entry point.

Usage:
    .venv/bin/python -m training.train_classifier \
        --experiment_id P1 \
        --task lens_status \
        --head_type linear \
        --backbone_mode frozen \
        --learning_rate 1e-3 \
        --num_epochs 10
"""

import argparse
import json
import logging
import os
import sys

# CUDA_VISIBLE_DEVICES must be set before torch import.
_gpu_arg = None
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu_arg = sys.argv[i + 1]
        break
if _gpu_arg is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg

import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Trainer, TrainingArguments

from data.collator import classification_collate_fn
from data.dataset import VisilantClassificationDataset
from data.splits import load_and_filter_data, load_splits, save_runtime_bad_images
from models.classifier import VisionClassifier
from training.callbacks import NaNLossCallback
from utils.metrics import compute_classification_metrics, compute_clinical_metrics

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Classification")
    parser.add_argument("--experiment_id", type=str, required=True, help="Experiment ID (e.g., P1)")
    parser.add_argument("--config", type=str, default="experiments/config/base.yaml")
    parser.add_argument("--task", type=str, required=True, choices=["lens_status", "corneal"])
    parser.add_argument("--head_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--backbone_mode", type=str, default="frozen", choices=["frozen", "full", "lora"])
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Phase 1 checkpoint for backbone")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--per_device_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--gpu", type=str, default=None)
    return parser.parse_args()


class FocalLoss(torch.nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class ClassificationTrainer(Trainer):
    """Custom Trainer with class-weighted or focal loss."""

    def __init__(self, class_weights=None, use_focal=False, focal_gamma=2.0,
                 use_weighted_sampler=False, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.use_focal = use_focal
        self.use_weighted_sampler = use_weighted_sampler

        if use_focal:
            self.focal_loss_fn = FocalLoss(gamma=focal_gamma, weight=class_weights)

    def get_train_dataloader(self):
        if not self.use_weighted_sampler:
            return super().get_train_dataloader()

        sample_weights = self.train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]

        if self.use_focal:
            loss_fn = self.focal_loss_fn
            if loss_fn.weight is not None and loss_fn.weight.device != logits.device:
                loss_fn.weight = loss_fn.weight.to(logits.device)
            loss = loss_fn(logits, labels)
        elif self.class_weights is not None:
            loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=self.class_weights.to(logits.device)
            )
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        # NaN guard: log diagnostics and zero out to skip this batch's gradient
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                "NaN/Inf loss detected — labels: %s, logit range: [%.4f, %.4f]",
                labels.cpu().tolist(),
                logits.min().item(),
                logits.max().item(),
            )
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        return (loss, outputs) if return_outputs else loss


def make_compute_metrics(class_names):
    """Create a compute_metrics function for the Trainer."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        metrics = compute_classification_metrics(logits, labels, class_names)
        # Return only scalar metrics for Trainer
        return {k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return compute_metrics


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # GPU already set via early sys.argv parsing above

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine task-specific settings
    if args.task == "lens_status":
        class_names = config["labels"]["lens_status_classes"]
        label_column = "mapped_lens_status"
    else:
        class_names = config["labels"]["corneal_classes"]
        label_column = "corneal_binned"

    output_dir = os.path.join(config["paths"]["output_base"], f"{args.experiment_id}_{args.task}")
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment config
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"=== Experiment {args.experiment_id}: {args.task}, head={args.head_type}, backbone={args.backbone_mode} ===")

    # Load data
    df = load_and_filter_data(
        csv_path=config["data"]["csv_path"],
        image_dir=config["data"]["image_dir"],
        image_extension=config["data"]["image_extension"],
        drop_lens_status=config["labels"]["lens_status_drop"],
        corneal_map=config["labels"]["corneal_abnormality_map"],
    )

    splits = load_splits(config["data"]["splits_path"])
    train_df = df.loc[df.index.isin(splits["train_indices"])].copy()
    val_df = df.loc[df.index.isin(splits["val_indices"])].copy()

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Datasets
    train_dataset = VisilantClassificationDataset(
        train_df, config["data"]["image_dir"], label_column, class_names,
    )
    val_dataset = VisilantClassificationDataset(
        val_df, config["data"]["image_dir"], label_column, class_names,
    )

    print(f"Classes: {class_names}")
    print(f"Train samples after filtering: {len(train_dataset)}")
    print(f"Class weights: {train_dataset.get_class_weights().tolist()}")

    # Model
    model = VisionClassifier(
        model_id=config["model"]["model_id"],
        num_classes=len(class_names),
        head_type=args.head_type,
        backbone_mode=args.backbone_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        checkpoint_path=args.checkpoint_path,
        embedding_dim=config["model"]["embedding_dim"],
    )

    class_weights = train_dataset.get_class_weights()

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
        load_best_model_at_end=True,
        metric_for_best_model="macro_auroc",
        greater_is_better=True,
        gradient_checkpointing=True,
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        report_to="wandb",
        run_name=f"classify-{args.experiment_id}-{args.task}",
        seed=config["training"]["seed"],
        remove_unused_columns=False,
    )

    # Trainer
    trainer = ClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=classification_collate_fn,
        compute_metrics=make_compute_metrics(class_names),
        class_weights=class_weights if not args.use_focal_loss else class_weights,
        use_focal=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_weighted_sampler=args.use_weighted_sampler,
        callbacks=[NaNLossCallback()],
    )

    print("Starting training...")
    trainer.train()

    # Persist runtime-discovered bad images
    n_bad = save_runtime_bad_images()
    if n_bad:
        logger.info("Persisted %d runtime-discovered bad images to bad_images.json", n_bad)

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final"))

    # Final evaluation on val set
    print("\nFinal evaluation on val set...")
    eval_results = trainer.evaluate()
    print(f"Val metrics: {eval_results}")

    # Clinical metrics for key classes
    predictions = trainer.predict(val_dataset)
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = predictions.label_ids

    if args.task == "lens_status":
        clinical = compute_clinical_metrics(
            logits, labels, class_names,
            target_classes=["mature_cataract", "aphakia"],
            target_specificity=0.95,
        )
    else:
        clinical = compute_clinical_metrics(
            logits, labels, class_names,
            target_classes=["Active corneal infection"],
            target_specificity=0.95,
        )

    print(f"Clinical metrics: {clinical}")

    # Save all results
    all_metrics = {
        **{k: v for k, v in eval_results.items()},
        **clinical,
        "class_names": class_names,
    }
    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    full_report = compute_classification_metrics(logits, labels, class_names)
    print(f"\n{full_report['classification_report']}")

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(full_report["classification_report"])

    print(f"\n=== Experiment {args.experiment_id} ({args.task}) complete ===")


if __name__ == "__main__":
    main()
