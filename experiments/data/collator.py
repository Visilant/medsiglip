"""Data collators for contrastive and classification training."""

import logging

import torch

logger = logging.getLogger(__name__)


def contrastive_collate_fn(examples: list[dict | None]) -> dict:
    """Collate for SiglipModel contrastive training."""
    valid = [ex for ex in examples if ex is not None]
    if not valid:
        raise RuntimeError(
            f"Entire batch of {len(examples)} samples was None — systemic data problem"
        )
    if len(valid) < len(examples):
        logger.warning("Filtered %d bad samples from batch", len(examples) - len(valid))
    pixel_values = torch.stack([torch.tensor(ex["pixel_values"]) if not isinstance(ex["pixel_values"], torch.Tensor) else ex["pixel_values"] for ex in valid])
    input_ids = torch.tensor([ex["input_ids"] for ex in valid])
    attention_mask = torch.tensor([ex["attention_mask"] for ex in valid])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }


def classification_collate_fn(examples: list[dict | None]) -> dict:
    """Collate for classification training."""
    valid = [ex for ex in examples if ex is not None]
    if not valid:
        raise RuntimeError(
            f"Entire batch of {len(examples)} samples was None — systemic data problem"
        )
    if len(valid) < len(examples):
        logger.warning("Filtered %d bad samples from batch", len(examples) - len(valid))
    pixel_values = torch.stack([torch.tensor(ex["pixel_values"]) if not isinstance(ex["pixel_values"], torch.Tensor) else ex["pixel_values"] for ex in valid])
    labels = torch.tensor([ex["labels"] for ex in valid], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }
