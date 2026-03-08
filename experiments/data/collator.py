"""Data collators for contrastive and classification training."""

import torch


def contrastive_collate_fn(examples: list[dict]) -> dict:
    """Collate for SiglipModel contrastive training."""
    pixel_values = torch.stack([torch.tensor(ex["pixel_values"]) if not isinstance(ex["pixel_values"], torch.Tensor) else ex["pixel_values"] for ex in examples])
    input_ids = torch.tensor([ex["input_ids"] for ex in examples])
    attention_mask = torch.tensor([ex["attention_mask"] for ex in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }


def classification_collate_fn(examples: list[dict]) -> dict:
    """Collate for classification training."""
    pixel_values = torch.stack([torch.tensor(ex["pixel_values"]) if not isinstance(ex["pixel_values"], torch.Tensor) else ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }
