"""PyTorch datasets for contrastive and classification fine-tuning."""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile

# Allow PIL to load truncated images (zero-fills missing bytes) instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

from data.caption_builder import CaptionProcessor


def get_image_transform(size: int = 448):
    """Standard MedSigLIP image preprocessing: resize, to tensor, normalize to [-1, 1]."""
    return Compose([
        Resize((size, size), interpolation=InterpolationMode.BILINEAR),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class VisilantContrastiveDataset(Dataset):
    """Dataset for contrastive (image-text) fine-tuning.

    Each sample returns preprocessed pixel values and tokenized caption.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        caption_processor: CaptionProcessor,
        image_size: int = 448,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.caption_processor = caption_processor
        self.transform = get_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load and transform image
        img_path = os.path.join(self.image_dir, row["image_file"])
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        # Build and tokenize caption
        caption = self.caption_processor.build_and_truncate(row.to_dict())
        tokens = self.caption_processor.tokenize(caption)

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }


class VisilantClassificationDataset(Dataset):
    """Dataset for classification fine-tuning.

    Returns preprocessed pixel values and integer label for a given task.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        label_column: str,
        class_names: list[str],
        image_size: int = 448,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.label_column = label_column
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.transform = get_image_transform(image_size)

        # Pre-filter to valid labels
        valid_mask = self.df[label_column].isin(self.class_to_idx)
        if not valid_mask.all():
            n_dropped = (~valid_mask).sum()
            print(f"Warning: dropping {n_dropped} samples with unknown labels")
            self.df = self.df[valid_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, row["image_file"])
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        label = self.class_to_idx[row[self.label_column]]

        return {
            "pixel_values": pixel_values,
            "labels": label,
        }

    def get_class_weights(self, clamp_range: tuple[float, float] = (0.5, 10.0)) -> torch.Tensor:
        """Compute inverse-frequency class weights, clamped to range."""
        counts = self.df[self.label_column].value_counts()
        total = len(self.df)
        weights = []
        for cls in self.class_names:
            count = counts.get(cls, 1)
            w = total / (self.num_classes * count)
            w = max(clamp_range[0], min(clamp_range[1], w))
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self, cap: float = 10.0) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights(clamp_range=(1.0, cap))
        sample_weights = torch.tensor(
            [class_weights[self.class_to_idx[row[self.label_column]]].item()
             for _, row in self.df.iterrows()],
            dtype=torch.float64,
        )
        return sample_weights
