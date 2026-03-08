"""Tests for dataset module."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from data.dataset import (
    VisilantClassificationDataset,
    get_image_transform,
)


class TestGetImageTransform:
    def test_output_shape(self):
        transform = get_image_transform(448)
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        result = transform(img)
        assert result.shape == (3, 448, 448)

    def test_output_dtype(self):
        transform = get_image_transform(224)
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = transform(img)
        assert result.dtype == torch.float32

    def test_output_range(self):
        transform = get_image_transform(448)
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = transform(img)
        assert result.min() >= -1.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6


class TestVisilantClassificationDataset:
    @pytest.fixture
    def tmp_dataset(self, tmp_path):
        """Create a temp directory with synthetic images and a DataFrame."""
        image_dir = str(tmp_path)
        # Create 5 dummy images
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(image_dir, f"img_{i}.jpg"))

        df = pd.DataFrame({
            "image_file": [f"img_{i}.jpg" for i in range(5)],
            "label": ["A", "A", "B", "B", "A"],
        })
        class_names = ["A", "B"]
        return df, image_dir, class_names

    def test_len(self, tmp_dataset):
        df, image_dir, class_names = tmp_dataset
        ds = VisilantClassificationDataset(df, image_dir, "label", class_names)
        assert len(ds) == 5

    def test_getitem_keys(self, tmp_dataset):
        df, image_dir, class_names = tmp_dataset
        ds = VisilantClassificationDataset(df, image_dir, "label", class_names, image_size=64)
        sample = ds[0]
        assert "pixel_values" in sample
        assert "labels" in sample

    def test_getitem_pixel_shape(self, tmp_dataset):
        df, image_dir, class_names = tmp_dataset
        ds = VisilantClassificationDataset(df, image_dir, "label", class_names, image_size=64)
        sample = ds[0]
        assert sample["pixel_values"].shape == (3, 64, 64)

    def test_getitem_label_type(self, tmp_dataset):
        df, image_dir, class_names = tmp_dataset
        ds = VisilantClassificationDataset(df, image_dir, "label", class_names, image_size=64)
        sample = ds[0]
        assert isinstance(sample["labels"], int)

    def test_unknown_label_filtered(self, tmp_dataset):
        df, image_dir, class_names = tmp_dataset
        df_extra = pd.concat([df, pd.DataFrame({
            "image_file": ["img_0.jpg"],
            "label": ["UNKNOWN"],
        })], ignore_index=True)
        ds = VisilantClassificationDataset(df_extra, image_dir, "label", class_names)
        assert len(ds) == 5  # UNKNOWN row dropped

    def test_class_weights_shape(self, tmp_dataset):
        df, image_dir, class_names = tmp_dataset
        ds = VisilantClassificationDataset(df, image_dir, "label", class_names)
        weights = ds.get_class_weights()
        assert weights.shape == (2,)
        assert weights.dtype == torch.float32

    def test_class_weights_clamped(self, tmp_dataset):
        df, image_dir, class_names = tmp_dataset
        ds = VisilantClassificationDataset(df, image_dir, "label", class_names)
        weights = ds.get_class_weights(clamp_range=(0.5, 10.0))
        assert (weights >= 0.5).all()
        assert (weights <= 10.0).all()

    def test_sample_weights_length(self, tmp_dataset):
        df, image_dir, class_names = tmp_dataset
        ds = VisilantClassificationDataset(df, image_dir, "label", class_names)
        sample_weights = ds.get_sample_weights()
        assert len(sample_weights) == len(ds)
        assert sample_weights.dtype == torch.float64
