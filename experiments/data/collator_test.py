"""Tests for collator module."""

import torch
import pytest

from data.collator import classification_collate_fn, contrastive_collate_fn


class TestContrastiveCollateFn:
    def _make_examples(self, n=3):
        return [
            {
                "pixel_values": torch.randn(3, 64, 64),
                "input_ids": [1, 2, 3, 0],
                "attention_mask": [1, 1, 1, 0],
            }
            for _ in range(n)
        ]

    def test_output_keys(self):
        batch = contrastive_collate_fn(self._make_examples())
        assert "pixel_values" in batch
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "return_loss" in batch

    def test_return_loss_flag(self):
        batch = contrastive_collate_fn(self._make_examples())
        assert batch["return_loss"] is True

    def test_pixel_values_shape(self):
        batch = contrastive_collate_fn(self._make_examples(4))
        assert batch["pixel_values"].shape == (4, 3, 64, 64)

    def test_input_ids_shape(self):
        batch = contrastive_collate_fn(self._make_examples(2))
        assert batch["input_ids"].shape == (2, 4)


class TestClassificationCollateFn:
    def _make_examples(self, n=3):
        return [
            {
                "pixel_values": torch.randn(3, 64, 64),
                "labels": i % 4,
            }
            for i in range(n)
        ]

    def test_output_keys(self):
        batch = classification_collate_fn(self._make_examples())
        assert "pixel_values" in batch
        assert "labels" in batch

    def test_labels_dtype(self):
        batch = classification_collate_fn(self._make_examples())
        assert batch["labels"].dtype == torch.long

    def test_batch_size(self):
        batch = classification_collate_fn(self._make_examples(5))
        assert batch["pixel_values"].shape[0] == 5
        assert batch["labels"].shape[0] == 5
