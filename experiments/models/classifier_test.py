"""Tests for classifier module."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch
import pytest

from models.classifier import VisionClassifier


def _make_mock_auto_model():
    """Create a mock AutoModel with a vision_model that returns pooler_output."""
    mock_model = MagicMock()
    mock_vision = MagicMock()
    mock_vision.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    mock_model.vision_model = mock_vision
    return mock_model


class TestVisionClassifier:
    @patch("models.classifier.AutoModel")
    def test_linear_head_output_shape(self, mock_auto):
        mock_auto.from_pretrained.return_value = _make_mock_auto_model()
        model = VisionClassifier(
            num_classes=4, head_type="linear", backbone_mode="frozen", embedding_dim=32
        )
        # Simulate forward
        fake_pooled = torch.randn(2, 32)
        model.vision_model.return_value = SimpleNamespace(pooler_output=fake_pooled)
        out = model(torch.randn(2, 3, 64, 64))
        assert out["logits"].shape == (2, 4)

    @patch("models.classifier.AutoModel")
    def test_mlp_head_output_shape(self, mock_auto):
        mock_auto.from_pretrained.return_value = _make_mock_auto_model()
        model = VisionClassifier(
            num_classes=7, head_type="mlp", backbone_mode="frozen", embedding_dim=32
        )
        fake_pooled = torch.randn(3, 32)
        model.vision_model.return_value = SimpleNamespace(pooler_output=fake_pooled)
        out = model(torch.randn(3, 3, 64, 64))
        assert out["logits"].shape == (3, 7)

    @patch("models.classifier.AutoModel")
    def test_frozen_backbone_no_grad(self, mock_auto):
        param1 = torch.nn.Parameter(torch.randn(2, 2))
        mock_model = _make_mock_auto_model()
        mock_model.vision_model.parameters.return_value = [param1]
        mock_auto.from_pretrained.return_value = mock_model
        VisionClassifier(
            num_classes=2, backbone_mode="frozen", embedding_dim=32
        )
        assert not param1.requires_grad

    @patch("models.classifier.AutoModel")
    def test_unknown_head_raises(self, mock_auto):
        mock_auto.from_pretrained.return_value = _make_mock_auto_model()
        with pytest.raises(ValueError, match="Unknown head_type"):
            VisionClassifier(
                num_classes=2, head_type="transformer", backbone_mode="frozen", embedding_dim=32
            )

    @patch("models.classifier.AutoModel")
    def test_forward_with_labels_has_loss(self, mock_auto):
        mock_auto.from_pretrained.return_value = _make_mock_auto_model()
        model = VisionClassifier(
            num_classes=3, head_type="linear", backbone_mode="frozen", embedding_dim=32
        )
        fake_pooled = torch.randn(2, 32)
        model.vision_model.return_value = SimpleNamespace(pooler_output=fake_pooled)
        out = model(torch.randn(2, 3, 64, 64), labels=torch.tensor([0, 1]))
        assert out["loss"] is not None

    @patch("models.classifier.AutoModel")
    def test_forward_without_labels_no_loss(self, mock_auto):
        mock_auto.from_pretrained.return_value = _make_mock_auto_model()
        model = VisionClassifier(
            num_classes=3, head_type="linear", backbone_mode="frozen", embedding_dim=32
        )
        fake_pooled = torch.randn(2, 32)
        model.vision_model.return_value = SimpleNamespace(pooler_output=fake_pooled)
        out = model(torch.randn(2, 3, 64, 64))
        assert out["loss"] is None
