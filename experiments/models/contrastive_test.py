"""Tests for contrastive model loading."""

from unittest.mock import MagicMock, patch

import pytest

from models.contrastive import load_contrastive_model


class TestLoadContrastiveModel:
    @patch("models.contrastive.AutoModel")
    def test_full_mode_returns_model(self, mock_auto):
        mock_model = MagicMock()
        mock_auto.from_pretrained.return_value = mock_model
        result = load_contrastive_model(mode="full")
        assert result is mock_model

    @patch("models.contrastive.get_peft_model")
    @patch("models.contrastive.AutoModel")
    def test_lora_mode_calls_peft(self, mock_auto, mock_peft):
        mock_model = MagicMock()
        mock_auto.from_pretrained.return_value = mock_model
        mock_peft.return_value = mock_model
        result = load_contrastive_model(mode="lora")
        mock_peft.assert_called_once()

    @patch("models.contrastive.AutoModel")
    def test_unknown_mode_raises(self, mock_auto):
        mock_auto.from_pretrained.return_value = MagicMock()
        with pytest.raises(ValueError, match="Unknown mode"):
            load_contrastive_model(mode="bad_mode")
