"""Tests for caption_builder module."""

from unittest.mock import MagicMock, patch

import pytest

from data.caption_builder import (
    CAPTION_BUILDERS,
    CaptionProcessor,
    _clean_label,
    _normalize_image_type,
    build_caption_clinical,
    build_caption_label_only,
    build_caption_sentence,
)


class TestCleanLabel:
    def test_replaces_underscores(self):
        assert _clean_label("active_corneal_infection") == "active corneal infection"

    def test_no_underscores(self):
        assert _clean_label("Normal") == "Normal"

    def test_casts_to_str(self):
        assert _clean_label(123) == "123"


class TestNormalizeImageType:
    def test_known_mapping(self):
        assert _normalize_image_type("Slit lamp blue light") == "blue"
        assert _normalize_image_type("diffuse") == "diffuse"

    def test_unknown_passes_through(self):
        assert _normalize_image_type("unknown_type") == "unknown_type"


class TestBuildCaptionClinical:
    def test_full_row(self):
        row = {
            "image_type": "diffuse",
            "dilation_status": "dilated",
            "age": 45,
            "gender": "M",
            "mapped_lens_status": "immature_cataract",
            "mapped_corneal_abnormality": "Normal",
            "Visual Acuity": "6/12",
        }
        caption = build_caption_clinical(row)
        assert "Slit lamp photo" in caption
        assert "dilated" in caption
        assert "45yo M" in caption
        assert "Lens: immature cataract" in caption
        assert "Cornea: Normal" in caption
        assert "VA: 6/12" in caption

    def test_empty_dilation(self):
        row = {"image_type": "blue", "dilation_status": ""}
        caption = build_caption_clinical(row)
        assert "blue illumination." in caption
        assert caption.count("illumination") == 1

    def test_dash_dilation_excluded(self):
        row = {"image_type": "slit", "dilation_status": "-"}
        caption = build_caption_clinical(row)
        assert ", -." not in caption

    def test_nan_age_excluded(self):
        row = {"image_type": "blue", "age": "nan", "gender": "F"}
        caption = build_caption_clinical(row)
        assert "yo" not in caption

    def test_nan_lens_excluded(self):
        row = {"image_type": "blue", "mapped_lens_status": "nan"}
        caption = build_caption_clinical(row)
        assert "Lens:" not in caption


class TestBuildCaptionLabelOnly:
    def test_both_labels(self):
        row = {"mapped_lens_status": "PCIOL", "mapped_corneal_abnormality": "Normal"}
        caption = build_caption_label_only(row)
        assert "Lens: PCIOL." in caption
        assert "Cornea: Normal." in caption

    def test_missing_labels(self):
        caption = build_caption_label_only({})
        assert caption == ""


class TestBuildCaptionSentence:
    def test_sentence_format(self):
        row = {"mapped_lens_status": "clear_crystalline_lens", "mapped_corneal_abnormality": "Normal"}
        caption = build_caption_sentence(row)
        assert caption == "Slit lamp photograph showing clear crystalline lens and Normal."


class TestCaptionProcessor:
    @patch("data.caption_builder.AutoTokenizer")
    def test_build_and_truncate_short_caption(self, mock_tokenizer_cls):
        mock_tok = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tok
        # Return 10 tokens — well under 62 limit
        mock_tok.encode.return_value = list(range(10))

        proc = CaptionProcessor.__new__(CaptionProcessor)
        proc.tokenizer = mock_tok
        proc.max_tokens = 64
        proc.max_content_tokens = 62
        proc.build_fn = build_caption_clinical

        row = {"image_type": "blue", "dilation_status": ""}
        result = proc.build_and_truncate(row)
        assert "Slit lamp photo" in result

    @patch("data.caption_builder.AutoTokenizer")
    def test_build_and_truncate_triggers_truncation(self, mock_tokenizer_cls):
        mock_tok = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tok
        # First call returns too many tokens, second (after truncation) returns few
        mock_tok.encode.side_effect = [list(range(70)), list(range(30))]

        proc = CaptionProcessor.__new__(CaptionProcessor)
        proc.tokenizer = mock_tok
        proc.max_tokens = 64
        proc.max_content_tokens = 62
        proc.build_fn = build_caption_clinical

        row = {
            "image_type": "blue",
            "dilation_status": "dilated",
            "age": 50,
            "gender": "M",
            "mapped_lens_status": "PCIOL",
            "mapped_corneal_abnormality": "Normal",
            "Visual Acuity": "6/6",
        }
        result = proc.build_and_truncate(row)
        # Should have removed trailing clauses
        assert result.endswith(".")
