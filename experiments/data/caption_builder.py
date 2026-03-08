"""Caption construction for contrastive fine-tuning.

Builds text captions from structured metadata fields, with tokenization-aware
truncation to fit within MedSigLIP's 64-token limit.
"""

import re
from transformers import AutoTokenizer


# Normalize image_type values
IMAGE_TYPE_MAP = {
    "diffuse": "diffuse",
    "blue": "blue",
    "diffuse /(tabletop slit lamp/)": "diffuse tabletop",
    "blue /(tabletop slit lamp/)": "blue tabletop",
    "slit": "slit",
    "Slit lamp blue light": "blue",
    "Slit lamp diffuse illumination": "diffuse",
    "lamp diffuse illumination": "diffuse",
}


def _clean_label(value: str) -> str:
    """Replace underscores with spaces in label values."""
    return str(value).replace("_", " ")


def _normalize_image_type(raw: str) -> str:
    return IMAGE_TYPE_MAP.get(raw, raw)


def build_caption_clinical(row: dict) -> str:
    """Build clinical-style caption (primary template).

    Template: Slit lamp photo, {image_type} illumination, {dilation_status}.
    {age:.0f}yo {gender}. Lens: {lens}. Cornea: {cornea}. VA: {va}.
    """
    parts = []

    # Opening clause
    img_type = _normalize_image_type(str(row.get("image_type", "")))
    dilation = str(row.get("dilation_status", ""))
    if dilation in ("-", "-Slit", ""):
        opening = f"Slit lamp photo, {img_type} illumination."
    else:
        opening = f"Slit lamp photo, {img_type} illumination, {dilation}."
    parts.append(opening)

    # Demographics
    age = row.get("age")
    gender = row.get("gender", "")
    if age is not None and str(age) != "" and str(age) != "nan":
        parts.append(f"{float(age):.0f}yo {gender}.")

    # Lens status
    lens = row.get("mapped_lens_status", "")
    if lens and str(lens) != "nan":
        parts.append(f"Lens: {_clean_label(lens)}.")

    # Corneal abnormality
    cornea = row.get("mapped_corneal_abnormality", "")
    if cornea and str(cornea) != "nan":
        parts.append(f"Cornea: {_clean_label(cornea)}.")

    # Visual acuity
    va = row.get("Visual Acuity", "")
    if va and str(va) != "nan":
        parts.append(f"VA: {va}.")

    return " ".join(parts)


def build_caption_label_only(row: dict) -> str:
    """Label-only caption (ablation variant A)."""
    parts = []
    lens = row.get("mapped_lens_status", "")
    if lens and str(lens) != "nan":
        parts.append(f"Lens: {_clean_label(lens)}.")
    cornea = row.get("mapped_corneal_abnormality", "")
    if cornea and str(cornea) != "nan":
        parts.append(f"Cornea: {_clean_label(cornea)}.")
    return " ".join(parts)


def build_caption_sentence(row: dict) -> str:
    """Sentence-style caption (ablation variant C)."""
    lens = _clean_label(row.get("mapped_lens_status", ""))
    cornea = _clean_label(row.get("mapped_corneal_abnormality", ""))
    return f"Slit lamp photograph showing {lens} and {cornea}."


CAPTION_BUILDERS = {
    "clinical": build_caption_clinical,
    "label_only": build_caption_label_only,
    "sentence": build_caption_sentence,
}


class CaptionProcessor:
    """Builds captions with tokenization-aware truncation."""

    def __init__(self, model_id: str = "google/medsiglip-448", max_tokens: int = 64,
                 caption_style: str = "clinical"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_tokens = max_tokens
        # Reserve 2 tokens for BOS/EOS
        self.max_content_tokens = max_tokens - 2
        self.build_fn = CAPTION_BUILDERS[caption_style]

    def build_and_truncate(self, row: dict) -> str:
        """Build caption, truncating clauses if needed to fit token limit."""
        caption = self.build_fn(row)
        tokens = self.tokenizer.encode(caption, add_special_tokens=False)

        if len(tokens) <= self.max_content_tokens:
            return caption

        # Greedy truncation: remove last sentence clause until it fits
        sentences = caption.split(". ")
        while len(sentences) > 1:
            sentences.pop()
            truncated = ". ".join(sentences) + "."
            tokens = self.tokenizer.encode(truncated, add_special_tokens=False)
            if len(tokens) <= self.max_content_tokens:
                return truncated

        # Last resort: let tokenizer truncate
        return caption

    def tokenize(self, caption: str) -> dict:
        """Tokenize a caption with padding to max_tokens."""
        return self.tokenizer(
            caption,
            max_length=self.max_tokens,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
