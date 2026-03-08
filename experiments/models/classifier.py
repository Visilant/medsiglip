"""Classification model: vision backbone + classification head(s)."""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


class VisionClassifier(nn.Module):
    """MedSigLIP vision encoder + classification head.

    Supports linear probe, MLP head, and full/LoRA fine-tuning of backbone.
    """

    def __init__(
        self,
        model_id: str = "google/medsiglip-448",
        num_classes: int = 7,
        head_type: str = "linear",  # "linear" or "mlp"
        backbone_mode: str = "frozen",  # "frozen", "full", "lora"
        lora_r: int = 16,
        lora_alpha: int = 32,
        checkpoint_path: str | None = None,
        embedding_dim: int = 1152,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Load backbone
        if checkpoint_path:
            base_model = AutoModel.from_pretrained(checkpoint_path)
        else:
            base_model = AutoModel.from_pretrained(model_id)
        self.vision_model = base_model.vision_model

        # Apply backbone mode
        if backbone_mode == "frozen":
            for param in self.vision_model.parameters():
                param.requires_grad = False
        elif backbone_mode == "lora":
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.vision_model = get_peft_model(self.vision_model, lora_config)
            self.vision_model.print_trainable_parameters()
        # "full" mode: all params trainable (default)

        # Classification head
        if head_type == "linear":
            self.head = nn.Linear(embedding_dim, num_classes)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        # Extract vision features
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        # Pool: use the pooler output (CLS-like)
        pooled = vision_outputs.pooler_output  # (batch, embedding_dim)

        logits = self.head(pooled)

        loss = None
        if labels is not None:
            # Loss is computed externally via Trainer's compute_loss_func
            # but provide a default for standalone use
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


class MultiTaskClassifier(nn.Module):
    """Shared backbone with separate heads for lens status and corneal abnormality."""

    def __init__(
        self,
        model_id: str = "google/medsiglip-448",
        num_lens_classes: int = 7,
        num_corneal_classes: int = 4,
        head_type: str = "linear",
        backbone_mode: str = "frozen",
        checkpoint_path: str | None = None,
        embedding_dim: int = 1152,
        dropout: float = 0.1,
        loss_weight_lens: float = 0.5,
        loss_weight_corneal: float = 0.5,
    ):
        super().__init__()
        self.loss_weight_lens = loss_weight_lens
        self.loss_weight_corneal = loss_weight_corneal

        # Shared backbone
        if checkpoint_path:
            base_model = AutoModel.from_pretrained(checkpoint_path)
        else:
            base_model = AutoModel.from_pretrained(model_id)
        self.vision_model = base_model.vision_model

        if backbone_mode == "frozen":
            for param in self.vision_model.parameters():
                param.requires_grad = False

        # Separate heads
        def make_head(n_classes):
            if head_type == "linear":
                return nn.Linear(embedding_dim, n_classes)
            return nn.Sequential(
                nn.Linear(embedding_dim, 512), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, n_classes),
            )

        self.lens_head = make_head(num_lens_classes)
        self.corneal_head = make_head(num_corneal_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        lens_labels: torch.Tensor | None = None,
        corneal_labels: torch.Tensor | None = None,
    ) -> dict:
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output

        lens_logits = self.lens_head(pooled)
        corneal_logits = self.corneal_head(pooled)

        loss = None
        if lens_labels is not None and corneal_labels is not None:
            ce = nn.CrossEntropyLoss()
            loss = (self.loss_weight_lens * ce(lens_logits, lens_labels) +
                    self.loss_weight_corneal * ce(corneal_logits, corneal_labels))

        return {
            "loss": loss,
            "lens_logits": lens_logits,
            "corneal_logits": corneal_logits,
        }
