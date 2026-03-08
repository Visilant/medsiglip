"""Contrastive model setup: full fine-tune, LoRA, and partial freeze configs."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


def load_contrastive_model(
    model_id: str = "google/medsiglip-448",
    mode: str = "full",  # "full", "lora", "partial_freeze"
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: list[str] | None = None,
    freeze_layers_before: int | None = None,
) -> AutoModel:
    """Load SiglipModel for contrastive training with different fine-tuning strategies.

    Args:
        mode: "full" = all params trainable,
              "lora" = LoRA adapters on both encoders,
              "partial_freeze" = freeze early vision layers + full text encoder.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_target_modules: LoRA target modules. Default: q/k/v/out projections.
        freeze_layers_before: For partial_freeze, freeze vision layers before this index.
    """
    model = AutoModel.from_pretrained(model_id)

    if mode == "full":
        # All parameters trainable (default)
        return model

    elif mode == "lora":
        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "out_proj",
            ]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    elif mode == "partial_freeze":
        # Freeze all vision encoder layers except the last N
        if freeze_layers_before is None:
            # Default: keep last 6 layers trainable (SigLIP has 27 vision layers)
            freeze_layers_before = 21

        # Freeze vision encoder embedding and early layers
        for name, param in model.named_parameters():
            if "vision_model" in name:
                # Check if this is an encoder layer
                if "encoder.layers." in name:
                    layer_num = int(name.split("encoder.layers.")[1].split(".")[0])
                    if layer_num < freeze_layers_before:
                        param.requires_grad = False
                elif "embeddings" in name:
                    param.requires_grad = False
                # Post-layernorm and head stay trainable

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Partial freeze: {trainable:,} / {total:,} params trainable ({100*trainable/total:.1f}%)")
        return model

    else:
        raise ValueError(f"Unknown mode: {mode}")
