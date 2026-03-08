"""Evaluation routines: zero-shot classification, retrieval, classification metrics."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from utils.metrics import compute_classification_metrics, compute_clinical_metrics


@torch.no_grad()
def zero_shot_evaluate(
    model,
    processor,
    dataloader: DataLoader,
    class_prompts: list[str],
    class_names: list[str],
    label_column: str,
    device: str = "cuda",
) -> dict:
    """Zero-shot classification using cosine similarity between image and text embeddings.

    Args:
        model: SiglipModel (contrastive model with both encoders).
        processor: AutoProcessor for tokenization.
        dataloader: yields dicts with 'pixel_values' and label info.
        class_prompts: text prompts for each class, e.g. ["slit lamp showing clear crystalline lens", ...].
        class_names: names for metric reporting.
        label_column: which label to evaluate.
        device: cuda or cpu.
    """
    model.eval()
    model = model.to(device)

    # Encode class prompts
    text_inputs = processor.tokenizer(
        class_prompts,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    all_logits = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Zero-shot eval"):
        pixel_values = batch["pixel_values"].to(device)
        image_embeds = model.get_image_features(pixel_values=pixel_values)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = image_embeds @ text_embeds.T  # (batch, num_classes)
        all_logits.append(similarity.cpu().numpy())
        all_labels.append(batch["labels"].numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return compute_classification_metrics(all_logits, all_labels, class_names)


@torch.no_grad()
def retrieval_evaluate(
    model,
    processor,
    dataloader: DataLoader,
    device: str = "cuda",
    max_samples: int = 2000,
) -> dict:
    """Image-text retrieval metrics: Recall@1/5/10 for image->text and text->image.

    Assumes dataloader yields contrastive samples with pixel_values and input_ids.
    """
    model.eval()
    model = model.to(device)

    all_image_embeds = []
    all_text_embeds = []
    count = 0

    for batch in tqdm(dataloader, desc="Retrieval eval"):
        if count >= max_samples:
            break

        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        image_embeds = model.get_image_features(pixel_values=pixel_values)
        text_embeds = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())
        count += len(pixel_values)

    image_embeds = torch.cat(all_image_embeds, dim=0)[:max_samples]
    text_embeds = torch.cat(all_text_embeds, dim=0)[:max_samples]

    # Similarity matrix
    sim = image_embeds @ text_embeds.T  # (N, N)
    n = sim.shape[0]

    metrics = {}
    for direction, matrix in [("i2t", sim), ("t2i", sim.T)]:
        ranks = []
        for i in range(n):
            sorted_indices = matrix[i].argsort(descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
        ranks = np.array(ranks)
        for k in [1, 5, 10]:
            metrics[f"{direction}_recall@{k}"] = (ranks < k).mean()

    return metrics


@torch.no_grad()
def extract_embeddings(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract vision embeddings and labels from a classification dataloader.

    Returns (embeddings, labels) as numpy arrays.
    """
    model.eval()
    if hasattr(model, "vision_model"):
        vision_model = model.vision_model.to(device)
    else:
        vision_model = model.to(device)

    all_embeds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        pixel_values = batch["pixel_values"].to(device)
        outputs = vision_model(pixel_values=pixel_values)
        embeds = outputs.pooler_output
        all_embeds.append(embeds.cpu().numpy())
        if "labels" in batch:
            all_labels.append(batch["labels"].numpy())

    embeddings = np.concatenate(all_embeds, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    return embeddings, labels
