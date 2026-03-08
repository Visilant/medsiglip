"""Evaluation metrics: AUROC, F1, confusion matrix, clinical metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute comprehensive classification metrics.

    Args:
        logits: raw model logits, shape (N, num_classes)
        labels: integer labels, shape (N,)
        class_names: list of class names

    Returns:
        dict with accuracy, macro_f1, per_class_f1, macro_auroc, per_class_auroc,
        confusion_matrix, classification_report.
    """
    preds = logits.argmax(axis=1)
    num_classes = len(class_names)

    # Probabilities via softmax
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }

    # Per-class F1
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        if i < len(per_class_f1):
            metrics[f"f1_{name}"] = per_class_f1[i]

    # AUROC (one-vs-rest)
    try:
        # One-hot encode labels for AUROC
        labels_onehot = np.zeros((len(labels), num_classes))
        labels_onehot[np.arange(len(labels)), labels] = 1

        # Only compute for classes present in labels
        present = labels_onehot.sum(axis=0) > 0
        if present.sum() >= 2:
            metrics["macro_auroc"] = roc_auc_score(
                labels_onehot[:, present],
                probs[:, present],
                average="macro",
                multi_class="ovr",
            )
            per_class_auroc = roc_auc_score(
                labels_onehot[:, present],
                probs[:, present],
                average=None,
                multi_class="ovr",
            )
            present_idx = np.where(present)[0]
            for j, auroc_val in enumerate(per_class_auroc):
                metrics[f"auroc_{class_names[present_idx[j]]}"] = auroc_val
    except ValueError:
        metrics["macro_auroc"] = float("nan")

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(labels, preds).tolist()

    # Full classification report
    metrics["classification_report"] = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )

    return metrics


def compute_clinical_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    target_classes: list[str],
    target_specificity: float = 0.95,
) -> dict:
    """Compute clinical metrics: sensitivity at fixed specificity.

    For each target class, find the threshold on the predicted probability
    that achieves the target specificity, then report sensitivity at that threshold.
    """
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    num_classes = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    metrics = {}
    for target in target_classes:
        if target not in class_to_idx:
            continue
        idx = class_to_idx[target]

        # Binary: target class vs rest
        binary_labels = (labels == idx).astype(int)
        scores = probs[:, idx]

        if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
            metrics[f"sensitivity_at_{int(target_specificity*100)}spec_{target}"] = float("nan")
            continue

        # Find threshold for target specificity
        negatives = scores[binary_labels == 0]
        threshold = np.percentile(negatives, target_specificity * 100)

        # Sensitivity at that threshold
        positives = scores[binary_labels == 1]
        sensitivity = (positives >= threshold).mean()

        metrics[f"sensitivity_at_{int(target_specificity*100)}spec_{target}"] = sensitivity
        metrics[f"threshold_{target}"] = threshold

    return metrics


def youdens_j_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    class_idx: int,
) -> tuple[float, float, float]:
    """Find optimal threshold using Youden's J statistic for a binary classification.

    Returns (threshold, sensitivity, specificity).
    """
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    binary_labels = (labels == class_idx).astype(int)
    scores = probs[:, class_idx]

    thresholds = np.linspace(0, 1, 1000)
    best_j = -1
    best_thresh = 0.5
    best_sens = 0
    best_spec = 0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (binary_labels == 1)).sum()
        tn = ((preds == 0) & (binary_labels == 0)).sum()
        fp = ((preds == 1) & (binary_labels == 0)).sum()
        fn = ((preds == 0) & (binary_labels == 1)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1

        if j > best_j:
            best_j = j
            best_thresh = t
            best_sens = sens
            best_spec = spec

    return best_thresh, best_sens, best_spec
