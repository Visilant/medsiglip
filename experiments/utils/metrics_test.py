"""Tests for metrics module."""

import numpy as np
import pytest

from utils.metrics import (
    compute_classification_metrics,
    compute_clinical_metrics,
    youdens_j_threshold,
)


def _make_perfect_predictions(n=100, num_classes=3):
    """Create logits where argmax matches labels perfectly."""
    labels = np.array([i % num_classes for i in range(n)])
    logits = np.full((n, num_classes), -10.0)
    for i in range(n):
        logits[i, labels[i]] = 10.0
    return logits, labels


def _make_random_predictions(n=200, num_classes=3, seed=42):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, num_classes, n)
    logits = rng.randn(n, num_classes)
    return logits, labels


class TestComputeClassificationMetrics:
    def test_perfect_accuracy(self):
        logits, labels = _make_perfect_predictions()
        class_names = ["A", "B", "C"]
        m = compute_classification_metrics(logits, labels, class_names)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_perfect_f1(self):
        logits, labels = _make_perfect_predictions()
        m = compute_classification_metrics(logits, labels, ["A", "B", "C"])
        assert m["macro_f1"] == pytest.approx(1.0)

    def test_random_accuracy_range(self):
        logits, labels = _make_random_predictions()
        m = compute_classification_metrics(logits, labels, ["A", "B", "C"])
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self):
        logits, labels = _make_random_predictions(num_classes=4)
        m = compute_classification_metrics(logits, labels, ["A", "B", "C", "D"])
        cm = np.array(m["confusion_matrix"])
        assert cm.shape == (4, 4)

    def test_per_class_f1_keys(self):
        logits, labels = _make_random_predictions()
        m = compute_classification_metrics(logits, labels, ["cat", "dog", "bird"])
        assert "f1_cat" in m
        assert "f1_dog" in m
        assert "f1_bird" in m

    def test_auroc_present(self):
        logits, labels = _make_random_predictions()
        m = compute_classification_metrics(logits, labels, ["A", "B", "C"])
        assert "macro_auroc" in m

    def test_zero_division_handled(self):
        # All same label — macro_f1 should still work (zero_division=0)
        n = 50
        logits = np.random.randn(n, 3)
        labels = np.zeros(n, dtype=int)
        m = compute_classification_metrics(logits, labels, ["A", "B", "C"])
        assert isinstance(m["macro_f1"], float)

    def test_single_class_auroc_not_computed(self):
        # Only one class present — AUROC block is skipped (< 2 classes present)
        n = 50
        logits = np.random.randn(n, 2)
        labels = np.zeros(n, dtype=int)
        m = compute_classification_metrics(logits, labels, ["A", "B"])
        assert "macro_auroc" not in m


class TestComputeClinicalMetrics:
    def test_sensitivity_key(self):
        logits, labels = _make_random_predictions(n=200, num_classes=3)
        m = compute_clinical_metrics(logits, labels, ["A", "B", "C"], ["A"])
        assert "sensitivity_at_95spec_A" in m
        assert "threshold_A" in m

    def test_sensitivity_range(self):
        logits, labels = _make_random_predictions(n=200, num_classes=3)
        m = compute_clinical_metrics(logits, labels, ["A", "B", "C"], ["B"])
        sens = m["sensitivity_at_95spec_B"]
        assert 0.0 <= sens <= 1.0

    def test_unknown_target_skipped(self):
        logits, labels = _make_random_predictions()
        m = compute_clinical_metrics(logits, labels, ["A", "B", "C"], ["NONEXISTENT"])
        assert len(m) == 0


class TestYoudensJThreshold:
    def test_returns_three_values(self):
        logits, labels = _make_random_predictions(n=200, num_classes=2)
        thresh, sens, spec = youdens_j_threshold(logits, labels, class_idx=0)
        assert isinstance(thresh, float)
        assert isinstance(sens, (float, np.floating))
        assert isinstance(spec, (float, np.floating))

    def test_threshold_range(self):
        logits, labels = _make_random_predictions(n=200, num_classes=2)
        thresh, sens, spec = youdens_j_threshold(logits, labels, class_idx=0)
        assert 0.0 <= thresh <= 1.0
        assert 0.0 <= sens <= 1.0
        assert 0.0 <= spec <= 1.0
