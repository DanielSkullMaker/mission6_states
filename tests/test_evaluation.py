"""Тесты evaluation/metrics.py — метрики оценки классификатора (Фаза C, NB8).

Раньше эти проверки жили только в блоке ``if __name__ == "__main__"`` самого
модуля (Фаза 0 плана рефакторинга требует перенести такие блоки в tests/).
"""

import numpy as np
import pytest

from subspace_conjugacy.evaluation.metrics import (
    confidence_summary,
    evaluate_classifier,
    per_class_accuracy,
)


class TestPerClassAccuracy:
    def test_perfect_predictions_give_accuracy_one_per_class(self):
        y_true = np.array(["glioma"] * 5 + ["meningioma"] * 5)
        acc = per_class_accuracy(y_true, y_true)
        assert acc == {"glioma": 1.0, "meningioma": 1.0}

    def test_partial_errors_reduce_only_affected_class(self):
        y_true = np.array(["glioma"] * 4 + ["meningioma"] * 4)
        y_pred = y_true.copy()
        y_pred[0] = "meningioma"  # одна ошибка в классе glioma

        acc = per_class_accuracy(y_true, y_pred)

        assert acc["glioma"] == pytest.approx(0.75)
        assert acc["meningioma"] == pytest.approx(1.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            per_class_accuracy(["a", "b", "c"], ["a", "b"])


class TestEvaluateClassifier:
    def test_report_structure_and_keys(self):
        y_true = np.array(["glioma"] * 5 + ["meningioma"] * 5 + ["pituitary"] * 5)
        report = evaluate_classifier(y_true, y_true)

        assert set(report.keys()) == {
            "accuracy",
            "per_class_accuracy",
            "confusion_matrix",
            "labels",
            "n_samples",
        }
        assert report["accuracy"] == 1.0
        assert report["n_samples"] == 15
        assert report["confusion_matrix"].shape == (3, 3)

    def test_accuracy_reflects_errors(self):
        y_true = np.array(["glioma"] * 5 + ["meningioma"] * 5)
        y_pred = y_true.copy()
        y_pred[0] = "meningioma"
        y_pred[6] = "glioma"

        report = evaluate_classifier(y_true, y_pred)

        assert report["accuracy"] == pytest.approx(8 / 10)

    def test_explicit_labels_control_confusion_matrix_order(self):
        y_true = np.array(["a", "b", "a", "b"])
        y_pred = np.array(["a", "a", "a", "b"])

        report = evaluate_classifier(y_true, y_pred, labels=["b", "a"])

        np.testing.assert_array_equal(report["labels"], ["b", "a"])
        assert report["confusion_matrix"].shape == (2, 2)

    def test_accuracy_always_within_unit_range(self):
        rng = np.random.default_rng(0)
        y_true = rng.choice(["glioma", "meningioma", "pituitary"], size=50)
        y_pred = rng.choice(["glioma", "meningioma", "pituitary"], size=50)

        report = evaluate_classifier(y_true, y_pred)
        assert 0.0 <= report["accuracy"] <= 1.0


class TestConfidenceSummary:
    def test_summary_keys_and_values(self):
        confidence = np.array([1.0, 2.0, -0.5, 3.0])
        summary = confidence_summary(confidence)

        assert set(summary.keys()) == {"mean", "min", "max", "fraction_negative"}
        assert summary["min"] == -0.5
        assert summary["max"] == 3.0
        assert summary["mean"] == pytest.approx(np.mean(confidence))
        assert summary["fraction_negative"] == pytest.approx(0.25)

    def test_all_non_negative_gives_zero_fraction_negative(self):
        confidence = np.array([0.1, 0.5, 2.0])
        summary = confidence_summary(confidence)
        assert summary["fraction_negative"] == 0.0
