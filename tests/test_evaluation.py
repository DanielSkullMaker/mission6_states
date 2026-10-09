"""Тесты evaluation/metrics.py — метрики оценки классификатора (Фаза C, NB8).

Раньше эти проверки жили только в блоке ``if __name__ == "__main__"`` самого
модуля (Фаза 0 плана рефакторинга требует перенести такие блоки в tests/).
"""

import numpy as np
import pytest

from subspace_conjugacy.evaluation.metrics import (
    build_confusion_matrix,
    confidence_summary,
    confusion_matrix_to_table,
    evaluate_classifier,
    per_class_accuracy,
    plot_confusion_matrix,
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


class TestBuildConfusionMatrix:
    def test_matches_evaluate_classifier_raw_counts(self):
        y_true = np.array(["glioma"] * 4 + ["meningioma"] * 4)
        y_pred = y_true.copy()
        y_pred[0] = "meningioma"

        report = evaluate_classifier(y_true, y_pred)
        result = build_confusion_matrix(y_true, y_pred)

        np.testing.assert_array_equal(result["matrix"], report["confusion_matrix"])
        np.testing.assert_array_equal(result["labels"], report["labels"])
        assert result["normalize"] is None

    def test_explicit_labels_control_order(self):
        y_true = np.array(["a", "b", "a", "b"])
        y_pred = np.array(["a", "a", "a", "b"])

        result = build_confusion_matrix(y_true, y_pred, labels=["b", "a"])

        np.testing.assert_array_equal(result["labels"], ["b", "a"])
        assert result["matrix"].shape == (2, 2)

    def test_normalize_true_rows_sum_to_one(self):
        y_true = np.array(["a"] * 4 + ["b"] * 4)
        y_pred = np.array(["a", "a", "a", "b", "b", "b", "b", "a"])

        result = build_confusion_matrix(y_true, y_pred, normalize="true")

        np.testing.assert_allclose(result["matrix"].sum(axis=1), [1.0, 1.0])
        assert result["normalize"] == "true"

    def test_normalize_all_sums_to_one_total(self):
        y_true = np.array(["a"] * 3 + ["b"] * 3)
        y_pred = y_true.copy()

        result = build_confusion_matrix(y_true, y_pred, normalize="all")

        assert result["matrix"].sum() == pytest.approx(1.0)


class TestConfusionMatrixToTable:
    def test_header_and_rows_structure(self):
        matrix = np.array([[8, 2], [1, 9]])
        header, rows = confusion_matrix_to_table(matrix, ["glioma", "meningioma"])

        assert header == ["Истина \\ Предсказано", "glioma", "meningioma"]
        assert rows == [["glioma", 8, 2], ["meningioma", 1, 9]]

    def test_raw_float_counts_rendered_as_int(self):
        matrix = np.array([[8.0, 2.0], [1.0, 9.0]])
        _, rows = confusion_matrix_to_table(matrix, ["a", "b"])

        assert rows[0][1:] == [8, 2]
        assert all(isinstance(v, int) for row in rows for v in row[1:])

    def test_value_format_applies_to_normalized_matrix(self):
        matrix = np.array([[0.8, 0.2], [0.1, 0.9]])
        _, rows = confusion_matrix_to_table(matrix, ["a", "b"], value_format=".0%")

        assert rows[0][1:] == ["80%", "20%"]

    def test_custom_corner_label(self):
        header, _ = confusion_matrix_to_table(
            np.array([[1, 0], [0, 1]]), ["a", "b"], corner_label="True \\ Pred",
        )
        assert header[0] == "True \\ Pred"

    def test_consistent_with_build_confusion_matrix_end_to_end(self):
        y_true = np.array(["a", "a", "b", "b", "b"])
        y_pred = np.array(["a", "b", "b", "b", "a"])
        result = build_confusion_matrix(y_true, y_pred, labels=["a", "b"])

        header, rows = confusion_matrix_to_table(result["matrix"], result["labels"])

        assert header == ["Истина \\ Предсказано", "a", "b"]
        assert rows[0][0] == "a" and rows[1][0] == "b"


class TestPlotConfusionMatrix:
    def test_returns_axes_with_expected_tick_labels(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

        matrix = np.array([[8, 2], [1, 9]])
        ax = plot_confusion_matrix(matrix, ["glioma", "meningioma"])

        assert [t.get_text() for t in ax.get_xticklabels()] == ["glioma", "meningioma"]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["glioma", "meningioma"]
        ax.figure.clf()

    def test_accepts_existing_axes(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        returned_ax = plot_confusion_matrix(np.array([[1, 0], [0, 1]]), ["a", "b"], ax=ax)

        assert returned_ax is ax
        plt.close(fig)

    def test_title_is_set_when_provided(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

        ax = plot_confusion_matrix(
            np.array([[1, 0], [0, 1]]), ["a", "b"], title="Матрица ошибок",
        )
        assert ax.get_title() == "Матрица ошибок"
        ax.figure.clf()


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
