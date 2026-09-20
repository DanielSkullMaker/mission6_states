"""Тесты models/sequential_classifier.py — SequentialClassifier.

Двухэтапная последовательная классификация (Korshikov & Fursov: "first
determine the projection of the image, and then to determine the presence
of a tumor and its specific type" — refactoring_plan.txt, раздел 10,
находка №4). Мотивирующий сценарий — проекция -> тип опухоли, но
SequentialClassifier сам по себе универсален (composition поверх любых
sklearn-совместимых классификаторов), поэтому тесты используют простые
синтетические группы, не привязанные к конкретному домену.
"""

import numpy as np
import pytest

from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier
from subspace_conjugacy.models.sequential_classifier import SequentialClassifier


def _make_group(rng, offset, n, n_features):
    return rng.standard_normal((n, n_features)) + offset


@pytest.fixture
def two_stage_data():
    """Этап 1: 2 группы ("a", "b") по X1 (64 признака).
    Этап 2: внутри каждой группы — 2 подкласса ("x", "y") по X2 (32 признака).
    """
    rng = np.random.default_rng(0)
    n_per_group = 20

    X1 = np.vstack([
        _make_group(rng, 0.0, n_per_group, 64),
        _make_group(rng, 6.0, n_per_group, 64),
    ])
    y1 = np.array(["a"] * n_per_group + ["b"] * n_per_group)

    X2_by_group = {
        "a": np.vstack([
            _make_group(rng, 0.0, n_per_group // 2, 32),
            _make_group(rng, 6.0, n_per_group // 2, 32),
        ]),
        "b": np.vstack([
            _make_group(rng, 0.0, n_per_group // 2, 32),
            _make_group(rng, 6.0, n_per_group // 2, 32),
        ]),
    }
    y2_by_group = {
        "a": np.array(["x"] * (n_per_group // 2) + ["y"] * (n_per_group // 2)),
        "b": np.array(["x"] * (n_per_group // 2) + ["y"] * (n_per_group // 2)),
    }
    return X1, y1, X2_by_group, y2_by_group


def _make_classifier():
    return SubspaceConjugacyClassifier(n_subclasses=4)


class TestSequentialClassifierConstruction:
    def test_empty_stage2_raises(self):
        with pytest.raises(ValueError):
            SequentialClassifier(_make_classifier(), {})

    def test_not_fitted_initially(self):
        seq = SequentialClassifier(
            _make_classifier(), {"a": _make_classifier(), "b": _make_classifier()}
        )
        assert seq.is_fitted_ is False

    def test_prefit_classifiers_report_fitted_without_calling_fit(self, two_stage_data):
        """Если stage1/stage2 переданы УЖЕ обученными, is_fitted_ должен
        сразу вернуть True, не требуя вызова SequentialClassifier.fit()."""
        X1, y1, X2_by_group, y2_by_group = two_stage_data

        stage1 = _make_classifier()
        stage1.fit(X1, y1)
        stage2 = {}
        for label in ["a", "b"]:
            clf = _make_classifier()
            clf.fit(X2_by_group[label], y2_by_group[label])
            stage2[label] = clf

        seq = SequentialClassifier(stage1, stage2)
        assert seq.is_fitted_ is True


class TestSequentialClassifierFit:
    def test_fit_trains_stage1_and_all_stage2(self, two_stage_data):
        X1, y1, X2_by_group, y2_by_group = two_stage_data
        seq = SequentialClassifier(
            _make_classifier(), {"a": _make_classifier(), "b": _make_classifier()}
        )
        seq.fit(X1, y1, X2_by_group, y2_by_group)

        assert seq.is_fitted_ is True
        assert seq.stage1_classifier.is_fitted_
        assert seq.stage2_classifiers["a"].is_fitted_
        assert seq.stage2_classifiers["b"].is_fitted_

    def test_missing_stage2_classifier_raises(self, two_stage_data):
        X1, y1, X2_by_group, y2_by_group = two_stage_data
        seq = SequentialClassifier(_make_classifier(), {"a": _make_classifier()})

        with pytest.raises(ValueError, match="b"):
            seq.fit(X1, y1, X2_by_group, y2_by_group)

    def test_missing_stage2_data_raises(self, two_stage_data):
        X1, y1, X2_by_group, y2_by_group = two_stage_data
        del X2_by_group["b"]
        seq = SequentialClassifier(
            _make_classifier(), {"a": _make_classifier(), "b": _make_classifier()}
        )

        with pytest.raises(ValueError, match="b"):
            seq.fit(X1, y1, X2_by_group, y2_by_group)


class TestSequentialClassifierPredict:
    def test_predict_returns_both_stage_labels(self, two_stage_data):
        X1, y1, X2_by_group, y2_by_group = two_stage_data
        seq = SequentialClassifier(
            _make_classifier(), {"a": _make_classifier(), "b": _make_classifier()}
        )
        seq.fit(X1, y1, X2_by_group, y2_by_group)

        rng = np.random.default_rng(1)
        X1_test = np.vstack([_make_group(rng, 0.0, 3, 64), _make_group(rng, 6.0, 3, 64)])
        X2_test = np.vstack([_make_group(rng, 0.0, 3, 32), _make_group(rng, 6.0, 3, 32)])

        stage1_pred, stage2_pred = seq.predict(X1_test, X2_test)

        assert len(stage1_pred) == 6
        assert len(stage2_pred) == 6
        assert set(stage1_pred).issubset({"a", "b"})
        assert set(stage2_pred).issubset({"x", "y"})

    def test_predict_stage2_uses_predicted_stage1_labels(self, two_stage_data):
        """predict() должен передавать ПРЕДСКАЗАННЫЕ (не истинные) метки
        этапа 1 в этап 2 — так реализовано "результат этапа 1 становится
        данными для этапа 2" из статьи."""
        X1, y1, X2_by_group, y2_by_group = two_stage_data
        seq = SequentialClassifier(
            _make_classifier(), {"a": _make_classifier(), "b": _make_classifier()}
        )
        seq.fit(X1, y1, X2_by_group, y2_by_group)

        rng = np.random.default_rng(2)
        X1_test = _make_group(rng, 0.0, 5, 64)
        X2_test = _make_group(rng, 0.0, 5, 32)

        stage1_pred = seq.predict_stage1(X1_test)
        expected_stage2 = seq.predict_stage2(X2_test, stage1_pred)
        _, actual_stage2 = seq.predict(X1_test, X2_test)

        np.testing.assert_array_equal(actual_stage2, expected_stage2)

    def test_predict_stage2_can_use_true_labels_for_isolated_evaluation(self, two_stage_data):
        """Позволяет оценить качество этапа 2 отдельно от ошибок этапа 1,
        передав истинные (а не предсказанные) метки этапа 1."""
        X1, y1, X2_by_group, y2_by_group = two_stage_data
        seq = SequentialClassifier(
            _make_classifier(), {"a": _make_classifier(), "b": _make_classifier()}
        )
        seq.fit(X1, y1, X2_by_group, y2_by_group)

        true_stage1_labels = np.array(["a", "a", "b", "b"])
        X2_test = np.vstack([
            X2_by_group["a"][:2], X2_by_group["b"][:2],
        ])

        stage2_pred = seq.predict_stage2(X2_test, true_stage1_labels)
        assert len(stage2_pred) == 4

    def test_predict_stage2_unknown_label_raises(self, two_stage_data):
        X1, y1, X2_by_group, y2_by_group = two_stage_data
        seq = SequentialClassifier(
            _make_classifier(), {"a": _make_classifier(), "b": _make_classifier()}
        )
        seq.fit(X1, y1, X2_by_group, y2_by_group)

        with pytest.raises(ValueError):
            seq.predict_stage2(np.random.randn(3, 32), np.array(["a", "c", "a"]))

    def test_predict_before_fit_raises(self):
        seq = SequentialClassifier(
            _make_classifier(), {"a": _make_classifier(), "b": _make_classifier()}
        )
        with pytest.raises(RuntimeError):
            seq.predict(np.random.randn(3, 64), np.random.randn(3, 32))


class TestSequentialClassifierRepr:
    def test_repr_contains_key_info(self, two_stage_data):
        X1, y1, X2_by_group, y2_by_group = two_stage_data
        seq = SequentialClassifier(
            _make_classifier(), {"a": _make_classifier(), "b": _make_classifier()}
        )

        assert "fitted=False" in repr(seq)

        seq.fit(X1, y1, X2_by_group, y2_by_group)
        assert "fitted=True" in repr(seq)
        assert "SubspaceConjugacyClassifier" in repr(seq)
