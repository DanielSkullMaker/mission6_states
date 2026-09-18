"""Тесты SubspaceConjugacyClassifier (Фаза C теории, NB8).

Все тесты используют синтетические данные (fixtures из tests/conftest.py) —
это theory-тесты, проверяющие инварианты Фазы C независимо от конкретного
датасета. Проверка на реальных МРТ-данных — в tests/test_parity/.
"""

import numpy as np
import pytest

from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier

N_SUBCLASSES = 4


@pytest.fixture
def fitted_classifier(three_class_dataset) -> SubspaceConjugacyClassifier:
    X, y = three_class_dataset
    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at=2)
    clf.fit(X, y)
    return clf


@pytest.mark.theory
class TestFitUsesCanonicalClusterer:
    """Классификатор обязан использовать FursovClusterer (канон), не legacy."""

    def test_fit_produces_freeze_basis_at_2_by_default(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
        clf.fit(X, y)

        for cls in clf.classes_:
            for Y in clf.subspaces_[cls]:
                assert Y.shape == (X.shape[1], 2)

    def test_fit_matches_standalone_fursov_clusterer(self, three_class_dataset):
        """subspaces_[cls] должны совпадать с независимым FursovClusterer.fit()."""
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(
            n_subclasses=N_SUBCLASSES, freeze_basis_at=2, reg_param=1e-8
        )
        clf.fit(X, y)

        for cls in clf.classes_:
            X_cls = X[y == cls]
            reference = FursovClusterer(
                n_subclasses=N_SUBCLASSES, freeze_basis_at=2, reg_param=1e-8
            )
            reference.fit(X_cls)

            for Y_clf, Y_ref in zip(clf.subspaces_[cls], reference.subspaces_):
                np.testing.assert_array_equal(Y_clf, Y_ref)

    def test_freeze_basis_at_is_configurable(self, three_class_dataset):
        """freeze_basis_at ограничивает МАКСИМУМ размера базиса, а не задаёт
        его фиксированно для всех подклассов: B.2 — жадный argmax R, поэтому
        подклассы растут неравномерно (см. ConjugacyClusterGrowth) — часть
        подклассов может остаться с исходной парой (k=2), если ни один
        оставшийся вектор не набрал по ним максимум R ни на одной итерации.
        """
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at=3)
        clf.fit(X, y)

        for cls in clf.classes_:
            basis_sizes = [Y.shape[1] for Y in clf.subspaces_[cls]]
            assert all(2 <= size <= 3 for size in basis_sizes)
            assert max(basis_sizes) == 3  # freeze_basis_at реально применяется


@pytest.mark.theory
class TestPhaseCFlatArgmax:
    """Теория C: R_{c,s} для 24 (n_classes*n_subclasses) базисов, flat argmax."""

    def test_flat_subclass_labels_length(self, fitted_classifier):
        clf = fitted_classifier
        n_classes = len(clf.classes_)
        assert len(clf.flat_subclass_labels_) == n_classes * N_SUBCLASSES

    def test_r_matrix_flat_shape(self, fitted_classifier, three_class_dataset):
        X, _ = three_class_dataset
        clf = fitted_classifier
        R_flat = clf.predict_r_matrix_flat(X)

        assert R_flat.shape == (X.shape[0], len(clf.flat_subclass_labels_))
        assert np.all(R_flat >= 0.0) and np.all(R_flat <= 1.0)

    def test_r_matrix_per_class_equals_max_of_flat(self, fitted_classifier, three_class_dataset):
        """predict_r_matrix (max по подклассам класса) должен совпасть с
        поклассовым максимумом predict_r_matrix_flat — это одна и та же
        Фаза C, посчитанная двумя разными агрегациями."""
        X, _ = three_class_dataset
        clf = fitted_classifier

        R_per_class = clf.predict_r_matrix(X)
        R_flat = clf.predict_r_matrix_flat(X)

        for cls_idx, cls in enumerate(clf.classes_):
            mask = clf.flat_subclass_labels_ == cls
            expected = np.max(R_flat[:, mask], axis=1)
            np.testing.assert_allclose(R_per_class[:, cls_idx], expected)

    def test_predict_subclass_owner_matches_predict(self, fitted_classifier, three_class_dataset):
        """Класс-владелец лучшего плоского подкласса == predict().

        Теория C: subclass* = argmax_{c,s} R_{c,s}; class* = c(subclass*).
        Это ДОЛЖНО совпадать с classifier.predict(), который берёт
        argmax по максимумам R внутри каждого класса — это два
        математически эквивалентных способа выразить один и тот же
        плоский argmax (argmax группового максимума == группа глобального
        максимума).
        """
        X, _ = three_class_dataset
        clf = fitted_classifier

        flat_idx = clf.predict_subclass(X)
        owner_classes = clf.flat_subclass_labels_[flat_idx]
        predicted_classes = clf.predict(X)

        np.testing.assert_array_equal(owner_classes, predicted_classes)

    def test_predict_subclass_range(self, fitted_classifier, three_class_dataset):
        X, _ = three_class_dataset
        clf = fitted_classifier
        flat_idx = clf.predict_subclass(X)

        assert np.all(flat_idx >= 0)
        assert np.all(flat_idx < len(clf.flat_subclass_labels_))


@pytest.mark.theory
class TestConfidenceRatio:
    """NB8: proportion = best_R / mean(R_others) - 1."""

    def test_confidence_ratio_matches_manual_formula(self, fitted_classifier, three_class_dataset):
        X, _ = three_class_dataset
        clf = fitted_classifier

        R_flat = clf.predict_r_matrix_flat(X)
        confidence = clf.predict_confidence_ratio(X)

        best = np.max(R_flat, axis=1)
        mean_others = (np.sum(R_flat, axis=1) - best) / (R_flat.shape[1] - 1)
        expected = best / mean_others - 1

        np.testing.assert_allclose(confidence, expected, rtol=1e-9)

    def test_confidence_ratio_shape(self, fitted_classifier, three_class_dataset):
        X, _ = three_class_dataset
        confidence = fitted_classifier.predict_confidence_ratio(X)
        assert confidence.shape == (X.shape[0],)

    def test_confidence_ratio_requires_multiple_subspaces(self):
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
        clf.fit_from_subclass_bases({"only_class": [np.random.randn(16, 2)]})

        with pytest.raises(ValueError):
            clf.predict_confidence_ratio(np.random.randn(3, 16))


class TestFitFromSubclassBases:
    """fit_from_subclass_bases: сборка классификатора без повторной кластеризации."""

    def test_roundtrip_matches_fit(self, three_class_dataset):
        X, y = three_class_dataset

        clf_fit = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at=2)
        clf_fit.fit(X, y)

        clf_loaded = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
        clf_loaded.fit_from_subclass_bases(clf_fit.subspaces_)

        assert clf_loaded.is_fitted_
        assert set(clf_loaded.classes_) == set(clf_fit.classes_)
        np.testing.assert_array_equal(
            clf_fit.predict(X), clf_loaded.predict(X)
        )
        np.testing.assert_allclose(
            clf_fit.predict_r_matrix(X), clf_loaded.predict_r_matrix(X)
        )

    def test_rejects_empty_dict(self):
        clf = SubspaceConjugacyClassifier()
        with pytest.raises(ValueError):
            clf.fit_from_subclass_bases({})

    def test_rejects_mismatched_feature_dimension(self):
        clf = SubspaceConjugacyClassifier()
        with pytest.raises(ValueError):
            clf.fit_from_subclass_bases({
                "a": [np.random.randn(16, 2)],
                "b": [np.random.randn(32, 2)],
            })

    def test_sets_n_features_in(self):
        clf = SubspaceConjugacyClassifier()
        clf.fit_from_subclass_bases({
            "a": [np.random.randn(16, 2), np.random.randn(16, 2)],
            "b": [np.random.randn(16, 2), np.random.randn(16, 2)],
        })
        assert clf.n_features_in_ == 16


class TestPredictProba:
    def test_probabilities_sum_to_one(self, fitted_classifier, three_class_dataset):
        X, _ = three_class_dataset
        probs = fitted_classifier.predict_proba(X)

        assert probs.shape == (X.shape[0], len(fitted_classifier.classes_))
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-9)
        assert np.all(probs >= 0.0)


class TestValidationErrors:
    def test_fit_requires_at_least_two_classes(self):
        clf = SubspaceConjugacyClassifier(n_subclasses=2)
        X = np.random.randn(20, 16)
        y = np.full(20, "only_one_class")

        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_fit_requires_enough_samples_per_class(self):
        clf = SubspaceConjugacyClassifier(n_subclasses=8)
        X = np.random.randn(10, 16)
        y = np.array(["a"] * 5 + ["b"] * 5)

        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_predict_before_fit_raises(self):
        clf = SubspaceConjugacyClassifier(n_subclasses=4)
        with pytest.raises(RuntimeError):
            clf.predict(np.random.randn(3, 16))

    def test_predict_subclass_before_fit_raises(self):
        clf = SubspaceConjugacyClassifier(n_subclasses=4)
        with pytest.raises(RuntimeError):
            clf.predict_subclass(np.random.randn(3, 16))


@pytest.mark.theory
class TestClassificationAccuracyOnSeparatedClusters:
    """С хорошо разделёнными классами classifier должен предсказывать точно.

    Не количественный порог accuracy (это делает test_parity на реальных
    данных) — здесь просто sanity-check, что вся Фаза C работает end-to-end
    и не деградирует до случайного угадывания на тривиально разделимых
    данных.
    """

    def test_high_accuracy_on_well_separated_classes(self, random_seed):
        """Классы должны отличаться НАПРАВЛЕНИЕМ, а не только величиной сдвига.

        R(x, Y) масштабно-инвариантен (см. test_r_is_scale_invariant в
        test_clustering_theory.py): R(alpha*x, Y) == R(x, Y). Поэтому просто
        добавить разным классам разный константный сдвиг вдоль ОДНОГО и того
        же направления не создаёт разделимости для этого метода — нужны
        разные направления в пространстве признаков.
        """
        np.random.seed(random_seed)
        n_per_class = 40
        n_features = 32
        n_classes = 3

        class_directions = np.random.randn(n_classes, n_features)
        class_directions /= np.linalg.norm(class_directions, axis=1, keepdims=True)

        X_list, y_list = [], []
        for idx in range(n_classes):
            magnitudes = np.random.uniform(8.0, 12.0, size=(n_per_class, 1))
            noise = np.random.randn(n_per_class, n_features) * 0.5
            X_list.append(magnitudes * class_directions[idx] + noise)
            y_list.append(np.full(n_per_class, idx))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        clf = SubspaceConjugacyClassifier(n_subclasses=4, freeze_basis_at=2)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        accuracy = np.mean(y_pred == y)

        assert accuracy >= 0.9
