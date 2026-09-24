"""Тесты models/multi_representation.py — гибридизация нескольких
векторных представлений одного объекта на уровне показателя сопряжённости
(статья 3, "Мультипредставительная гибридизация признакового пространства в
методе сопряжённости", theory/article_plans/03_multipredstavitelnaya_gibridizatsiya.txt).
"""

import numpy as np
import pytest
from sklearn.base import is_classifier

from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier
from subspace_conjugacy.models.multi_representation import (
    MultiRepresentationConjugacyClassifier,
)


_CLASSES = ["a", "b", "c"]
_N_FEATURES = 30
_BLOCK = 10
_K_TRUE = 4


def _block_diagonal_dataset(n_per_class, seed, bases):
    """Данные, "удобные" для критерия сопряжённости: у каждого класса
    объекты лежат (почти) в своём подпространстве через начало координат —
    ненулевые координаты только в "своём" блоке признаков, остальное —
    шум. Имитирует то, что подпространственный метод реально распознаёт
    хорошо (в отличие от изотропных гауссовых блобов со сдвигом среднего,
    которые критерий R(x, Y) не разделяет даже при большом сдвиге —
    R нормирован на x^T x и чувствителен к направлению, а не к смещению)."""
    r = np.random.default_rng(seed)
    parts = []
    for c_idx in range(len(_CLASSES)):
        X_c = np.zeros((n_per_class, _N_FEATURES))
        coeffs = r.standard_normal((n_per_class, _K_TRUE))
        X_c[:, c_idx * _BLOCK:(c_idx + 1) * _BLOCK] = coeffs @ bases[c_idx].T
        X_c += 0.02 * r.standard_normal(X_c.shape)
        X_c += 3.0
        parts.append(X_c)
    X = np.vstack(parts)
    y = np.repeat(_CLASSES, n_per_class)
    return X, y


@pytest.fixture
def three_class_two_representations():
    """3 разделимых класса, две НЕЗАВИСИМЫЕ "развёртки" (представления) —
    каждая представление своей отдельной блочно-диагональной структурой,
    как horizontal/vertical развёртка одного изображения несут разную, но
    обе информативную структуру."""
    bases_hor = [np.random.default_rng(10 + i).standard_normal((_BLOCK, _K_TRUE)) for i in range(3)]
    bases_ver = [np.random.default_rng(20 + i).standard_normal((_BLOCK, _K_TRUE)) for i in range(3)]

    X_hor_train, y_train = _block_diagonal_dataset(30, 500, bases_hor)
    X_ver_train, _ = _block_diagonal_dataset(30, 500, bases_ver)
    X_hor_test, y_test = _block_diagonal_dataset(15, 501, bases_hor)
    X_ver_test, _ = _block_diagonal_dataset(15, 501, bases_ver)
    return X_hor_train, X_ver_train, y_train, X_hor_test, X_ver_test, y_test


class _StubEstimator:
    """Обученная модель-заглушка с полным контролем над R-матрицей."""

    def __init__(self, classes, r_matrix):
        self.classes_ = np.array(classes)
        self.is_fitted_ = True
        self._r_matrix = np.asarray(r_matrix, dtype=np.float64)

    def predict_r_matrix(self, X):
        return self._r_matrix


@pytest.fixture
def stub_estimators():
    classes = ["a", "b", "c"]
    # Объект 0: оба представления согласны на классе "a" — все правила
    # слияния дают одно и то же решение. Объект 1: repr1 уверенно указывает
    # на "b" (0.9), а repr2 на этот же класс даёт 0.0 — mean/sum/geometric_mean
    # "разбавляют" эту уверенность до "a", а max (и weighted с перекосом в
    # сторону repr1) её сохраняют — так тест демонстрирует, что выбор
    # правила слияния действительно меняет итоговое решение.
    r_repr1 = np.array([[0.9, 0.1, 0.0], [0.5, 0.9, 0.0]])
    r_repr2 = np.array([[0.1, 0.8, 0.0], [0.5, 0.0, 0.0]])
    return {
        "repr1": _StubEstimator(classes, r_repr1),
        "repr2": _StubEstimator(classes, r_repr2),
    }


def _dummy_X(n_samples):
    return np.zeros((n_samples, 1))


class TestValidation:
    def test_fit_rejects_single_representation(self, three_class_two_representations):
        X_hor, _, y, *_ = three_class_two_representations
        clf = MultiRepresentationConjugacyClassifier(n_subclasses=2)
        with pytest.raises(ValueError, match="минимум 2"):
            clf.fit({"horizontal": X_hor}, y)

    def test_fit_rejects_mismatched_sample_counts(self, three_class_two_representations):
        X_hor, X_ver, y, *_ = three_class_two_representations
        clf = MultiRepresentationConjugacyClassifier(n_subclasses=2)
        with pytest.raises(ValueError, match="Число объектов"):
            clf.fit({"horizontal": X_hor, "vertical": X_ver[:-5]}, y)

    def test_fit_rejects_invalid_fusion(self, three_class_two_representations):
        X_hor, X_ver, y, *_ = three_class_two_representations
        clf = MultiRepresentationConjugacyClassifier(fusion="unknown", n_subclasses=2)
        with pytest.raises(ValueError, match="fusion"):
            clf.fit({"horizontal": X_hor, "vertical": X_ver}, y)

    def test_predict_before_fit_raises(self, three_class_two_representations):
        X_hor, X_ver, *_ = three_class_two_representations
        clf = MultiRepresentationConjugacyClassifier()
        with pytest.raises(RuntimeError, match="не обучен"):
            clf.predict({"horizontal": X_hor, "vertical": X_ver})

    def test_predict_missing_representation_raises(self, three_class_two_representations):
        X_hor, X_ver, y, X_hor_test, X_ver_test, _ = three_class_two_representations
        clf = MultiRepresentationConjugacyClassifier(n_subclasses=2).fit(
            {"horizontal": X_hor, "vertical": X_ver}, y
        )
        with pytest.raises(ValueError, match="vertical"):
            clf.predict({"horizontal": X_hor_test})

    def test_from_fitted_estimators_rejects_single_model(self, stub_estimators):
        with pytest.raises(ValueError, match="минимум 2"):
            MultiRepresentationConjugacyClassifier.from_fitted_estimators(
                {"repr1": stub_estimators["repr1"]}
            )

    def test_from_fitted_estimators_rejects_unfitted(self, stub_estimators):
        unfitted = _StubEstimator(["a", "b", "c"], [[0.1, 0.2, 0.3]])
        unfitted.is_fitted_ = False
        with pytest.raises(ValueError, match="не обучена"):
            MultiRepresentationConjugacyClassifier.from_fitted_estimators(
                {"repr1": stub_estimators["repr1"], "repr2": unfitted}
            )

    def test_from_fitted_estimators_rejects_mismatched_classes(self, stub_estimators):
        different_classes = _StubEstimator(["x", "y", "z"], [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        with pytest.raises(ValueError, match="одном и том же наборе классов"):
            MultiRepresentationConjugacyClassifier.from_fitted_estimators(
                {"repr1": stub_estimators["repr1"], "repr2": different_classes}
            )

    def test_weighted_fusion_missing_weight_raises(self, stub_estimators):
        clf = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
            stub_estimators, fusion="weighted", weights={"repr1": 1.0}
        )
        with pytest.raises(ValueError, match="repr2"):
            clf.predict_r_matrix({"repr1": _dummy_X(2), "repr2": _dummy_X(2)})


class TestFusionRules:
    def test_mean_fusion(self, stub_estimators):
        clf = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
            stub_estimators, fusion="mean"
        )
        R = clf.predict_r_matrix({"repr1": _dummy_X(2), "repr2": _dummy_X(2)})
        expected = np.array([[0.5, 0.45, 0.0], [0.5, 0.45, 0.0]])
        np.testing.assert_allclose(R, expected)

    def test_sum_fusion(self, stub_estimators):
        clf = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
            stub_estimators, fusion="sum"
        )
        R = clf.predict_r_matrix({"repr1": _dummy_X(2), "repr2": _dummy_X(2)})
        expected = np.array([[1.0, 0.9, 0.0], [1.0, 0.9, 0.0]])
        np.testing.assert_allclose(R, expected)

    def test_max_fusion(self, stub_estimators):
        clf = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
            stub_estimators, fusion="max"
        )
        R = clf.predict_r_matrix({"repr1": _dummy_X(2), "repr2": _dummy_X(2)})
        expected = np.array([[0.9, 0.8, 0.0], [0.5, 0.9, 0.0]])
        np.testing.assert_allclose(R, expected)

    def test_weighted_fusion(self, stub_estimators):
        clf = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
            stub_estimators, fusion="weighted", weights={"repr1": 3.0, "repr2": 1.0}
        )
        R = clf.predict_r_matrix({"repr1": _dummy_X(2), "repr2": _dummy_X(2)})
        expected = np.array([[0.7, 0.275, 0.0], [0.5, 0.675, 0.0]])
        np.testing.assert_allclose(R, expected, atol=1e-9)

    def test_weighted_without_weights_matches_mean(self, stub_estimators):
        clf = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
            stub_estimators, fusion="weighted", weights=None
        )
        R = clf.predict_r_matrix({"repr1": _dummy_X(2), "repr2": _dummy_X(2)})
        expected = np.array([[0.5, 0.45, 0.0], [0.5, 0.45, 0.0]])
        np.testing.assert_allclose(R, expected)

    def test_geometric_mean_fusion(self, stub_estimators):
        clf = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
            stub_estimators, fusion="geometric_mean"
        )
        R = clf.predict_r_matrix({"repr1": _dummy_X(2), "repr2": _dummy_X(2)})
        # Ненулевые записи считаются точно; записи, где хотя бы одно
        # представление даёт ровно 0, после eps-клиппинга остаются очень
        # маленькими (но не обязаны быть математически точным нулём).
        np.testing.assert_allclose(R[0, 0], np.sqrt(0.9 * 0.1), atol=1e-6)
        np.testing.assert_allclose(R[0, 1], np.sqrt(0.1 * 0.8), atol=1e-6)
        np.testing.assert_allclose(R[1, 0], np.sqrt(0.5 * 0.5), atol=1e-6)
        assert R[0, 2] < 1e-5
        assert R[1, 1] < 1e-5
        assert R[1, 2] < 1e-5

    def test_fusion_changes_decision(self, stub_estimators):
        """max сохраняет уверенность repr1 в классе "b" для объекта 1, а
        mean её "разбавляет" нулём repr2 и решает в пользу класса "a"."""
        mean_clf = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
            stub_estimators, fusion="mean"
        )
        max_clf = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
            stub_estimators, fusion="max"
        )
        X = {"repr1": _dummy_X(2), "repr2": _dummy_X(2)}
        mean_pred = mean_clf.predict(X)
        max_pred = max_clf.predict(X)
        assert mean_pred[1] == "a"
        assert max_pred[1] == "b"

    def test_predict_r_matrix_by_representation_passthrough(self, stub_estimators):
        clf = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
            stub_estimators, fusion="mean"
        )
        r_by_repr = clf.predict_r_matrix_by_representation(
            {"repr1": _dummy_X(2), "repr2": _dummy_X(2)}
        )
        np.testing.assert_allclose(r_by_repr["repr1"], stub_estimators["repr1"]._r_matrix)
        np.testing.assert_allclose(r_by_repr["repr2"], stub_estimators["repr2"]._r_matrix)


class TestEndToEnd:
    def test_fit_predict_on_separable_data(self, three_class_two_representations):
        X_hor_train, X_ver_train, y_train, X_hor_test, X_ver_test, y_test = (
            three_class_two_representations
        )
        clf = MultiRepresentationConjugacyClassifier(n_subclasses=2, fusion="mean")
        clf.fit({"horizontal": X_hor_train, "vertical": X_ver_train}, y_train)

        assert clf.representations_ == ["horizontal", "vertical"]
        assert set(clf.classes_.tolist()) == {"a", "b", "c"}
        assert isinstance(
            clf.estimators_by_representation_["horizontal"], SubspaceConjugacyClassifier
        )

        y_pred = clf.predict({"horizontal": X_hor_test, "vertical": X_ver_test})
        accuracy = float(np.mean(y_pred == y_test))
        assert accuracy > 0.8

    def test_predict_proba_rows_sum_to_one(self, three_class_two_representations):
        X_hor_train, X_ver_train, y_train, X_hor_test, X_ver_test, _ = (
            three_class_two_representations
        )
        clf = MultiRepresentationConjugacyClassifier(n_subclasses=2).fit(
            {"horizontal": X_hor_train, "vertical": X_ver_train}, y_train
        )
        proba = clf.predict_proba({"horizontal": X_hor_test, "vertical": X_ver_test})
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(proba.shape[0]), atol=1e-8)

    def test_predict_r_matrix_by_representation_shapes(self, three_class_two_representations):
        X_hor_train, X_ver_train, y_train, X_hor_test, X_ver_test, _ = (
            three_class_two_representations
        )
        clf = MultiRepresentationConjugacyClassifier(n_subclasses=2).fit(
            {"horizontal": X_hor_train, "vertical": X_ver_train}, y_train
        )
        r_by_repr = clf.predict_r_matrix_by_representation(
            {"horizontal": X_hor_test, "vertical": X_ver_test}
        )
        assert r_by_repr["horizontal"].shape == (X_hor_test.shape[0], 3)
        assert r_by_repr["vertical"].shape == (X_ver_test.shape[0], 3)

    def test_is_classifier(self, three_class_two_representations):
        clf = MultiRepresentationConjugacyClassifier(n_subclasses=2)
        assert is_classifier(clf)
