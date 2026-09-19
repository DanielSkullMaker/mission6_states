"""Тесты models/base.py — BaseSubspaceEstimator (общий sklearn-интерфейс).

Раньше эти проверки жили только в блоке ``if __name__ == "__main__"`` самого
модуля (Фаза 0 плана рефакторинга требует перенести такие блоки в tests/).

BaseSubspaceEstimator абстрактный, поэтому тесты используют минимальную
конкретную реализацию (``_DummyEstimator``), как и демо-блок оригинала.
"""

from typing import Optional

import numpy as np
import pytest

from subspace_conjugacy.models.base import BaseSubspaceEstimator


class _DummyEstimator(BaseSubspaceEstimator):
    """Минимальная конкретная реализация для тестирования базового класса."""

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "_DummyEstimator":
        self._validate_input_params()
        X_clean, y_clean = self._validate_data(X, y)
        self.n_features_in_ = X_clean.shape[1]
        self.is_fitted_ = True
        return self

    def predict_r_matrix(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X_clean, _ = self._validate_data(X)
        return np.ones((X_clean.shape[0], self.n_subclasses))


class TestBaseSubspaceEstimatorInit:
    def test_default_hyperparameters(self):
        est = _DummyEstimator()
        assert est.n_subclasses == 8
        assert est.n_centers_init == 2
        assert est.reg_param == 1e-8
        assert est.is_fitted_ is False
        assert est.n_features_in_ is None

    def test_custom_hyperparameters(self):
        est = _DummyEstimator(n_subclasses=4, n_centers_init=3, reg_param=1e-6)
        assert est.n_subclasses == 4
        assert est.n_centers_init == 3
        assert est.reg_param == 1e-6


class TestValidateInputParams:
    def test_negative_n_subclasses_raises_on_fit(self):
        est = _DummyEstimator(n_subclasses=-1)
        with pytest.raises(ValueError):
            est.fit(np.random.randn(10, 5))

    def test_zero_n_centers_init_raises(self):
        est = _DummyEstimator(n_centers_init=0)
        with pytest.raises(ValueError):
            est.fit(np.random.randn(10, 5))

    def test_non_positive_reg_param_raises(self):
        est = _DummyEstimator(reg_param=0.0)
        with pytest.raises(ValueError):
            est.fit(np.random.randn(10, 5))


class TestValidateData:
    def test_fit_sets_n_features_in(self):
        est = _DummyEstimator(n_subclasses=4)
        X = np.random.randn(20, 16)
        y = np.random.randint(0, 2, size=20)
        est.fit(X, y)
        assert est.is_fitted_ is True
        assert est.n_features_in_ == 16

    def test_1d_input_reshaped_to_row(self):
        est = _DummyEstimator()
        est.fit(np.random.randn(1, 16))
        assert est.n_features_in_ == 16

    def test_empty_input_raises(self):
        est = _DummyEstimator()
        with pytest.raises(ValueError):
            est.fit(np.array([]).reshape(0, 5))

    def test_wrong_ndim_raises(self):
        est = _DummyEstimator()
        with pytest.raises(ValueError):
            est.fit(np.random.randn(3, 3, 3))

    def test_mismatched_y_length_raises(self):
        est = _DummyEstimator()
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, size=7)
        with pytest.raises(ValueError):
            est.fit(X, y)

    def test_predict_with_mismatched_feature_count_raises_after_fit(self):
        est = _DummyEstimator(n_subclasses=4)
        est.fit(np.random.randn(20, 16))

        with pytest.raises(ValueError):
            est.predict_r_matrix(np.random.randn(5, 8))

    def test_predict_with_matching_feature_count_succeeds(self):
        est = _DummyEstimator(n_subclasses=4)
        est.fit(np.random.randn(20, 16))

        R = est.predict_r_matrix(np.random.randn(5, 16))
        assert R.shape == (5, 4)


class TestCheckIsFitted:
    def test_predict_before_fit_raises_runtime_error(self):
        est = _DummyEstimator()
        with pytest.raises(RuntimeError):
            est.predict_r_matrix(np.random.randn(5, 10))

    def test_predict_after_fit_succeeds(self):
        est = _DummyEstimator(n_subclasses=3)
        est.fit(np.random.randn(10, 4))
        R = est.predict_r_matrix(np.random.randn(2, 4))
        assert R.shape == (2, 3)


def test_base_estimator_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseSubspaceEstimator()
