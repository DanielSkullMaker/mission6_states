"""Тесты utils/validation.py.

Раньше эти проверки жили только в блоке ``if __name__ == "__main__"`` самого
модуля (Фаза 0 плана рефакторинга требует перенести такие блоки в tests/).
"""

import numpy as np
import pytest

from subspace_conjugacy.utils.validation import (
    check_array_X,
    check_basis_matrix,
    check_hyperparameters,
    check_is_fitted,
    check_X_y,
)


class TestCheckArrayX:
    def test_accepts_list_and_returns_2d_array(self):
        X = check_array_X([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert isinstance(X, np.ndarray)
        assert X.shape == (2, 3)
        assert X.dtype == np.float64

    def test_1d_input_reshaped_to_row_when_accept_1d(self):
        X = check_array_X([1.0, 2.0, 3.0], accept_1d=True)
        assert X.shape == (1, 3)

    def test_1d_input_rejected_when_accept_1d_false(self):
        with pytest.raises(ValueError):
            check_array_X([1.0, 2.0, 3.0], accept_1d=False)

    def test_empty_array_raises(self):
        with pytest.raises(ValueError):
            check_array_X(np.array([]))

    def test_wrong_dimensionality_raises(self):
        with pytest.raises(ValueError):
            check_array_X(np.zeros((2, 2, 2)))

    def test_nan_rejected_by_default(self):
        with pytest.raises(ValueError):
            check_array_X([[1.0, np.nan], [3.0, 4.0]])

    def test_nan_allowed_when_flag_set(self):
        X = check_array_X([[1.0, np.nan]], allow_nan=True)
        assert np.isnan(X).any()

    def test_inf_rejected_by_default(self):
        with pytest.raises(ValueError):
            check_array_X([[1.0, np.inf]])

    def test_inf_allowed_when_flag_set(self):
        X = check_array_X([[1.0, np.inf]], allow_inf=True)
        assert np.isinf(X).any()

    def test_uncastable_input_raises_type_error(self):
        class NotCastable:
            def __array__(self, *args, **kwargs):
                raise RuntimeError("cannot cast")

        with pytest.raises(TypeError):
            check_array_X(NotCastable())


class TestCheckXY:
    def test_valid_pair_passes_through(self):
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, size=10)
        X_out, y_out = check_X_y(X, y)
        assert X_out.shape == (10, 5)
        assert y_out.shape == (10,)

    def test_mismatched_lengths_raise(self):
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, size=7)
        with pytest.raises(ValueError):
            check_X_y(X, y)

    def test_column_vector_y_is_squeezed_to_1d(self):
        X = np.random.randn(5, 3)
        y = np.random.randint(0, 2, size=(5, 1))
        X_out, y_out = check_X_y(X, y)
        assert y_out.ndim == 1
        assert y_out.shape[0] == 5

    def test_y_that_cannot_be_squeezed_to_1d_raises(self):
        X = np.random.randn(4, 3)
        y = np.random.randint(0, 2, size=(4, 2))
        with pytest.raises(ValueError):
            check_X_y(X, y)


class TestCheckBasisMatrix:
    def test_valid_basis_passes_through(self):
        Y = np.random.randn(32, 4)
        Y_out = check_basis_matrix(Y, expected_n_features=32)
        assert Y_out.shape == (32, 4)

    def test_wrong_n_features_raises(self):
        Y = np.random.randn(32, 4)
        with pytest.raises(ValueError):
            check_basis_matrix(Y, expected_n_features=64)

    def test_zero_column_raises(self):
        Y = np.random.randn(32, 4)
        Y[:, 0] = 0.0
        with pytest.raises(ValueError):
            check_basis_matrix(Y, expected_n_features=32)

    def test_transposed_row_vector_is_corrected(self):
        # (1, N) вектор при известном expected_n_features -> транспонируется в (N, 1).
        y_row = np.random.randn(1, 32)
        Y_out = check_basis_matrix(y_row, expected_n_features=32)
        assert Y_out.shape == (32, 1)

    def test_high_condition_number_does_not_raise(self):
        # Почти коллинеарные столбцы -> большое число обусловленности,
        # но защита деления на ноль делегирована регуляризации, а не check_basis_matrix.
        rng = np.random.default_rng(3)
        base = rng.standard_normal(16)
        Y = np.column_stack([base, base * (1 + 1e-10)])
        Y_out = check_basis_matrix(Y, max_condition_number=1e6)
        assert Y_out.shape == (16, 2)


class TestCheckHyperparameters:
    def test_valid_hyperparameters_pass(self):
        check_hyperparameters(n_subclasses=8, reg_param=1e-8, n_centers_init=2)

    def test_negative_n_subclasses_raises(self):
        with pytest.raises(ValueError):
            check_hyperparameters(n_subclasses=-3, reg_param=1e-8)

    def test_zero_n_subclasses_raises(self):
        with pytest.raises(ValueError):
            check_hyperparameters(n_subclasses=0, reg_param=1e-8)

    def test_non_integer_n_subclasses_raises(self):
        with pytest.raises(ValueError):
            check_hyperparameters(n_subclasses=8.5, reg_param=1e-8)

    def test_non_positive_reg_param_raises(self):
        with pytest.raises(ValueError):
            check_hyperparameters(n_subclasses=8, reg_param=0.0)

    def test_invalid_n_centers_init_raises(self):
        with pytest.raises(ValueError):
            check_hyperparameters(n_subclasses=8, reg_param=1e-8, n_centers_init=0)

    def test_n_centers_init_optional(self):
        check_hyperparameters(n_subclasses=8, reg_param=1e-8, n_centers_init=None)


class TestCheckIsFitted:
    class _MockEstimator:
        def __init__(self, fitted: bool = False):
            self.is_fitted_ = fitted

    def test_unfitted_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            check_is_fitted(self._MockEstimator(fitted=False))

    def test_fitted_passes_silently(self):
        check_is_fitted(self._MockEstimator(fitted=True))

    def test_multiple_attributes_all_required(self):
        class MultiAttr:
            is_fitted_ = True
            has_centers_ = False

        with pytest.raises(RuntimeError):
            check_is_fitted(MultiAttr(), attributes=["is_fitted_", "has_centers_"])

    def test_multiple_attributes_all_present_passes(self):
        class MultiAttr:
            is_fitted_ = True
            has_centers_ = True

        check_is_fitted(MultiAttr(), attributes=["is_fitted_", "has_centers_"])
