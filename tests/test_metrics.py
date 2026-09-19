"""Тесты core/metrics.py — математическое ядро теории (R(x,Y), cosine).

Раньше эти проверки жили только в блоке ``if __name__ == "__main__"`` самого
модуля (Фаза 0 плана рефакторинга требует перенести такие блоки в tests/).
Здесь та же логика оформлена как настоящие pytest-тесты, плюс несколько
дополнительных краевых случаев, которых не было в демо-блоке.
"""

import numpy as np
import pytest

from subspace_conjugacy.core.metrics import (
    compute_gram_inverse,
    conjugate_criterion,
    cosine_similarity_matrix,
)


class TestComputeGramInverse:
    def test_returns_correct_shape(self):
        Y = np.random.randn(64, 3)
        inv_gram = compute_gram_inverse(Y)
        assert inv_gram.shape == (3, 3)

    def test_inverse_of_orthonormal_basis_is_identity(self):
        Y = np.eye(10)[:, :3]
        inv_gram = compute_gram_inverse(Y, reg_param=0.0)
        np.testing.assert_allclose(inv_gram, np.eye(3), atol=1e-10)

    def test_falls_back_to_pinv_on_singular_gram(self):
        # Два одинаковых столбца -> вырожденная матрица Грама.
        Y = np.column_stack([np.ones(10), np.ones(10)])
        inv_gram = compute_gram_inverse(Y, reg_param=0.0)
        assert np.all(np.isfinite(inv_gram))

    def test_regularization_prevents_blowup_on_near_singular_basis(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal(32)
        Y = np.column_stack([base, base * 2.0 + 1e-12])
        inv_gram = compute_gram_inverse(Y, reg_param=1e-6)
        assert np.all(np.isfinite(inv_gram))


class TestConjugateCriterion:
    def test_single_vector_returns_float_in_unit_range(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(128)
        Y = rng.standard_normal((128, 4))

        r = conjugate_criterion(x, Y)

        assert isinstance(r, float)
        assert 0.0 <= r <= 1.0

    def test_batch_returns_array_of_correct_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 128))
        Y = rng.standard_normal((128, 4))

        R = conjugate_criterion(X, Y)

        assert isinstance(R, np.ndarray)
        assert R.shape == (5,)
        assert np.all((R >= 0.0) & (R <= 1.0))

    def test_vector_lying_exactly_in_subspace_has_r_equal_one(self):
        Y = np.eye(10)[:, :3]
        x = Y[:, 0] * 2.0 + Y[:, 1] * 3.0  # линейная комбинация базисных векторов
        # reg_param по умолчанию (1e-8) чуть смещает результат от идеальной 1.0.
        r = conjugate_criterion(x, Y, reg_param=0.0)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_vector_orthogonal_to_subspace_has_r_equal_zero(self):
        Y = np.eye(10)[:, :3]
        x = np.eye(10)[:, 5]  # ортогонален первым трём базисным векторам
        r = conjugate_criterion(x, Y)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_1d_basis_vector_is_reshaped_to_column(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(16)
        y_1d = rng.standard_normal(16)
        r = conjugate_criterion(x, y_1d)
        assert 0.0 <= r <= 1.0

    def test_mismatched_feature_dimension_raises(self):
        X = np.random.randn(3, 16)
        Y = np.random.randn(32, 2)
        with pytest.raises(ValueError):
            conjugate_criterion(X, Y)

    def test_degenerate_basis_does_not_produce_nan(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((5, 128))
        Y_basis = rng.standard_normal((128, 4))
        # Добавляем коллинеарный столбец -> вырожденный базис.
        Y_singular = np.column_stack([Y_basis, Y_basis[:, [0]] * 2.5])

        R = conjugate_criterion(X, Y_singular)

        assert not np.isnan(R).any()
        assert np.all((R >= 0.0) & (R <= 1.0))

    def test_zero_vector_does_not_divide_by_zero(self):
        x = np.zeros(16)
        Y = np.random.randn(16, 2)
        r = conjugate_criterion(x, Y)
        assert np.isfinite(r)


class TestCosineSimilarityMatrix:
    def test_pairwise_self_similarity_shape_and_diagonal(self):
        X = np.random.randn(10, 32)
        sim = cosine_similarity_matrix(X)
        assert sim.shape == (10, 10)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-10)

    def test_values_within_valid_cosine_range(self):
        X = np.random.randn(20, 16)
        sim = cosine_similarity_matrix(X)
        assert np.all(sim >= -1.0 - 1e-10)
        assert np.all(sim <= 1.0 + 1e-10)

    def test_symmetric_for_self_similarity(self):
        X = np.random.randn(15, 8)
        sim = cosine_similarity_matrix(X)
        np.testing.assert_allclose(sim, sim.T, atol=1e-10)

    def test_orthogonal_vectors_have_zero_similarity(self):
        X = np.eye(4)
        sim = cosine_similarity_matrix(X)
        off_diagonal = sim[~np.eye(4, dtype=bool)]
        np.testing.assert_allclose(off_diagonal, 0.0, atol=1e-10)

    def test_two_matrix_form_returns_rectangular_shape(self):
        X = np.random.randn(5, 32)
        Y = np.random.randn(7, 32)
        sim = cosine_similarity_matrix(X, Y)
        assert sim.shape == (5, 7)

    def test_zero_vector_does_not_produce_nan(self):
        X = np.vstack([np.zeros(8), np.random.randn(8)])
        sim = cosine_similarity_matrix(X)
        assert not np.isnan(sim).any()
