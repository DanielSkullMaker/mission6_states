"""Тесты algorithms/reference_filter.py — LinearDependencyFilter.

Реализует "Problem Definition" из статьи Korshikov & Fursov: "almost
linearly dependent vectors are excluded from the set of reference vectors"
(refactoring_plan.txt, раздел 10, находка №3).

Все тесты используют N (размерность признаков) существенно больше M (числа
векторов), как и в реальном сценарии (МРТ: N=65536, M — сотни изображений)
— см. docstring модуля reference_filter.py про естественное насыщение
фильтра, когда число принятых векторов приближается к N.
"""

import numpy as np
import pytest

from subspace_conjugacy.algorithms.reference_filter import LinearDependencyFilter


@pytest.mark.theory
class TestLinearDependencyFilterBasics:
    def test_first_vector_always_kept(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 64))

        filt = LinearDependencyFilter()
        filt.fit(X)

        assert 0 in filt.kept_indices_
        assert 0 not in filt.excluded_indices_

    def test_no_duplicates_nothing_excluded(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((10, 64))

        filt = LinearDependencyFilter(threshold=0.999)
        filt.fit(X)

        assert len(filt.excluded_indices_) == 0
        np.testing.assert_array_equal(filt.kept_indices_, np.arange(10))

    def test_near_exact_duplicate_is_excluded(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((10, 64))
        X[7] = X[0] * 3.0 + 1e-7  # почти точный дубликат (с масштабом) X[0]

        filt = LinearDependencyFilter(threshold=0.999)
        filt.fit(X)

        assert 7 in filt.excluded_indices_
        assert filt.r_values_[7] >= 0.999

    def test_exact_duplicate_has_r_close_to_one(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((10, 64))
        X[5] = X[2].copy()

        filt = LinearDependencyFilter(threshold=0.999)
        filt.fit(X)

        assert 5 in filt.excluded_indices_
        assert filt.r_values_[5] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_never_excluded(self):
        X = np.eye(20)[:, :10].T  # 10 ортогональных единичных векторов, N=20

        filt = LinearDependencyFilter(threshold=0.999)
        filt.fit(X)

        assert len(filt.excluded_indices_) == 0

    def test_kept_and_excluded_partition_all_indices(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((15, 64))
        X[10] = X[1] * 2.0
        X[12] = X[3] * -1.5

        filt = LinearDependencyFilter(threshold=0.999)
        filt.fit(X)

        all_indices = sorted(list(filt.kept_indices_) + list(filt.excluded_indices_))
        assert all_indices == list(range(15))


@pytest.mark.theory
class TestLinearDependencyFilterThreshold:
    def test_lower_threshold_excludes_more(self):
        """Меньший threshold — более агрессивная фильтрация (менее почти
        зависимые, но всё же похожие векторы тоже начинают исключаться)."""
        rng = np.random.default_rng(5)
        base = rng.standard_normal(64)
        X = np.vstack([
            base,
            base + rng.standard_normal(64) * 0.01,  # почти параллелен base
            rng.standard_normal(64),
            rng.standard_normal(64),
        ])

        strict = LinearDependencyFilter(threshold=0.9999)
        strict.fit(X)
        lenient = LinearDependencyFilter(threshold=0.5)
        lenient.fit(X)

        assert len(lenient.excluded_indices_) >= len(strict.excluded_indices_)

    def test_threshold_must_be_in_valid_range(self):
        with pytest.raises(ValueError):
            LinearDependencyFilter(threshold=0.0)
        with pytest.raises(ValueError):
            LinearDependencyFilter(threshold=1.5)
        with pytest.raises(ValueError):
            LinearDependencyFilter(threshold=-0.1)

    def test_threshold_at_upper_bound_is_valid(self):
        LinearDependencyFilter(threshold=1.0)  # не должно бросать


@pytest.mark.theory
class TestLinearDependencyFilterTransform:
    def test_transform_returns_only_kept_rows(self):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((10, 64))
        X[5] = X[0] * 2.0 + 1e-7

        filt = LinearDependencyFilter(threshold=0.999)
        filt.fit(X)
        X_filtered = filt.transform(X)

        assert X_filtered.shape[0] == len(filt.kept_indices_)
        np.testing.assert_array_equal(X_filtered, X[filt.kept_indices_])

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((10, 64))
        X[3] = X[1] * 1.5 + 1e-7

        a = LinearDependencyFilter(threshold=0.999)
        result_a = a.fit_transform(X)

        b = LinearDependencyFilter(threshold=0.999)
        b.fit(X)
        result_b = b.transform(X)

        np.testing.assert_array_equal(result_a, result_b)

    def test_transform_before_fit_raises(self):
        filt = LinearDependencyFilter()
        with pytest.raises(RuntimeError):
            filt.transform(np.random.randn(5, 10))


@pytest.mark.theory
class TestLinearDependencyFilterEdgeCases:
    def test_single_vector_input(self):
        X = np.random.randn(1, 16)
        filt = LinearDependencyFilter()
        filt.fit(X)

        assert list(filt.kept_indices_) == [0]
        assert len(filt.excluded_indices_) == 0

    def test_empty_input_raises(self):
        filt = LinearDependencyFilter()
        with pytest.raises(ValueError):
            filt.fit(np.zeros((0, 16)))

    def test_1d_input_raises(self):
        filt = LinearDependencyFilter()
        with pytest.raises(ValueError):
            filt.fit(np.zeros(16))

    def test_saturation_once_accepted_count_reaches_dimensionality(self):
        """Как только принятых векторов становится N (размерность), любой
        следующий вектор математически неизбежно лежит в их span — см.
        Notes в docstring модуля. Не баг, а свойство критерия сопряжённости."""
        rng = np.random.default_rng(8)
        N = 8
        X = rng.standard_normal((N + 5, N))  # M > N: заведомо избыточный набор

        filt = LinearDependencyFilter(threshold=0.999)
        filt.fit(X)

        # Первые N векторов (если независимы) образуют базис полного ранга;
        # все последующие обязаны быть исключены.
        assert len(filt.kept_indices_) <= N + 1  # +1 запас на численные эффекты
        assert len(filt.excluded_indices_) >= 4

    def test_repr_before_and_after_fit(self):
        filt = LinearDependencyFilter(threshold=0.99)
        assert "not fitted" in repr(filt)

        filt.fit(np.random.randn(5, 16))
        assert "kept=" in repr(filt)
        assert "excluded=" in repr(filt)
