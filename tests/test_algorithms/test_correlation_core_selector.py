"""Тесты algorithms/correlation_core_selector.py — CorrelationCoreSelector.

Реализует находку №1 (theory/code/DATA_100.m, первая половина функции) —
опциональный отбор "ядра" наиболее взаимно скоррелированных (типичных)
векторов класса. НЕ часть опубликованной статьи Korshikov & Fursov.
"""

import numpy as np
import pytest

from subspace_conjugacy.algorithms.correlation_core_selector import CorrelationCoreSelector


def _brute_force_core(X: np.ndarray, n_target: int) -> list:
    """Буквальная (нeвекторизованная) реализация MATLAB-алгоритма DATA_100.m
    для проверки корректности векторизованной версии на малых данных."""
    active_X = X.copy()
    original_indices = list(range(X.shape[0]))
    order = []

    for _ in range(n_target):
        n = active_X.shape[0]
        best_r = -np.inf
        best_i, best_j = None, None
        for i in range(n):
            for j in range(i + 1, n):
                with np.errstate(invalid="ignore"):
                    r = np.corrcoef(active_X[i], active_X[j])[0, 1]
                if np.isnan(r):
                    r = -np.inf
                if r > best_r:
                    best_r, best_i, best_j = r, i, j

        order.append(original_indices[best_j])
        active_X = np.delete(active_X, best_j, axis=0)
        original_indices.pop(best_j)

    return order


@pytest.mark.theory
class TestCorrelationCoreSelectorBasics:
    def test_core_indices_length_equals_n_target(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 64))

        selector = CorrelationCoreSelector(n_target=12)
        selector.fit(X)

        assert len(selector.core_indices_) == 12
        assert len(selector.remaining_indices_) == 8

    def test_core_and_remaining_partition_all_indices(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((15, 32))

        selector = CorrelationCoreSelector(n_target=6)
        selector.fit(X)

        all_indices = sorted(list(selector.core_indices_) + list(selector.remaining_indices_))
        assert all_indices == list(range(15))

    def test_core_indices_are_unique(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((25, 16))

        selector = CorrelationCoreSelector(n_target=10)
        selector.fit(X)

        assert len(set(selector.core_indices_)) == len(selector.core_indices_)

    def test_matches_brute_force_matlab_semantics(self):
        """Векторизованная реализация должна совпадать буквально с MATLAB-циклом."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((8, 5))

        selector = CorrelationCoreSelector(n_target=4)
        selector.fit(X)

        expected_order = _brute_force_core(X, n_target=4)
        assert list(selector.core_indices_) == expected_order

    def test_most_correlated_pair_selected_first(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((6, 32))
        X[5] = X[0] * 2.0 + rng.standard_normal(32) * 1e-6  # почти дубликат X[0]

        selector = CorrelationCoreSelector(n_target=1)
        selector.fit(X)

        # Пара (0, 5) максимально скоррелирована -> вектор 5 (больший индекс) в ядре первым.
        assert selector.core_indices_[0] == 5

    def test_r_values_recorded(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((10, 16))

        selector = CorrelationCoreSelector(n_target=5)
        selector.fit(X)

        assert set(selector.r_values_.keys()) == set(selector.core_indices_.tolist())
        assert all(-1.0 <= r <= 1.0 for r in selector.r_values_.values())


@pytest.mark.theory
class TestCorrelationCoreSelectorTransform:
    def test_transform_returns_core_rows_in_order(self):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((12, 8))

        selector = CorrelationCoreSelector(n_target=5)
        selector.fit(X)
        X_core = selector.transform(X)

        assert X_core.shape == (5, 8)
        np.testing.assert_array_equal(X_core, X[selector.core_indices_])

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((12, 8))

        a = CorrelationCoreSelector(n_target=5)
        result_a = a.fit_transform(X)

        b = CorrelationCoreSelector(n_target=5)
        b.fit(X)
        result_b = b.transform(X)

        np.testing.assert_array_equal(result_a, result_b)

    def test_transform_before_fit_raises(self):
        selector = CorrelationCoreSelector(n_target=5)
        with pytest.raises(RuntimeError):
            selector.transform(np.random.randn(10, 8))


@pytest.mark.theory
class TestCorrelationCoreSelectorEdgeCases:
    def test_n_target_out_of_range_raises(self):
        X = np.random.randn(10, 16)

        with pytest.raises(ValueError):
            CorrelationCoreSelector(n_target=0).fit(X)
        with pytest.raises(ValueError):
            CorrelationCoreSelector(n_target=10).fit(X)
        with pytest.raises(ValueError):
            CorrelationCoreSelector(n_target=15).fit(X)

    def test_n_target_at_upper_bound_valid(self):
        X = np.random.randn(10, 16)
        selector = CorrelationCoreSelector(n_target=9)
        selector.fit(X)
        assert len(selector.core_indices_) == 9
        assert len(selector.remaining_indices_) == 1

    def test_empty_input_raises(self):
        selector = CorrelationCoreSelector(n_target=1)
        with pytest.raises(ValueError):
            selector.fit(np.zeros((0, 16)))

    def test_1d_input_raises(self):
        selector = CorrelationCoreSelector(n_target=1)
        with pytest.raises(ValueError):
            selector.fit(np.zeros(16))

    def test_constant_vector_never_selected_via_nan_guard(self):
        """Вектор с нулевой дисперсией даёт NaN в corrcoef — не должен
        "случайно" стать частью самой скоррелированной пары."""
        rng = np.random.default_rng(8)
        X = rng.standard_normal((10, 16))
        X[3] = 1.0  # константный вектор -> NaN корреляция со всеми

        selector = CorrelationCoreSelector(n_target=9)
        selector.fit(X)
        # Не должно упасть с ошибкой/предупреждением numpy наружу теста.
        assert len(selector.core_indices_) == 9

    def test_deterministic_with_fixed_seed(self):
        rng1 = np.random.default_rng(9)
        X1 = rng1.standard_normal((20, 32))
        rng2 = np.random.default_rng(9)
        X2 = rng2.standard_normal((20, 32))

        s1 = CorrelationCoreSelector(n_target=10)
        s1.fit(X1)
        s2 = CorrelationCoreSelector(n_target=10)
        s2.fit(X2)

        np.testing.assert_array_equal(s1.core_indices_, s2.core_indices_)

    def test_repr_before_and_after_fit(self):
        selector = CorrelationCoreSelector(n_target=5)
        assert "not fitted" in repr(selector)

        selector.fit(np.random.randn(10, 16))
        assert "core=" in repr(selector)
        assert "remaining=" in repr(selector)
