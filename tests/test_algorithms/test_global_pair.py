"""Theory tests для GlobalMinCosinePairFinder (canonical algorithm A.1).

Проверяет корректность канонического алгоритма на синтетических данных.
"""

import pytest
import numpy as np

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.core.metrics import cosine_similarity_matrix


@pytest.mark.theory
class TestGlobalMinCosinePairTheory:
    """Тесты канонической теории A.1 на синтетических данных."""

    def test_finds_orthogonal_vectors(self):
        """Должен найти пару ортогональных векторов (cos=0)."""
        # Создаём датасет только из ортогональных и близких векторов
        X = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],  # Похож на первый
            [0.0, 1.0, 0.0],  # Ортогонален первому
            [0.1, 0.9, 0.0],  # Похож на третий
        ])

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)

        idx1, idx2 = finder.pair_indices_
        # Должна быть найдена пара с минимальным cos (ортогональные векторы)
        cos_value = finder.similarity_value_
        assert np.isclose(cos_value, 0.0, atol=1e-6), (
            f"Ожидалось cos≈0 для ортогональной пары, получено {cos_value:.6f}"
        )

    def test_finds_opposite_vectors(self):
        """Должен найти пару противоположных векторов (cos=-1)."""
        X = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],  # Противоположен вектору 0
            [0.1, 0.2, 0.3],
        ])

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)

        idx1, idx2 = finder.pair_indices_
        assert {idx1, idx2} == {0, 2}, "Должны быть найдены противоположные векторы"
        assert np.isclose(finder.similarity_value_, -1.0, atol=1e-6)

    def test_pair_has_minimum_cosine(self):
        """Найденная пара должна иметь минимальное cos среди всех пар."""
        np.random.seed(42)
        X = np.random.randn(20, 64)

        finder = GlobalMinCosinePairFinder()
        finder.fit(X, store_matrix=True)

        idx1, idx2 = finder.pair_indices_
        min_cos = finder.similarity_value_
        sim_matrix = finder.similarity_matrix_

        # Проверяем, что это действительно минимум
        # (исключаем диагональ и нижний треугольник для уникальности)
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                cos_ij = sim_matrix[i, j]
                assert cos_ij >= min_cos - 1e-6, (
                    f"Найдена меньшая пара: cos({i},{j})={cos_ij:.6f} < "
                    f"cos({idx1},{idx2})={min_cos:.6f}"
                )

    def test_indices_are_different(self):
        """Индексы пары должны быть различными (i ≠ j)."""
        np.random.seed(42)
        X = np.random.randn(10, 32)

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)

        idx1, idx2 = finder.pair_indices_
        assert idx1 != idx2, "Индексы пары должны быть различными"

    def test_indices_sorted(self):
        """Индексы должны быть отсортированы (idx1 < idx2)."""
        np.random.seed(42)
        X = np.random.randn(10, 32)

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)

        idx1, idx2 = finder.pair_indices_
        assert idx1 < idx2, "Первый индекс должен быть меньше второго"

    def test_cosine_value_in_valid_range(self):
        """Косинусное сходство должно быть в диапазоне [-1, 1]."""
        np.random.seed(42)
        X = np.random.randn(50, 128)

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)

        cos_value = finder.similarity_value_
        assert -1.0 <= cos_value <= 1.0, (
            f"Косинусное сходство {cos_value} вне допустимого диапазона [-1, 1]"
        )

    def test_get_pair_vectors(self):
        """get_pair_vectors должен вернуть правильные векторы."""
        np.random.seed(42)
        X = np.random.randn(10, 64)

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)

        idx1, idx2 = finder.pair_indices_
        v1, v2 = finder.get_pair_vectors(X)

        assert np.array_equal(v1, X[idx1])
        assert np.array_equal(v2, X[idx2])

    def test_manual_cosine_verification(self):
        """Проверка косинусного сходства вручную."""
        np.random.seed(42)
        X = np.random.randn(10, 64)

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)

        v1, v2 = finder.get_pair_vectors(X)
        manual_cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        assert np.isclose(manual_cos, finder.similarity_value_, atol=1e-6)


@pytest.mark.theory
class TestGlobalMinCosinePairEdgeCases:
    """Тесты краевых случаев."""

    def test_two_vectors_only(self):
        """С двумя векторами должна вернуться единственная возможная пара."""
        X = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)

        assert finder.pair_indices_ == (0, 1)
        assert np.isclose(finder.similarity_value_, 0.0)

    def test_raises_error_with_one_vector(self):
        """С одним вектором должна быть ошибка."""
        X = np.array([[1.0, 2.0, 3.0]])

        finder = GlobalMinCosinePairFinder()
        with pytest.raises(ValueError, match="минимум 2 вектора"):
            finder.fit(X)

    def test_raises_error_with_empty_array(self):
        """С пустым массивом должна быть ошибка."""
        X = np.array([]).reshape(0, 10)

        finder = GlobalMinCosinePairFinder()
        with pytest.raises(ValueError, match="пустая матрица"):
            finder.fit(X)

    def test_raises_error_before_fit(self):
        """Вызов get_pair_vectors до fit должен вызывать ошибку."""
        finder = GlobalMinCosinePairFinder()
        X = np.random.randn(10, 32)

        with pytest.raises(RuntimeError, match="не обучена"):
            finder.get_pair_vectors(X)

    def test_deterministic_with_fixed_seed(self):
        """Результат должен быть воспроизводимым при фиксированном seed."""
        np.random.seed(42)
        X1 = np.random.randn(50, 128)

        np.random.seed(42)
        X2 = np.random.randn(50, 128)

        finder1 = GlobalMinCosinePairFinder()
        finder1.fit(X1)

        finder2 = GlobalMinCosinePairFinder()
        finder2.fit(X2)

        assert finder1.pair_indices_ == finder2.pair_indices_
        assert np.isclose(finder1.similarity_value_, finder2.similarity_value_)


@pytest.mark.theory
class TestGlobalMinCosinePairVsClusterer:
    """Тесты соответствия с SubspaceClusterer._select_initial_centers."""

    def test_matches_clusterer_initial_pair(self):
        """Должен давать ту же пару, что и SubspaceClusterer (шаг A.1)."""
        from subspace_conjugacy.models.clusterer import SubspaceClusterer

        np.random.seed(42)
        X = np.random.randn(100, 512)

        # Canonical алгоритм
        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        canonical_pair = finder.pair_indices_

        # SubspaceClusterer (строки 102-106)
        clusterer = SubspaceClusterer(n_subclasses=8)
        sim_matrix = cosine_similarity_matrix(X)
        np.fill_diagonal(sim_matrix, np.inf)
        i1, i2 = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
        clusterer_pair = tuple(sorted([int(i1), int(i2)]))

        assert canonical_pair == clusterer_pair, (
            f"GlobalMinCosinePairFinder и SubspaceClusterer дают разные пары: "
            f"{canonical_pair} vs {clusterer_pair}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "theory"])
