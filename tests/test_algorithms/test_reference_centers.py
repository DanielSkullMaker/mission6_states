"""Theory tests для ReferenceCenterBuilder (canonical algorithm A.2-A.3).

Проверяет корректность канонического алгоритма поиска центров подклассов
через последовательный минимум показателя сопряжённости R(x, Y).
"""

import pytest
import numpy as np

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
from subspace_conjugacy.core.metrics import conjugate_criterion


@pytest.mark.theory
class TestReferenceCenterBuilderTheory:
    """Тесты канонической теории A.2-A.3 на синтетических данных."""

    def test_returns_correct_number_of_centers(self):
        """Должен вернуть ровно n_subclasses центров."""
        np.random.seed(42)
        X = np.random.randn(100, 64)
        initial_pair = (5, 23)

        for n_subclasses in [2, 4, 8, 12]:
            builder = ReferenceCenterBuilder(n_subclasses=n_subclasses)
            builder.fit(X, initial_pair)

            assert len(builder.center_indices_) == n_subclasses, (
                f"Ожидалось {n_subclasses} центров, получено {len(builder.center_indices_)}"
            )

    def test_initial_pair_preserved(self):
        """Первые два центра должны совпадать с initial_pair."""
        np.random.seed(42)
        X = np.random.randn(50, 32)
        initial_pair = (3, 17)

        builder = ReferenceCenterBuilder(n_subclasses=8)
        builder.fit(X, initial_pair)

        centers = builder.center_indices_
        assert centers[0] == initial_pair[0], "Первый центр должен совпадать с initial_pair[0]"
        assert centers[1] == initial_pair[1], "Второй центр должен совпадать с initial_pair[1]"

    def test_all_centers_unique(self):
        """Все центры должны быть уникальными (нет дубликатов)."""
        np.random.seed(42)
        X = np.random.randn(100, 128)
        initial_pair = (10, 45)

        builder = ReferenceCenterBuilder(n_subclasses=16)
        builder.fit(X, initial_pair)

        centers = builder.center_indices_
        unique_centers = set(centers)

        assert len(unique_centers) == len(centers), (
            f"Найдены дубликаты в центрах: {centers}"
        )

    def test_centers_within_valid_range(self):
        """Все индексы центров должны быть в диапазоне [0, M)."""
        np.random.seed(42)
        X = np.random.randn(80, 64)
        M = X.shape[0]
        initial_pair = (0, 1)

        builder = ReferenceCenterBuilder(n_subclasses=10)
        builder.fit(X, initial_pair)

        centers = builder.center_indices_

        assert all(0 <= c < M for c in centers), (
            f"Некоторые индексы вне диапазона [0, {M}): {centers}"
        )

    def test_third_center_has_minimum_r_with_pair(self):
        """Третий центр должен иметь минимальный R относительно первой пары."""
        np.random.seed(42)
        X = np.random.randn(30, 16)
        initial_pair = (0, 1)

        builder = ReferenceCenterBuilder(n_subclasses=3, reg_param=1e-8)
        builder.fit(X, initial_pair, store_history=True)

        centers = builder.center_indices_
        third_center = centers[2]

        # Проверяем, что third_center действительно имеет минимальный R
        Y_pair = X[list(initial_pair)].T  # (N, 2)
        remaining = [i for i in range(len(X)) if i not in initial_pair]
        r_values = conjugate_criterion(X[remaining], Y_pair, reg_param=1e-8)

        min_r_idx = np.argmin(r_values)
        expected_third = remaining[min_r_idx]

        assert third_center == expected_third, (
            f"Третий центр {third_center} не совпадает с ожидаемым {expected_third} "
            f"(argmin R относительно пары)"
        )

    def test_growing_basis_invariant(self):
        """Y должен расти на каждом шаге (k-1 центров для поиска k-го)."""
        np.random.seed(42)
        X = np.random.randn(50, 32)
        initial_pair = (5, 10)

        builder = ReferenceCenterBuilder(n_subclasses=6)
        builder.fit(X, initial_pair, store_history=True)

        # Проверяем размеры базисов на каждом шаге
        assert builder.r_values_history_ is not None
        history = builder.r_values_history_

        # Шаг 0: поиск 3-го центра, Y имеет 2 столбца
        # Шаг 1: поиск 4-го центра, Y имеет 3 столбца
        # ...
        # Шаг k: поиск (k+3)-го центра, Y имеет (k+2) столбца

        for step_idx, r_vals in enumerate(history):
            k_current = step_idx + 2  # Размерность Y на этом шаге
            n_remaining = len(X) - (k_current)  # Кандидаты

            assert len(r_vals) == n_remaining, (
                f"Шаг {step_idx}: ожидалось {n_remaining} кандидатов, "
                f"получено {len(r_vals)}"
            )

    def test_each_new_center_has_min_r(self):
        """Каждый новый центр должен быть argmin R среди оставшихся."""
        np.random.seed(42)
        X = np.random.randn(40, 48)
        initial_pair = (2, 15)

        builder = ReferenceCenterBuilder(n_subclasses=8, reg_param=1e-8)
        builder.fit(X, initial_pair, store_history=True)

        centers = list(builder.center_indices_)
        history = builder.r_values_history_

        # Проверяем каждый шаг
        for step_idx, r_vals in enumerate(history):
            current_centers = centers[:step_idx + 2]  # Уже найденные
            Y = X[current_centers].T

            remaining = [i for i in range(len(X)) if i not in current_centers]
            r_computed = conjugate_criterion(X[remaining], Y, reg_param=1e-8)

            # Индекс с минимальным R
            min_idx_local = np.argmin(r_computed)
            expected_center = remaining[min_idx_local]
            actual_center = centers[step_idx + 2]

            assert actual_center == expected_center, (
                f"Шаг {step_idx+1}: центр {actual_center} != ожидаемый {expected_center} "
                f"(argmin R)"
            )

    def test_get_center_vectors(self):
        """get_center_vectors должен вернуть правильные векторы."""
        np.random.seed(42)
        X = np.random.randn(60, 128)
        initial_pair = (3, 40)

        builder = ReferenceCenterBuilder(n_subclasses=10)
        builder.fit(X, initial_pair)

        centers_matrix = builder.get_center_vectors(X)

        assert centers_matrix.shape == (10, 128)

        # Проверяем совпадение с исходными векторами
        for i, idx in enumerate(builder.center_indices_):
            assert np.array_equal(centers_matrix[i], X[idx])


@pytest.mark.theory
class TestReferenceCenterBuilderEdgeCases:
    """Тесты краевых случаев."""

    def test_n_subclasses_equals_two(self):
        """С n_subclasses=2 должна вернуться только initial_pair."""
        np.random.seed(42)
        X = np.random.randn(50, 32)
        initial_pair = (10, 25)

        builder = ReferenceCenterBuilder(n_subclasses=2)
        builder.fit(X, initial_pair)

        centers = builder.center_indices_
        assert len(centers) == 2
        assert list(centers) == list(initial_pair)

    def test_n_subclasses_equals_M(self):
        """С n_subclasses=M должны быть использованы все векторы."""
        np.random.seed(42)
        M = 20
        X = np.random.randn(M, 16)
        initial_pair = (0, 1)

        builder = ReferenceCenterBuilder(n_subclasses=M)
        builder.fit(X, initial_pair)

        centers = builder.center_indices_
        assert len(centers) == M
        assert set(centers) == set(range(M))

    def test_raises_error_if_M_less_than_n_subclasses(self):
        """Должна быть ошибка, если векторов меньше, чем нужно центров."""
        np.random.seed(42)
        X = np.random.randn(10, 32)
        initial_pair = (0, 1)

        builder = ReferenceCenterBuilder(n_subclasses=20)

        with pytest.raises(ValueError, match="Недостаточно векторов"):
            builder.fit(X, initial_pair)

    def test_raises_error_with_invalid_initial_pair_same_indices(self):
        """Должна быть ошибка при initial_pair=(i, i)."""
        np.random.seed(42)
        X = np.random.randn(50, 32)

        builder = ReferenceCenterBuilder(n_subclasses=8)

        with pytest.raises(ValueError, match="должны быть различными"):
            builder.fit(X, initial_pair=(5, 5))

    def test_raises_error_with_invalid_initial_pair_out_of_range(self):
        """Должна быть ошибка при индексах вне диапазона."""
        np.random.seed(42)
        X = np.random.randn(50, 32)

        builder = ReferenceCenterBuilder(n_subclasses=8)

        with pytest.raises(ValueError, match="вне диапазона"):
            builder.fit(X, initial_pair=(5, 100))

    def test_raises_error_with_wrong_pair_length(self):
        """Должна быть ошибка при initial_pair != 2 элемента."""
        np.random.seed(42)
        X = np.random.randn(50, 32)

        builder = ReferenceCenterBuilder(n_subclasses=8)

        with pytest.raises(ValueError, match="содержать 2 индекса"):
            builder.fit(X, initial_pair=(1, 2, 3))

    def test_raises_error_before_fit(self):
        """get_center_vectors до fit должен вызывать ошибку."""
        builder = ReferenceCenterBuilder(n_subclasses=8)
        X = np.random.randn(50, 32)

        with pytest.raises(RuntimeError, match="не обучена"):
            builder.get_center_vectors(X)

    def test_deterministic_with_fixed_seed(self):
        """Результат должен быть воспроизводимым при фиксированном seed."""
        initial_pair = (5, 23)

        np.random.seed(42)
        X1 = np.random.randn(100, 128)

        np.random.seed(42)
        X2 = np.random.randn(100, 128)

        builder1 = ReferenceCenterBuilder(n_subclasses=12)
        builder1.fit(X1, initial_pair)

        builder2 = ReferenceCenterBuilder(n_subclasses=12)
        builder2.fit(X2, initial_pair)

        assert np.array_equal(builder1.center_indices_, builder2.center_indices_)


@pytest.mark.theory
class TestReferenceCenterBuilderIntegration:
    """Интеграционные тесты с GlobalMinCosinePairFinder (A.1 + A.2-A.3)."""

    def test_full_pipeline_A1_to_A3(self):
        """Полный цикл: A.1 (пара) → A.2-A.3 (центры)."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        # Фаза A.1: глобальная пара
        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X)
        initial_pair = pair_finder.pair_indices_

        # Фаза A.2-A.3: остальные центры
        builder = ReferenceCenterBuilder(n_subclasses=8)
        builder.fit(X, initial_pair)

        centers = builder.center_indices_

        assert len(centers) == 8
        assert centers[0] == initial_pair[0]
        assert centers[1] == initial_pair[1]
        assert len(set(centers)) == 8

    def test_pipeline_with_different_n_subclasses(self):
        """Проверка различных значений n_subclasses в цепочке A.1→A.2-A.3."""
        np.random.seed(42)
        X = np.random.randn(80, 128)

        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X)
        initial_pair = pair_finder.pair_indices_

        for n_sub in [4, 8, 16, 32]:
            builder = ReferenceCenterBuilder(n_subclasses=n_sub)
            builder.fit(X, initial_pair)

            assert len(builder.center_indices_) == n_sub
            assert all(idx in range(len(X)) for idx in builder.center_indices_)


@pytest.mark.theory
class TestReferenceCenterBuilderVsClusterer:
    """Тесты соответствия с SubspaceClusterer._select_initial_centers."""

    def test_matches_clusterer_center_selection(self):
        """Должен давать те же центры, что и SubspaceClusterer (шаги A.2-A.3)."""
        from subspace_conjugacy.models.clusterer import SubspaceClusterer

        np.random.seed(42)
        X = np.random.randn(100, 512)

        # Canonical алгоритм (A.1 + A.2-A.3)
        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X)
        initial_pair = pair_finder.pair_indices_

        builder = ReferenceCenterBuilder(n_subclasses=8, reg_param=1e-8)
        builder.fit(X, initial_pair)
        canonical_centers = builder.center_indices_

        # SubspaceClusterer._select_initial_centers (строки 102-123)
        clusterer = SubspaceClusterer(n_subclasses=8, reg_param=1e-8)
        from subspace_conjugacy.core.metrics import cosine_similarity_matrix

        # Повторяем логику clusterer._select_initial_centers
        sim_matrix = cosine_similarity_matrix(X)
        np.fill_diagonal(sim_matrix, np.inf)
        i1, i2 = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
        centers_clusterer = [int(i1), int(i2)]

        while len(centers_clusterer) < 8:
            Y = X[centers_clusterer].T
            remaining = [i for i in range(len(X)) if i not in centers_clusterer]
            r_vals = conjugate_criterion(X[remaining], Y, reg_param=1e-8)
            new_center = remaining[np.argmin(r_vals)]
            centers_clusterer.append(new_center)

        clusterer_centers = np.array(centers_clusterer, dtype=int)

        assert np.array_equal(canonical_centers, clusterer_centers), (
            f"ReferenceCenterBuilder и SubspaceClusterer дают разные центры:\n"
            f"Canonical: {canonical_centers}\n"
            f"Clusterer: {clusterer_centers}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "theory"])
