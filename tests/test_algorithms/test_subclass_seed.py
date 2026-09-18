"""Theory tests для CosineSecondVectorAttacher (canonical algorithm B.1).

Проверяет корректность канонического алгоритма формирования пар векторов
для подклассов через минимум косинусного сходства.
"""

import pytest
import numpy as np

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher
from subspace_conjugacy.core.metrics import cosine_similarity_matrix


@pytest.mark.theory
class TestCosineSecondVectorAttacherTheory:
    """Тесты канонической теории B.1 на синтетических данных."""

    def test_returns_correct_number_of_pairs(self):
        """Должен вернуть ровно n_subclasses пар."""
        np.random.seed(42)
        X = np.random.randn(100, 64)

        for n_centers in [2, 4, 8, 12]:
            centers = np.arange(n_centers)
            attacher = CosineSecondVectorAttacher()
            attacher.fit(X, centers)

            assert len(attacher.pairs_) == n_centers, (
                f"Ожидалось {n_centers} пар, получено {len(attacher.pairs_)}"
            )

    def test_first_elements_match_centers(self):
        """Первые элементы пар должны совпадать с center_indices."""
        np.random.seed(42)
        X = np.random.randn(80, 128)
        centers = np.array([5, 15, 25, 35, 45, 55, 65, 75])

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        for i, (center_in_pair, _) in enumerate(attacher.pairs_):
            assert center_in_pair == centers[i], (
                f"Пара {i}: первый элемент {center_in_pair} != центр {centers[i]}"
            )

    def test_all_indices_unique(self):
        """Все индексы в парах должны быть уникальными."""
        np.random.seed(42)
        X = np.random.randn(100, 256)
        centers = np.array([0, 10, 20, 30, 40, 50, 60, 70])

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        all_indices = attacher.pairs_.flatten()
        unique_indices = set(all_indices)

        assert len(unique_indices) == len(all_indices), (
            f"Найдены дубликаты в индексах: {all_indices}"
        )

    def test_second_vectors_not_in_centers(self):
        """Вторые векторы пар не должны быть центрами."""
        np.random.seed(42)
        X = np.random.randn(60, 32)
        centers = np.array([2, 8, 14, 20, 26, 32, 38, 44])

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        centers_set = set(centers)
        for _, second_idx in attacher.pairs_:
            assert second_idx not in centers_set, (
                f"Второй вектор {second_idx} является центром"
            )

    def test_each_second_vector_has_minimum_cosine(self):
        """Каждый второй вектор должен иметь минимальный cos с центром."""
        np.random.seed(42)
        X = np.random.randn(50, 64)
        centers = np.array([0, 5, 10, 15, 20, 25, 30, 35])

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers, store_cosine_values=True)

        cos_matrix = cosine_similarity_matrix(X)
        used_indices = set(centers)

        for i, (center_idx, second_idx) in enumerate(attacher.pairs_):
            # Доступные индексы на момент выбора этого second_idx
            # (исключаем уже использованные центры и вторые векторы предыдущих пар)
            remaining_at_step = set(range(len(X))) - used_indices

            # Косинусы центра с оставшимися векторами
            cos_with_center = [cos_matrix[center_idx, j] for j in remaining_at_step]
            min_cos_expected = min(cos_with_center)
            actual_cos = cos_matrix[center_idx, second_idx]

            assert np.isclose(actual_cos, min_cos_expected, atol=1e-6), (
                f"Пара {i}: косинус {actual_cos:.6f} не является минимальным "
                f"(ожидалось {min_cos_expected:.6f})"
            )

            # Добавляем second_idx в использованные
            used_indices.add(second_idx)

    def test_pairs_shape_is_correct(self):
        """Массив пар должен иметь форму (n_subclasses, 2)."""
        np.random.seed(42)
        X = np.random.randn(100, 128)
        centers = np.arange(0, 40, 5)  # 8 центров

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        assert attacher.pairs_.shape == (8, 2)

    def test_get_subspace_bases_returns_correct_shapes(self):
        """get_subspace_bases должен вернуть базисы (N, 2)."""
        np.random.seed(42)
        X = np.random.randn(80, 256)
        centers = np.array([1, 11, 21, 31, 41, 51, 61, 71])

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        bases = attacher.get_subspace_bases(X)

        assert len(bases) == len(centers)
        for i, Y in enumerate(bases):
            assert Y.shape == (256, 2), (
                f"Базис {i} имеет форму {Y.shape}, ожидалось (256, 2)"
            )

    def test_bases_columns_match_pair_vectors(self):
        """Столбцы базисов должны соответствовать векторам пар."""
        np.random.seed(42)
        X = np.random.randn(50, 32)
        centers = np.array([0, 6, 12, 18, 24, 30, 36, 42])

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        bases = attacher.get_subspace_bases(X)

        for i, Y in enumerate(bases):
            center_idx, second_idx = attacher.pairs_[i]
            expected_col1 = X[center_idx]
            expected_col2 = X[second_idx]

            assert np.array_equal(Y[:, 0], expected_col1), f"Базис {i}: столбец 0 не совпадает"
            assert np.array_equal(Y[:, 1], expected_col2), f"Базис {i}: столбец 1 не совпадает"

    def test_get_pair_vectors(self):
        """get_pair_vectors должен вернуть правильные векторы."""
        np.random.seed(42)
        X = np.random.randn(60, 64)
        centers = np.array([2, 12, 22, 32, 42, 52])

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        for i in range(len(centers)):
            v1, v2 = attacher.get_pair_vectors(X, subclass_index=i)
            center_idx, second_idx = attacher.pairs_[i]

            assert np.array_equal(v1, X[center_idx])
            assert np.array_equal(v2, X[second_idx])


@pytest.mark.theory
class TestCosineSecondVectorAttacherEdgeCases:
    """Тесты краевых случаев."""

    def test_two_centers_only(self):
        """С 2 центрами должны быть сформированы 2 пары."""
        np.random.seed(42)
        X = np.random.randn(20, 16)
        centers = np.array([0, 10])

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        assert len(attacher.pairs_) == 2
        assert attacher.pairs_.shape == (2, 2)

    def test_minimum_vectors_scenario(self):
        """С M = 2*n_centers должен использовать все векторы."""
        np.random.seed(42)
        n_centers = 5
        X = np.random.randn(n_centers * 2, 32)
        centers = np.arange(n_centers)

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        all_indices = set(attacher.pairs_.flatten())
        assert len(all_indices) == n_centers * 2

    def test_raises_error_with_insufficient_vectors(self):
        """Должна быть ошибка, если векторов меньше 2*n_centers."""
        np.random.seed(42)
        X = np.random.randn(10, 32)
        centers = np.arange(8)  # 8 центров, но нужно 16 векторов

        attacher = CosineSecondVectorAttacher()

        with pytest.raises(ValueError, match="Недостаточно векторов"):
            attacher.fit(X, centers)

    def test_raises_error_with_duplicate_centers(self):
        """Должна быть ошибка при дубликатах в center_indices."""
        np.random.seed(42)
        X = np.random.randn(50, 32)

        attacher = CosineSecondVectorAttacher()

        with pytest.raises(ValueError, match="дубликаты"):
            attacher.fit(X, np.array([0, 5, 5, 10]))

    def test_raises_error_with_centers_out_of_range(self):
        """Должна быть ошибка при индексах вне диапазона."""
        np.random.seed(42)
        X = np.random.randn(50, 32)

        attacher = CosineSecondVectorAttacher()

        with pytest.raises(ValueError, match="вне диапазона"):
            attacher.fit(X, np.array([0, 5, 100]))

    def test_raises_error_with_empty_centers(self):
        """Должна быть ошибка при пустом массиве центров."""
        np.random.seed(42)
        X = np.random.randn(50, 32)

        attacher = CosineSecondVectorAttacher()

        with pytest.raises(ValueError, match="не может быть пустым"):
            attacher.fit(X, np.array([]))

    def test_raises_error_before_fit(self):
        """Вызовы методов до fit должны вызывать ошибку."""
        attacher = CosineSecondVectorAttacher()
        X = np.random.randn(50, 32)

        with pytest.raises(RuntimeError, match="не обучена"):
            attacher.get_subspace_bases(X)

        with pytest.raises(RuntimeError, match="не обучена"):
            attacher.get_pair_vectors(X, 0)

    def test_get_pair_vectors_raises_error_with_invalid_index(self):
        """get_pair_vectors с некорректным индексом должен вызывать ошибку."""
        np.random.seed(42)
        X = np.random.randn(50, 32)
        centers = np.arange(8)

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        with pytest.raises(IndexError):
            attacher.get_pair_vectors(X, subclass_index=10)

    def test_deterministic_with_fixed_seed(self):
        """Результат должен быть воспроизводимым при фиксированном seed."""
        centers = np.array([5, 15, 25, 35, 45, 55, 65, 75])

        np.random.seed(42)
        X1 = np.random.randn(100, 128)

        np.random.seed(42)
        X2 = np.random.randn(100, 128)

        attacher1 = CosineSecondVectorAttacher()
        attacher1.fit(X1, centers)

        attacher2 = CosineSecondVectorAttacher()
        attacher2.fit(X2, centers)

        assert np.array_equal(attacher1.pairs_, attacher2.pairs_)


@pytest.mark.theory
class TestCosineSecondVectorAttacherIntegration:
    """Интеграционные тесты полного pipeline A.1 + A.2-A.3 + B.1."""

    def test_full_pipeline_A1_to_B1(self):
        """Полный цикл: A.1 (пара) → A.2-A.3 (центры) → B.1 (пары)."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        # Фаза A.1: глобальная пара
        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X)
        initial_pair = pair_finder.pair_indices_

        # Фаза A.2-A.3: центры
        builder = ReferenceCenterBuilder(n_subclasses=8)
        builder.fit(X, initial_pair)
        centers = builder.center_indices_

        # Фаза B.1: пары
        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        pairs = attacher.pairs_

        assert len(pairs) == 8
        assert pairs.shape == (8, 2)
        assert len(set(pairs.flatten())) == 16  # Все уникальные

    def test_pipeline_with_different_n_subclasses(self):
        """Проверка различных значений n_subclasses в цепочке A.1→A.3→B.1."""
        np.random.seed(42)
        X = np.random.randn(120, 128)

        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X)
        initial_pair = pair_finder.pair_indices_

        for n_sub in [4, 8, 12, 16]:
            builder = ReferenceCenterBuilder(n_subclasses=n_sub)
            builder.fit(X, initial_pair)
            centers = builder.center_indices_

            attacher = CosineSecondVectorAttacher()
            attacher.fit(X, centers)

            assert len(attacher.pairs_) == n_sub
            assert attacher.pairs_.shape == (n_sub, 2)

    def test_bases_ready_for_phase_B2(self):
        """Базисы готовы для использования в ConjugacyClusterGrowth (B.2)."""
        np.random.seed(42)
        X = np.random.randn(100, 512)

        # Полный pipeline A.1 → A.3 → B.1
        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X)

        builder = ReferenceCenterBuilder(n_subclasses=8)
        builder.fit(X, pair_finder.pair_indices_)

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, builder.center_indices_)

        bases = attacher.get_subspace_bases(X)

        # Проверка готовности для B.2
        assert len(bases) == 8
        assert all(Y.shape == (512, 2) for Y in bases)

        # Проверка, что базисы не вырожденные
        for i, Y in enumerate(bases):
            gram = Y.T @ Y
            det = np.linalg.det(gram)
            assert det > 1e-6, f"Базис {i} вырожден (det={det})"


@pytest.mark.theory
class TestCosineSecondVectorAttacherProperties:
    """Тесты математических свойств алгоритма."""

    def test_cosine_values_in_valid_range(self):
        """Сохранённые косинусные значения должны быть в [-1, 1]."""
        np.random.seed(42)
        X = np.random.randn(80, 64)
        centers = np.arange(0, 40, 5)

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers, store_cosine_values=True)

        cos_vals = attacher.cosine_values_

        assert cos_vals is not None
        assert all(-1.0 <= c <= 1.0 for c in cos_vals), (
            f"Косинусы вне диапазона [-1, 1]: {cos_vals}"
        )

    def test_bases_are_linearly_independent(self):
        """Пары векторов должны быть линейно независимыми."""
        np.random.seed(42)
        X = np.random.randn(100, 128)
        centers = np.array([0, 12, 24, 36, 48, 60, 72, 84])

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        for i in range(len(centers)):
            v1, v2 = attacher.get_pair_vectors(X, i)

            # Проверка линейной независимости через определитель Грама
            gram = np.array([[np.dot(v1, v1), np.dot(v1, v2)],
                            [np.dot(v2, v1), np.dot(v2, v2)]])
            det = np.linalg.det(gram)

            assert det > 1e-6, (
                f"Пара {i} линейно зависима (det={det})"
            )

    def test_sequential_removal_from_remaining(self):
        """Вторые векторы должны последовательно удаляться из remaining."""
        np.random.seed(42)
        X = np.random.randn(50, 32)
        centers = np.array([0, 6, 12, 18, 24, 30])

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        # Проверяем, что второй вектор i-й пары не используется в последующих парах
        second_vectors = [pair[1] for pair in attacher.pairs_]

        for i, second_idx in enumerate(second_vectors):
            # Этот вектор не должен быть вторым вектором в последующих парах
            assert second_idx not in second_vectors[i+1:], (
                f"Второй вектор {second_idx} пары {i} повторно использован"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "theory"])
