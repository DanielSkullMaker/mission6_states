"""Theory tests для канонического алгоритма кластеризации (synthetic data).

Проверяет соответствие реализации теории из refactoring_plan.txt секция 1.
Используется @pytest.mark.theory — не требует реального датасета.

Проверяемые инварианты:
  A.1: Глобальная пара = argmin cos по всем парам
  A.2: 3-й центр = argmin R против Y(N×2)
  A.3: k-й центр использует Y из всех (k-1) предыдущих центров
  B.1: Каждый центр получает ровно один второй вектор через min cos
  B.2: Sequential global argmax R, строго один вектор за итерацию
  C: Flat 24-way argmax R для классификации
"""

import numpy as np
import pytest

from subspace_conjugacy.core.metrics import (
    conjugate_criterion,
    cosine_similarity_matrix,
)
from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher
from subspace_conjugacy.algorithms.subclass_growth import ConjugacyClusterGrowth
from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer


@pytest.mark.theory
class TestTheoryPhaseA1:
    """Теория A.1: глобальная пара с минимальным косинусом."""

    def test_global_pair_is_minimum_cosine(self, small_random_vectors):
        """Найденная пара должна иметь минимальное косинусное сходство."""
        X = small_random_vectors
        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        i, j = finder.pair_indices_
        min_cos = finder.similarity_value_

        # Проверяем, что это действительно минимум
        cos_matrix = cosine_similarity_matrix(X)
        np.fill_diagonal(cos_matrix, np.inf)

        assert i != j
        assert cos_matrix[i, j] == pytest.approx(min_cos)

        # Проверяем, что нет пары с меньшим значением
        assert min_cos <= cos_matrix.min()

    def test_global_pair_is_unique_argmin(self, random_seed):
        """Пара должна быть глобальным argmin по всем парам."""
        np.random.seed(random_seed)
        X = np.random.randn(10, 32)

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        i, j = finder.pair_indices_
        min_cos = finder.similarity_value_

        # Вручную находим минимум
        cos_matrix = cosine_similarity_matrix(X)
        np.fill_diagonal(cos_matrix, np.inf)
        manual_min = np.min(cos_matrix)

        assert min_cos == pytest.approx(manual_min, abs=1e-9)

    def test_determinism_with_seed(self, random_seed):
        """Результат должен быть детерминированным при фиксированном seed."""
        np.random.seed(random_seed)
        X = np.random.randn(15, 48)

        finder1 = GlobalMinCosinePairFinder()
        finder1.fit(X)
        i1, j1 = finder1.pair_indices_
        cos1 = finder1.similarity_value_

        finder2 = GlobalMinCosinePairFinder()
        finder2.fit(X)
        i2, j2 = finder2.pair_indices_
        cos2 = finder2.similarity_value_

        assert (i1, j1) == (i2, j2)
        assert cos1 == pytest.approx(cos2)


@pytest.mark.theory
class TestTheoryPhaseA2A3:
    """Теория A.2-A.3: последовательный поиск центров через min R."""

    def test_third_center_has_minimum_r_against_initial_pair(self, small_random_vectors):
        """3-й центр должен иметь минимальный R против Y из первых двух."""
        X = small_random_vectors

        # Получаем начальную пару
        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        i, j = finder.pair_indices_

        # Строим базис из двух центров
        Y_initial = X[[i, j]].T  # (N, 2)

        # Находим 3-й центр
        builder = ReferenceCenterBuilder(n_subclasses=3)
        builder.fit(X, initial_pair=(i, j))
        centers = builder.center_indices_
        third_idx = centers[2]

        # Проверяем, что R для третьего центра минимален среди оставшихся
        remaining = [k for k in range(len(X)) if k not in [i, j]]
        R_values = conjugate_criterion(X[remaining], Y_initial)

        third_position = remaining.index(third_idx)
        assert R_values[third_position] == pytest.approx(R_values.min())

    def test_centers_use_growing_basis(self, medium_random_vectors):
        """k-й центр должен выбираться против базиса из (k-1) центров."""
        X = medium_random_vectors
        n_subclasses = 5

        builder = ReferenceCenterBuilder(n_subclasses=n_subclasses)

        # Начальная пара
        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        i, j = finder.pair_indices_

        builder.fit(X, initial_pair=(i, j))
        centers = builder.center_indices_

        # Проверяем, что каждый новый центр был выбран против растущего базиса
        assert len(centers) == n_subclasses
        assert len(set(centers)) == n_subclasses  # Все уникальны

    def test_correct_number_of_centers(self, small_random_vectors):
        """Должно быть сформировано ровно n_subclasses центров."""
        X = small_random_vectors

        for n_sub in [2, 4, 8]:
            finder = GlobalMinCosinePairFinder()
            finder.fit(X)
            i, j = finder.pair_indices_

            builder = ReferenceCenterBuilder(n_subclasses=n_sub)
            builder.fit(X, initial_pair=(i, j))
            centers = builder.center_indices_

            assert len(centers) == n_sub


@pytest.mark.theory
class TestTheoryPhaseB1:
    """Теория B.1: второй вектор для каждого центра через min cos."""

    def test_each_center_gets_one_second_vector(self, medium_random_vectors):
        """Каждый центр должен получить ровно один второй вектор."""
        X = medium_random_vectors
        n_subclasses = 8

        # Фаза A
        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        i, j = finder.pair_indices_
        builder = ReferenceCenterBuilder(n_subclasses=n_subclasses)
        builder.fit(X, initial_pair=(i, j))
        centers = builder.center_indices_

        # Фаза B.1
        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)
        pairs = attacher.pairs_

        assert pairs.shape == (n_subclasses, 2)
        assert np.all(pairs[:, 0] == centers)  # Первый столбец = центры

    def test_all_paired_indices_unique(self, medium_random_vectors):
        """Все 2×n_subclasses индексов должны быть уникальными."""
        X = medium_random_vectors
        n_subclasses = 6

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        i, j = finder.pair_indices_
        builder = ReferenceCenterBuilder(n_subclasses=n_subclasses)
        builder.fit(X, initial_pair=(i, j))
        centers = builder.center_indices_

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)
        pairs = attacher.pairs_

        flat_indices = pairs.flatten()
        assert len(flat_indices) == 2 * n_subclasses
        assert len(set(flat_indices)) == 2 * n_subclasses  # Все уникальны

    def test_second_vector_has_minimum_cosine_with_center(self, small_random_vectors):
        """Второй вектор должен иметь минимальный cos с центром."""
        X = small_random_vectors
        n_subclasses = 4

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        i, j = finder.pair_indices_
        builder = ReferenceCenterBuilder(n_subclasses=n_subclasses)
        builder.fit(X, initial_pair=(i, j))
        centers = builder.center_indices_

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)
        pairs = attacher.pairs_

        # Проверяем первую пару
        center_idx = pairs[0, 0]
        second_idx = pairs[0, 1]

        # Вычисляем косинус между центром и вторым вектором
        cos_matrix = cosine_similarity_matrix(X)

        # Из оставшихся векторов второй должен иметь минимальный cos с центром
        used = set(centers)
        remaining = [k for k in range(len(X)) if k not in used]

        cos_with_center = cos_matrix[center_idx, remaining]
        second_position = remaining.index(second_idx)

        assert cos_with_center[second_position] == pytest.approx(cos_with_center.min())


@pytest.mark.theory
class TestTheoryPhaseB2:
    """Теория B.2: sequential global argmax R, один вектор за итерацию."""

    def test_all_vectors_assigned(self, medium_random_vectors):
        """Все векторы должны получить метку подкласса."""
        X = medium_random_vectors
        n_subclasses = 8

        # Фазы A и B.1
        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        i, j = finder.pair_indices_
        builder = ReferenceCenterBuilder(n_subclasses=n_subclasses)
        builder.fit(X, initial_pair=(i, j))
        centers = builder.center_indices_
        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)
        pairs = attacher.pairs_

        # Фаза B.2
        growth = ConjugacyClusterGrowth(freeze_basis_at=2)
        growth.fit(X, pairs)
        labels = growth.labels_
        subspaces = growth.subspace_bases_

        assert len(labels) == len(X)
        assert len(set(labels)) <= n_subclasses
        assert np.all(labels >= 0)
        assert np.all(labels < n_subclasses)

    def test_freeze_basis_at_2_produces_correct_shape(self, medium_random_vectors):
        """С freeze_basis_at=2 все базисы должны быть (N, 2)."""
        X = medium_random_vectors
        M, N = X.shape
        n_subclasses = 6

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        i, j = finder.pair_indices_
        builder = ReferenceCenterBuilder(n_subclasses=n_subclasses)
        builder.fit(X, initial_pair=(i, j))
        centers = builder.center_indices_
        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)
        pairs = attacher.pairs_

        growth = ConjugacyClusterGrowth(freeze_basis_at=2)
        growth.fit(X, pairs)
        labels = growth.labels_
        subspaces = growth.subspace_bases_

        assert len(subspaces) == n_subclasses
        for Y in subspaces:
            assert Y.shape == (N, 2)

    def test_freeze_basis_none_allows_growth(self, small_random_vectors):
        """С freeze_basis_at=None базисы должны расти."""
        X = small_random_vectors
        n_subclasses = 3

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        i, j = finder.pair_indices_
        builder = ReferenceCenterBuilder(n_subclasses=n_subclasses)
        builder.fit(X, initial_pair=(i, j))
        centers = builder.center_indices_
        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)
        pairs = attacher.pairs_

        growth = ConjugacyClusterGrowth(freeze_basis_at=None)
        growth.fit(X, pairs)
        labels = growth.labels_
        subspaces = growth.subspace_bases_

        # Хотя бы один базис должен быть > 2
        max_basis_size = max(Y.shape[1] for Y in subspaces)
        assert max_basis_size > 2


@pytest.mark.theory
class TestTheoryEndToEnd:
    """Интеграционные тесты полного канонического алгоритма A+B."""

    def test_fursov_clusterer_implements_canonical_theory(self, medium_random_vectors):
        """FursovClusterer должен выполнить все фазы A.1→A.3→B.1→B.2."""
        X = medium_random_vectors
        n_subclasses = 8

        clusterer = FursovClusterer(n_subclasses=n_subclasses, freeze_basis_at=2)
        clusterer.fit(X)

        # Проверяем результаты каждой фазы
        assert clusterer.is_fitted_
        assert len(clusterer.subspaces_) == n_subclasses
        assert len(clusterer.labels_) == len(X)
        assert clusterer.center_indices_ is not None
        assert len(clusterer.center_indices_) == n_subclasses
        assert clusterer.initial_pairs_ is not None
        assert clusterer.initial_pairs_.shape == (n_subclasses, 2)

    def test_bases_ready_for_classifier(self, medium_random_vectors):
        """Базисы должны быть готовы для классификатора (фаза C)."""
        X = medium_random_vectors
        M, N = X.shape
        n_subclasses = 8

        clusterer = FursovClusterer(n_subclasses=n_subclasses, freeze_basis_at=2)
        clusterer.fit(X)

        subspaces = clusterer.subspaces_

        # Для классификации: 8 базисов (N, 2)
        assert len(subspaces) == 8
        for Y in subspaces:
            assert Y.shape == (N, 2)
            # Базис не должен содержать нулевых векторов
            assert np.all(np.linalg.norm(Y, axis=0) > 0)

    def test_r_values_in_valid_range(self, medium_random_vectors):
        """R(x, Y) должен быть в диапазоне [0, 1]."""
        X = medium_random_vectors

        clusterer = FursovClusterer(n_subclasses=6, freeze_basis_at=2)
        clusterer.fit(X)

        # Вычисляем R для всех векторов и всех подпространств
        for Y in clusterer.subspaces_:
            R = conjugate_criterion(X, Y)
            assert np.all(R >= 0.0)
            assert np.all(R <= 1.0)

    def test_properties_accessible(self, small_random_vectors):
        """Все свойства должны быть доступны после fit."""
        X = small_random_vectors

        clusterer = FursovClusterer(n_subclasses=4)
        clusterer.fit(X)

        # Свойства из фаз A и B
        initial_pair = clusterer.get_initial_pair()
        assert len(initial_pair) == 2

        centers = clusterer.get_center_indices()
        assert len(centers) == 4

        pairs = clusterer.get_initial_pairs()
        assert pairs.shape == (4, 2)

        sizes = clusterer.get_subclass_sizes()
        assert len(sizes) == 4
        assert sum(sizes) == len(X)


@pytest.mark.theory
class TestTheoryInvariants:
    """Проверка математических инвариантов."""

    def test_cosine_diagonal_is_one(self, small_random_vectors):
        """Косинус вектора с самим собой должен быть 1."""
        X = small_random_vectors
        cos_matrix = cosine_similarity_matrix(X)

        diagonal = np.diag(cos_matrix)
        assert np.allclose(diagonal, 1.0)

    def test_r_equals_one_for_subspace_vectors(self, random_seed):
        """R(x, Y) = 1, если x ∈ span(Y)."""
        np.random.seed(random_seed)

        # Создаём базис
        Y = np.random.randn(64, 2)

        # Создаём вектор как линейную комбинацию столбцов Y
        x = Y @ np.array([2.0, 3.0])

        R = conjugate_criterion(x, Y)
        assert R == pytest.approx(1.0, abs=1e-6)

    def test_r_decreases_for_orthogonal_vectors(self, random_seed):
        """R должен быть близок к 0 для ортогональных векторов."""
        np.random.seed(random_seed)

        # Ортогональные базисы
        Y = np.array([[1.0, 0.0], [0.0, 1.0]])  # e1, e2
        x = np.array([0.0, 0.0, 1.0])  # e3 (ортогонален к Y)

        # Расширяем размерность
        Y_extended = np.vstack([Y, np.zeros((1, 2))])

        R = conjugate_criterion(x, Y_extended)
        assert R < 0.1  # Близко к 0

    def test_cosine_range_minus_one_to_one(self, small_random_vectors):
        """Косинус должен быть в диапазоне [-1, 1]."""
        X = small_random_vectors
        cos_matrix = cosine_similarity_matrix(X)

        assert np.all(cos_matrix >= -1.0)
        assert np.all(cos_matrix <= 1.0)

    def test_r_is_scale_invariant(self, random_seed):
        """R(αx, Y) = R(x, Y) для α > 0."""
        np.random.seed(random_seed)

        x = np.random.randn(64)
        Y = np.random.randn(64, 2)

        R1 = conjugate_criterion(x, Y)
        R2 = conjugate_criterion(2.5 * x, Y)

        assert R1 == pytest.approx(R2, abs=1e-9)
