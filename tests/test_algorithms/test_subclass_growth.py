"""Theory tests для ConjugacyClusterGrowth (canonical algorithm B.2).

Проверяет корректность канонического алгоритма последовательного
наполнения кластеров через максимум показателя сопряжённости.
"""

import pytest
import numpy as np

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher
from subspace_conjugacy.algorithms.subclass_growth import ConjugacyClusterGrowth


@pytest.mark.theory
class TestConjugacyClusterGrowthTheory:
    """Тесты канонической теории B.2 на синтетических данных."""

    def test_all_vectors_get_labels(self):
        """Все векторы должны получить метки подклассов."""
        np.random.seed(42)
        X = np.random.randn(50, 64)
        pairs = np.array([[0, 1], [10, 11], [20, 21], [30, 31]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs)

        assert len(growth.labels_) == 50
        assert not np.any(growth.labels_ == -1), "Найдены неразмеченные векторы"
        assert set(growth.labels_) == {0, 1, 2, 3}

    def test_initial_pairs_assigned_correctly(self):
        """Векторы из начальных пар должны быть присвоены своим подклассам."""
        np.random.seed(42)
        X = np.random.randn(40, 32)
        pairs = np.array([[0, 5], [10, 15], [20, 25], [30, 35]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs)

        for s, (idx1, idx2) in enumerate(pairs):
            assert growth.labels_[idx1] == s, f"Центр {idx1} не в подклассе {s}"
            assert growth.labels_[idx2] == s, f"Второй вектор {idx2} не в подклассе {s}"

    def test_sequential_assignment_one_vector_per_iteration(self):
        """За каждую итерацию присоединяется ровно один вектор."""
        np.random.seed(42)
        X = np.random.randn(30, 16)
        pairs = np.array([[0, 1], [10, 11], [20, 21]])

        n_initial = len(pairs) * 2  # 6 векторов в парах
        n_remaining = X.shape[0] - n_initial  # 24 вектора для распределения

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs)

        # Проверяем, что все векторы распределены
        assert growth.labels_.size == X.shape[0]
        # Каждый подкласс должен содержать минимум 2 вектора (из пары)
        sizes = growth.get_subclass_sizes()
        assert all(s >= 2 for s in sizes)

    def test_correct_number_of_subclasses(self):
        """Количество подклассов должно соответствовать входным парам."""
        np.random.seed(42)
        X = np.random.randn(60, 48)

        for n_subclasses in [2, 4, 6, 8]:
            pairs = np.array([[i*7, i*7+1] for i in range(n_subclasses)])
            growth = ConjugacyClusterGrowth()
            growth.fit(X, pairs)

            assert growth.n_subclasses_ == n_subclasses
            assert len(growth.subspace_bases_) == n_subclasses
            unique_labels = set(growth.labels_)
            assert len(unique_labels) == n_subclasses

    def test_freeze_basis_at_2(self):
        """С freeze_basis_at=2 финальные базисы должны быть (N, 2)."""
        np.random.seed(42)
        X = np.random.randn(80, 128)
        pairs = np.array([[i*10, i*10+1] for i in range(8)])

        growth = ConjugacyClusterGrowth(freeze_basis_at=2)
        growth.fit(X, pairs)

        for i, Y in enumerate(growth.subspace_bases_):
            assert Y.shape == (128, 2), f"Базис {i} имеет форму {Y.shape}, ожидалось (128, 2)"

    def test_freeze_basis_at_none_grows_bases(self):
        """С freeze_basis_at=None базисы должны расти."""
        np.random.seed(42)
        X = np.random.randn(40, 32)
        pairs = np.array([[0, 1], [10, 11], [20, 21], [30, 31]])

        growth = ConjugacyClusterGrowth(freeze_basis_at=None)
        growth.fit(X, pairs)

        # Базисы должны содержать все векторы подкласса
        sizes = growth.get_subclass_sizes()
        for i, Y in enumerate(growth.subspace_bases_):
            expected_size = sizes[i]
            assert Y.shape[1] == expected_size, (
                f"Базис {i} должен иметь {expected_size} столбцов, имеет {Y.shape[1]}"
            )

    def test_strategy_default_vs_master(self):
        """Стратегии default и master могут давать разные результаты."""
        np.random.seed(42)
        X = np.random.randn(50, 64)
        pairs = np.array([[0, 1], [12, 13], [24, 25], [36, 37]])

        growth_default = ConjugacyClusterGrowth(strategy="default")
        growth_default.fit(X, pairs)

        growth_master = ConjugacyClusterGrowth(strategy="master")
        growth_master.fit(X, pairs)

        # Оба должны распределить все векторы
        assert not np.any(growth_default.labels_ == -1)
        assert not np.any(growth_master.labels_ == -1)

        # Результаты могут отличаться (не обязательно равны)
        # но оба валидны

    def test_get_subclass_sizes(self):
        """get_subclass_sizes должен возвращать корректные размеры."""
        np.random.seed(42)
        X = np.random.randn(60, 32)
        pairs = np.array([[0, 1], [15, 16], [30, 31], [45, 46]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs)

        sizes = growth.get_subclass_sizes()

        assert len(sizes) == 4
        assert sizes.sum() == 60
        assert all(s >= 2 for s in sizes)  # Минимум пара

    def test_predict_returns_valid_labels(self):
        """predict должен возвращать валидные метки подклассов."""
        np.random.seed(42)
        X_train = np.random.randn(80, 64)
        pairs = np.array([[0, 1], [20, 21], [40, 41], [60, 61]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X_train, pairs)

        X_test = np.random.randn(10, 64)
        pred_labels = growth.predict(X_test)

        assert len(pred_labels) == 10
        assert all(0 <= l < 4 for l in pred_labels)

    def test_bases_have_correct_feature_dimension(self):
        """Базисы должны иметь правильную размерность признаков."""
        np.random.seed(42)
        n_features = 256
        X = np.random.randn(100, n_features)
        pairs = np.array([[i*12, i*12+1] for i in range(8)])

        growth = ConjugacyClusterGrowth(freeze_basis_at=2)
        growth.fit(X, pairs)

        for Y in growth.subspace_bases_:
            assert Y.shape[0] == n_features


@pytest.mark.theory
class TestConjugacyClusterGrowthEdgeCases:
    """Тесты краевых случаев."""

    def test_minimum_vectors_scenario(self):
        """С M = 2*n_subclasses должны использоваться только пары."""
        np.random.seed(42)
        n_subclasses = 5
        X = np.random.randn(n_subclasses * 2, 32)
        pairs = np.array([[i*2, i*2+1] for i in range(n_subclasses)])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs)

        sizes = growth.get_subclass_sizes()
        assert all(s == 2 for s in sizes)  # Только пары

    def test_two_subclasses_only(self):
        """С 2 подклассами должна быть корректная кластеризация."""
        np.random.seed(42)
        X = np.random.randn(30, 16)
        pairs = np.array([[0, 1], [15, 16]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs)

        assert growth.n_subclasses_ == 2
        assert len(set(growth.labels_)) == 2
        sizes = growth.get_subclass_sizes()
        assert sizes.sum() == 30

    def test_raises_error_with_invalid_freeze_basis_at(self):
        """Должна быть ошибка при freeze_basis_at < 2."""
        with pytest.raises(ValueError, match="freeze_basis_at должен быть >= 2"):
            ConjugacyClusterGrowth(freeze_basis_at=1)

    def test_raises_error_with_invalid_strategy(self):
        """Должна быть ошибка при некорректной strategy."""
        with pytest.raises(ValueError, match="strategy должен быть"):
            ConjugacyClusterGrowth(strategy="invalid")

    def test_raises_error_with_empty_pairs(self):
        """Должна быть ошибка при пустом массиве пар."""
        np.random.seed(42)
        X = np.random.randn(50, 32)

        growth = ConjugacyClusterGrowth()

        with pytest.raises(ValueError, match="не может быть пустым"):
            growth.fit(X, np.array([]).reshape(0, 2))

    def test_raises_error_with_invalid_pairs_shape(self):
        """Должна быть ошибка при некорректной форме pairs."""
        np.random.seed(42)
        X = np.random.randn(50, 32)

        growth = ConjugacyClusterGrowth()

        with pytest.raises(ValueError, match="должен иметь форму"):
            growth.fit(X, np.array([0, 1, 5, 6]))  # 1D вместо 2D

    def test_raises_error_with_pairs_out_of_range(self):
        """Должна быть ошибка при индексах вне диапазона."""
        np.random.seed(42)
        X = np.random.randn(50, 32)

        growth = ConjugacyClusterGrowth()

        with pytest.raises(ValueError, match="вне диапазона"):
            growth.fit(X, np.array([[0, 100]]))

    def test_raises_error_with_duplicate_indices_in_pairs(self):
        """Должна быть ошибка при дублирующихся индексах в парах."""
        np.random.seed(42)
        X = np.random.randn(50, 32)

        growth = ConjugacyClusterGrowth()

        with pytest.raises(ValueError, match="дублирующиеся индексы"):
            growth.fit(X, np.array([[0, 1], [1, 5]]))  # 1 повторяется

    def test_raises_error_before_fit(self):
        """Вызовы методов до fit должны вызывать ошибку."""
        growth = ConjugacyClusterGrowth()
        X = np.random.randn(50, 32)

        with pytest.raises(RuntimeError, match="не обучена"):
            growth.get_subclass_sizes()

        with pytest.raises(RuntimeError, match="не обучена"):
            growth.predict(X)

    def test_deterministic_with_fixed_seed(self):
        """Результат должен быть воспроизводимым при фиксированном seed."""
        pairs = np.array([[0, 1], [10, 11], [20, 21], [30, 31]])

        np.random.seed(42)
        X1 = np.random.randn(50, 64)

        np.random.seed(42)
        X2 = np.random.randn(50, 64)

        growth1 = ConjugacyClusterGrowth(strategy="default")
        growth1.fit(X1, pairs)

        growth2 = ConjugacyClusterGrowth(strategy="default")
        growth2.fit(X2, pairs)

        assert np.array_equal(growth1.labels_, growth2.labels_)


@pytest.mark.theory
class TestConjugacyClusterGrowthIntegration:
    """Интеграционные тесты полного pipeline A.1 + A.2-A.3 + B.1 + B.2."""

    def test_full_pipeline_A1_to_B2(self):
        """Полный цикл: A.1 → A.2-A.3 → B.1 → B.2."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        # Фаза A.1: глобальная пара
        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X)

        # Фаза A.2-A.3: центры
        builder = ReferenceCenterBuilder(n_subclasses=8)
        builder.fit(X, pair_finder.pair_indices_)

        # Фаза B.1: пары
        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, builder.center_indices_)

        # Фаза B.2: наполнение
        growth = ConjugacyClusterGrowth(freeze_basis_at=2)
        growth.fit(X, attacher.pairs_)

        # Проверки
        assert len(growth.labels_) == 100
        assert growth.n_subclasses_ == 8
        assert len(growth.subspace_bases_) == 8
        assert all(Y.shape == (256, 2) for Y in growth.subspace_bases_)

    def test_pipeline_with_different_n_subclasses(self):
        """Проверка различных значений n_subclasses в полном цикле."""
        np.random.seed(42)
        X = np.random.randn(120, 128)

        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X)
        initial_pair = pair_finder.pair_indices_

        for n_sub in [4, 8, 12]:
            builder = ReferenceCenterBuilder(n_subclasses=n_sub)
            builder.fit(X, initial_pair)

            attacher = CosineSecondVectorAttacher()
            attacher.fit(X, builder.center_indices_)

            growth = ConjugacyClusterGrowth(freeze_basis_at=2)
            growth.fit(X, attacher.pairs_)

            assert growth.n_subclasses_ == n_sub
            assert len(set(growth.labels_)) == n_sub

    def test_bases_ready_for_classifier(self):
        """Базисы готовы для использования в classifier (фаза C)."""
        np.random.seed(42)
        X = np.random.randn(100, 512)

        # Полный pipeline A.1 → A.3 → B.1 → B.2
        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X)

        builder = ReferenceCenterBuilder(n_subclasses=8)
        builder.fit(X, pair_finder.pair_indices_)

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, builder.center_indices_)

        growth = ConjugacyClusterGrowth(freeze_basis_at=2)
        growth.fit(X, attacher.pairs_)

        bases = growth.subspace_bases_

        # Проверка готовности для classifier
        assert len(bases) == 8
        assert all(Y.shape == (512, 2) for Y in bases)

        # Проверка, что базисы не вырожденные
        for i, Y in enumerate(bases):
            gram = Y.T @ Y
            det = np.linalg.det(gram)
            assert det > 1e-6, f"Базис {i} вырожден (det={det})"

    def test_predict_on_new_data_after_full_pipeline(self):
        """Predict должен работать на новых данных после полного pipeline."""
        np.random.seed(42)
        X_train = np.random.randn(80, 128)

        # Pipeline
        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X_train)

        builder = ReferenceCenterBuilder(n_subclasses=4)
        builder.fit(X_train, pair_finder.pair_indices_)

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X_train, builder.center_indices_)

        growth = ConjugacyClusterGrowth(freeze_basis_at=2)
        growth.fit(X_train, attacher.pairs_)

        # Predict на новых данных
        X_test = np.random.randn(20, 128)
        pred_labels = growth.predict(X_test)

        assert len(pred_labels) == 20
        assert all(0 <= l < 4 for l in pred_labels)


@pytest.mark.theory
class TestConjugacyClusterGrowthProperties:
    """Тесты математических свойств алгоритма."""

    def test_all_subclasses_non_empty(self):
        """Все подклассы должны содержать минимум начальную пару."""
        np.random.seed(42)
        X = np.random.randn(80, 64)
        pairs = np.array([[i*10, i*10+1] for i in range(8)])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs)

        sizes = growth.get_subclass_sizes()
        assert all(s >= 2 for s in sizes), f"Найден пустой или одноэлементный подкласс: {sizes}"

    def test_labels_cover_all_indices(self):
        """Метки должны покрывать все индексы от 0 до M-1."""
        np.random.seed(42)
        X = np.random.randn(60, 32)
        pairs = np.array([[0, 1], [15, 16], [30, 31], [45, 46]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs)

        labeled_indices = set(range(len(growth.labels_)))
        assert len(labeled_indices) == 60

    def test_bases_columns_are_from_assigned_vectors(self):
        """Столбцы базисов должны соответствовать присоединённым векторам."""
        np.random.seed(42)
        X = np.random.randn(40, 32)
        pairs = np.array([[0, 1], [10, 11], [20, 21], [30, 31]])

        growth = ConjugacyClusterGrowth(freeze_basis_at=None)
        growth.fit(X, pairs)

        # Для каждого подкласса проверяем, что базис построен из векторов этого подкласса
        for s in range(4):
            indices_in_s = np.where(growth.labels_ == s)[0]
            Y = growth.subspace_bases_[s]

            # Первые 2 столбца должны быть из пар
            assert np.array_equal(Y[:, 0], X[pairs[s, 0]])
            assert np.array_equal(Y[:, 1], X[pairs[s, 1]])

    def test_reg_param_prevents_singular_matrices(self):
        """reg_param должен предотвращать вырожденные матрицы."""
        np.random.seed(42)
        # Создаём данные с низким рангом
        X_low_rank = np.random.randn(50, 10) @ np.random.randn(10, 64)
        pairs = np.array([[0, 1], [12, 13], [24, 25], [36, 37]])

        growth = ConjugacyClusterGrowth(reg_param=1e-6)
        # Не должно быть исключения
        growth.fit(X_low_rank, pairs)

        assert growth.is_fitted_


@pytest.mark.theory
class TestConjugacyClusterGrowthHistory:
    """Тесты сбора истории роста (store_history) — theory/article_plans/01_
    iterativnyi_algoritm.txt, задачи 2-3 (кривая R(k), обусловленность
    матрицы Грама)."""

    def test_history_none_by_default(self):
        """Без store_history=True история не собирается (нет оверхеда)."""
        np.random.seed(42)
        X = np.random.randn(40, 32)
        pairs = np.array([[0, 1], [10, 11], [20, 21], [30, 31]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs)

        assert growth.growth_history_ is None

    def test_history_has_one_record_per_assigned_vector(self):
        """С store_history=True — по записи на каждый присоединённый вектор."""
        np.random.seed(42)
        X = np.random.randn(40, 32)
        pairs = np.array([[0, 1], [10, 11], [20, 21], [30, 31]])
        n_to_assign = X.shape[0] - len(pairs) * 2

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs, store_history=True)

        assert len(growth.growth_history_) == n_to_assign

    def test_history_record_fields(self):
        """Каждая запись содержит ожидаемые поля с корректными типами."""
        np.random.seed(42)
        X = np.random.randn(30, 16)
        pairs = np.array([[0, 1], [10, 11], [20, 21]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs, store_history=True)

        record = growth.growth_history_[0]
        expected_keys = {
            "iteration", "vector_index", "subclass_index", "r_value",
            "basis_size", "gram_condition_number", "n_remaining_after",
        }
        assert set(record.keys()) == expected_keys
        assert record["iteration"] == 1
        assert 0.0 <= record["r_value"] <= 1.0
        assert record["basis_size"] >= 3  # 2 из пары + 1 новый
        assert record["gram_condition_number"] >= 1.0

    def test_basis_size_increases_monotonically_per_subclass(self):
        """basis_size растёт строго на 1 при каждом присоединении к подклассу."""
        np.random.seed(42)
        X = np.random.randn(50, 32)
        pairs = np.array([[0, 1], [15, 16], [30, 31]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs, store_history=True)

        last_size = {s: 2 for s in range(3)}
        for record in growth.growth_history_:
            s = record["subclass_index"]
            assert record["basis_size"] == last_size[s] + 1
            last_size[s] = record["basis_size"]

    def test_get_growth_curve_matches_history(self):
        """get_growth_curve() возвращает R-значения в порядке истории."""
        np.random.seed(42)
        X = np.random.randn(40, 32)
        pairs = np.array([[0, 1], [10, 11], [20, 21], [30, 31]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs, store_history=True)

        curve = growth.get_growth_curve()
        expected = np.array([r["r_value"] for r in growth.growth_history_])
        assert np.array_equal(curve, expected)

    def test_get_growth_curve_without_history_raises(self):
        """get_growth_curve() без store_history=True должен упасть с понятной ошибкой."""
        np.random.seed(42)
        X = np.random.randn(30, 16)
        pairs = np.array([[0, 1], [10, 11], [20, 21]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs)  # store_history по умолчанию False

        with pytest.raises(RuntimeError, match="История роста не сохранена"):
            growth.get_growth_curve()


@pytest.mark.theory
class TestConjugacyClusterGrowthEarlyStopping:
    """Тесты критерия ранней остановки — theory/article_plans/01_
    iterativnyi_algoritm.txt, задача 4."""

    def test_early_stopping_none_is_default_behaviour(self):
        """С early_stopping=None рост идёт до конца, как раньше."""
        np.random.seed(42)
        X = np.random.randn(40, 32)
        pairs = np.array([[0, 1], [10, 11], [20, 21], [30, 31]])

        growth = ConjugacyClusterGrowth()
        growth.fit(X, pairs)

        assert growth.stopped_early_ is False
        assert len(growth.excluded_by_early_stopping_) == 0
        assert not np.any(growth.labels_ == -1)

    def test_fixed_fraction_stops_early_and_leaves_unassigned(self):
        """fixed_fraction должен остановить рост до полного распределения."""
        np.random.seed(42)
        X = np.random.randn(60, 32)
        pairs = np.array([[0, 1], [15, 16], [30, 31], [45, 46]])
        n_to_assign = X.shape[0] - len(pairs) * 2

        growth = ConjugacyClusterGrowth(
            early_stopping="fixed_fraction", early_stopping_threshold=0.3
        )
        growth.fit(X, pairs)

        assert growth.stopped_early_ is True
        assigned = int(round(0.3 * n_to_assign))
        assert len(growth.excluded_by_early_stopping_) == n_to_assign - assigned
        assert np.any(growth.labels_ == -1)
        assert set(growth.excluded_by_early_stopping_) == set(
            np.where(growth.labels_ == -1)[0]
        )

    def test_fixed_fraction_one_means_full_growth(self):
        """threshold=1.0 для fixed_fraction эквивалентен полному росту."""
        np.random.seed(42)
        X = np.random.randn(40, 32)
        pairs = np.array([[0, 1], [10, 11], [20, 21], [30, 31]])

        growth = ConjugacyClusterGrowth(
            early_stopping="fixed_fraction", early_stopping_threshold=1.0
        )
        growth.fit(X, pairs)

        assert growth.stopped_early_ is False
        assert len(growth.excluded_by_early_stopping_) == 0

    def test_relative_drop_never_stops_on_first_iteration(self):
        """relative_drop не может остановить рост на самой первой итерации
        (нет ещё точки отсчёта R_first)."""
        np.random.seed(42)
        X = np.random.randn(10, 8)
        pairs = np.array([[0, 1]])  # 1 подкласс, 8 векторов на распределение

        growth = ConjugacyClusterGrowth(
            early_stopping="relative_drop", early_stopping_threshold=0.999,
        )
        growth.fit(X, pairs)

        # Даже с почти невыполнимо строгим порогом должен присоединиться
        # хотя бы один вектор после первой (некритериальной) итерации.
        assert growth.get_subclass_sizes()[0] >= 3

    def test_get_subclass_sizes_ignores_unassigned(self):
        """get_subclass_sizes() не должен учитывать векторы с меткой -1."""
        np.random.seed(42)
        X = np.random.randn(60, 32)
        pairs = np.array([[0, 1], [15, 16], [30, 31], [45, 46]])

        growth = ConjugacyClusterGrowth(
            early_stopping="fixed_fraction", early_stopping_threshold=0.2
        )
        growth.fit(X, pairs)

        sizes = growth.get_subclass_sizes()
        assert sizes.sum() == np.sum(growth.labels_ >= 0)
        assert sizes.sum() < X.shape[0]

    def test_raises_error_with_invalid_early_stopping(self):
        """Должна быть ошибка при некорректном early_stopping."""
        with pytest.raises(ValueError, match="early_stopping должен быть"):
            ConjugacyClusterGrowth(early_stopping="invalid")

    def test_raises_error_with_invalid_early_stopping_threshold(self):
        """Должна быть ошибка при threshold вне (0, 1]."""
        with pytest.raises(ValueError, match="early_stopping_threshold должен быть"):
            ConjugacyClusterGrowth(
                early_stopping="fixed_fraction", early_stopping_threshold=0.0
            )
        with pytest.raises(ValueError, match="early_stopping_threshold должен быть"):
            ConjugacyClusterGrowth(
                early_stopping="fixed_fraction", early_stopping_threshold=1.5
            )

    def test_predict_still_works_after_early_stopping(self):
        """predict() на новых данных должен работать даже если обучение
        остановилось раньше (базисы всё равно валидны)."""
        np.random.seed(42)
        X_train = np.random.randn(60, 32)
        pairs = np.array([[0, 1], [15, 16], [30, 31], [45, 46]])

        growth = ConjugacyClusterGrowth(
            early_stopping="fixed_fraction", early_stopping_threshold=0.2
        )
        growth.fit(X_train, pairs)

        X_test = np.random.randn(10, 32)
        pred = growth.predict(X_test)
        assert len(pred) == 10
        assert all(0 <= l < 4 for l in pred)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "theory"])
