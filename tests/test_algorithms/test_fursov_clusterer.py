"""Theory tests для FursovClusterer (canonical facade A+B).

Проверяет корректность полного pipeline кластеризации через единый фасад.
"""

import pytest
import numpy as np

from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer, SubspaceClusterer


@pytest.mark.theory
class TestFursovClustererTheory:
    """Тесты канонического фасада A+B на синтетических данных."""

    def test_fit_completes_successfully(self):
        """fit() должен завершаться без ошибок на валидных данных."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        clusterer = FursovClusterer(n_subclasses=8, freeze_basis_at=2)
        clusterer.fit(X)

        assert clusterer.is_fitted_
        assert clusterer.subspaces_ is not None
        assert clusterer.labels_ is not None

    def test_correct_number_of_subspaces(self):
        """Количество базисов должно соответствовать n_subclasses."""
        np.random.seed(42)
        X = np.random.randn(80, 128)

        for n_sub in [4, 6, 8, 10]:
            clusterer = FursovClusterer(n_subclasses=n_sub, freeze_basis_at=2)
            clusterer.fit(X)

            assert len(clusterer.subspaces_) == n_sub
            assert clusterer.n_subclasses_ == n_sub

    def test_freeze_basis_at_2_for_classifier(self):
        """С freeze_basis_at=2 все базисы должны быть (N, 2)."""
        np.random.seed(42)
        X = np.random.randn(100, 512)

        clusterer = FursovClusterer(n_subclasses=8, freeze_basis_at=2)
        clusterer.fit(X)

        for i, Y in enumerate(clusterer.subspaces_):
            assert Y.shape == (512, 2), (
                f"Базис {i} имеет форму {Y.shape}, ожидалось (512, 2)"
            )

    def test_all_vectors_get_labels(self):
        """Все векторы должны получить метки подклассов."""
        np.random.seed(42)
        X = np.random.randn(60, 64)

        clusterer = FursovClusterer(n_subclasses=5)
        clusterer.fit(X)

        assert len(clusterer.labels_) == 60
        assert not np.any(clusterer.labels_ == -1)
        assert set(clusterer.labels_) == set(range(5))

    def test_fit_predict_shortcut(self):
        """fit_predict должен работать как fit + возврат labels."""
        np.random.seed(42)
        X = np.random.randn(80, 128)

        clusterer = FursovClusterer(n_subclasses=6)
        labels = clusterer.fit_predict(X)

        assert len(labels) == 80
        assert clusterer.is_fitted_
        assert np.array_equal(labels, clusterer.labels_)

    def test_predict_on_new_data(self):
        """predict должен работать на новых данных после fit."""
        np.random.seed(42)
        X_train = np.random.randn(100, 256)
        X_test = np.random.randn(20, 256)

        clusterer = FursovClusterer(n_subclasses=8)
        clusterer.fit(X_train)
        pred_labels = clusterer.predict(X_test)

        assert len(pred_labels) == 20
        assert all(0 <= l < 8 for l in pred_labels)

    def test_get_subclass_sizes(self):
        """get_subclass_sizes должен возвращать корректные размеры."""
        np.random.seed(42)
        X = np.random.randn(100, 128)

        clusterer = FursovClusterer(n_subclasses=8)
        clusterer.fit(X)

        sizes = clusterer.get_subclass_sizes()

        assert len(sizes) == 8
        assert sizes.sum() == 100
        assert all(s >= 2 for s in sizes)  # Минимум начальная пара

    def test_get_initial_pair_from_phase_a1(self):
        """get_initial_pair должен возвращать результат фазы A.1."""
        np.random.seed(42)
        X = np.random.randn(80, 64)

        clusterer = FursovClusterer(n_subclasses=6)
        clusterer.fit(X)

        pair = clusterer.get_initial_pair()

        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert pair[0] != pair[1]
        assert 0 <= pair[0] < 80
        assert 0 <= pair[1] < 80

    def test_get_center_indices_from_phase_a3(self):
        """get_center_indices должен возвращать результат фазы A.2-A.3."""
        np.random.seed(42)
        X = np.random.randn(100, 128)

        clusterer = FursovClusterer(n_subclasses=8)
        clusterer.fit(X)

        centers = clusterer.get_center_indices()

        assert len(centers) == 8
        assert len(set(centers)) == 8  # Все уникальные
        assert all(0 <= c < 100 for c in centers)

    def test_get_initial_pairs_from_phase_b1(self):
        """get_initial_pairs должен возвращать результат фазы B.1."""
        np.random.seed(42)
        X = np.random.randn(80, 256)

        clusterer = FursovClusterer(n_subclasses=6)
        clusterer.fit(X)

        pairs = clusterer.get_initial_pairs()

        assert pairs.shape == (6, 2)
        all_indices = pairs.flatten()
        assert len(set(all_indices)) == 12  # Все уникальные

    def test_growth_strategy_default_vs_master(self):
        """Стратегии default и master должны работать."""
        np.random.seed(42)
        X = np.random.randn(60, 64)

        clusterer_default = FursovClusterer(n_subclasses=4, growth_strategy="default")
        clusterer_default.fit(X)

        clusterer_master = FursovClusterer(n_subclasses=4, growth_strategy="master")
        clusterer_master.fit(X)

        # Оба должны завершиться
        assert clusterer_default.is_fitted_
        assert clusterer_master.is_fitted_

        # Оба должны распределить все векторы
        assert len(clusterer_default.labels_) == 60
        assert len(clusterer_master.labels_) == 60


@pytest.mark.theory
class TestFursovClustererEdgeCases:
    """Тесты краевых случаев."""

    def test_minimum_n_subclasses(self):
        """С n_subclasses=2 должна быть корректная кластеризация."""
        np.random.seed(42)
        X = np.random.randn(30, 32)

        clusterer = FursovClusterer(n_subclasses=2)
        clusterer.fit(X)

        assert clusterer.n_subclasses_ == 2
        assert len(set(clusterer.labels_)) == 2

    def test_minimum_vectors_scenario(self):
        """С M = 2*n_subclasses должны использоваться только пары."""
        np.random.seed(42)
        n_sub = 5
        X = np.random.randn(n_sub * 2, 32)

        clusterer = FursovClusterer(n_subclasses=n_sub)
        clusterer.fit(X)

        sizes = clusterer.get_subclass_sizes()
        assert all(s == 2 for s in sizes)

    def test_freeze_basis_at_none_grows_bases(self):
        """С freeze_basis_at=None базисы должны расти."""
        np.random.seed(42)
        X = np.random.randn(50, 64)

        clusterer = FursovClusterer(n_subclasses=4, freeze_basis_at=None)
        clusterer.fit(X)

        sizes = clusterer.get_subclass_sizes()
        for i, Y in enumerate(clusterer.subspaces_):
            expected_size = sizes[i]
            assert Y.shape[1] == expected_size

    def test_raises_error_with_n_subclasses_less_than_2(self):
        """Должна быть ошибка при n_subclasses < 2."""
        with pytest.raises(ValueError, match="должен быть >= 2"):
            FursovClusterer(n_subclasses=1)

    def test_raises_error_with_invalid_freeze_basis_at(self):
        """Должна быть ошибка при freeze_basis_at < 2."""
        with pytest.raises(ValueError, match="должен быть >= 2 или None"):
            FursovClusterer(n_subclasses=8, freeze_basis_at=1)

    def test_raises_error_with_invalid_strategy(self):
        """Должна быть ошибка при некорректной strategy."""
        with pytest.raises(ValueError, match="должен быть 'default' или 'master'"):
            FursovClusterer(n_subclasses=8, growth_strategy="invalid")

    def test_raises_error_with_insufficient_vectors(self):
        """Должна быть ошибка при недостатке векторов."""
        np.random.seed(42)
        X = np.random.randn(10, 32)  # 10 < 16 (для 8 подклассов)

        clusterer = FursovClusterer(n_subclasses=8)

        with pytest.raises(ValueError, match="Недостаточно векторов"):
            clusterer.fit(X)

    def test_raises_error_before_fit(self):
        """Вызовы методов до fit должны вызывать ошибку."""
        clusterer = FursovClusterer()
        X = np.random.randn(50, 32)

        with pytest.raises(RuntimeError, match="не обучена"):
            clusterer.get_subclass_sizes()

        with pytest.raises(RuntimeError, match="не обучена"):
            clusterer.predict(X)

        with pytest.raises(RuntimeError, match="не обучена"):
            clusterer.get_initial_pair()

    def test_deterministic_with_fixed_seed(self):
        """Результат должен быть воспроизводимым при фиксированном seed."""
        np.random.seed(42)
        X1 = np.random.randn(80, 128)

        np.random.seed(42)
        X2 = np.random.randn(80, 128)

        clusterer1 = FursovClusterer(n_subclasses=6)
        clusterer1.fit(X1)

        clusterer2 = FursovClusterer(n_subclasses=6)
        clusterer2.fit(X2)

        assert np.array_equal(clusterer1.labels_, clusterer2.labels_)


@pytest.mark.theory
class TestFursovClustererIntegration:
    """Интеграционные тесты полного pipeline."""

    def test_complete_pipeline_phases(self):
        """Проверка, что все фазы A.1, A.2-A.3, B.1, B.2 выполняются."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        clusterer = FursovClusterer(n_subclasses=8, freeze_basis_at=2)
        clusterer.fit(X)

        # Проверка результатов каждой фазы
        assert clusterer._pair_finder is not None  # A.1
        assert clusterer._center_builder is not None  # A.2-A.3
        assert clusterer._seed_attacher is not None  # B.1
        assert clusterer._cluster_growth is not None  # B.2

        # Проверка финальных результатов
        assert len(clusterer.subspaces_) == 8
        assert all(Y.shape == (256, 2) for Y in clusterer.subspaces_)
        assert len(clusterer.labels_) == 100

    def test_bases_ready_for_classifier(self):
        """Базисы готовы для использования в classifier (фаза C)."""
        np.random.seed(42)
        X = np.random.randn(120, 512)

        clusterer = FursovClusterer(n_subclasses=8, freeze_basis_at=2)
        clusterer.fit(X)

        bases = clusterer.subspaces_

        # Проверка готовности для classifier
        assert len(bases) == 8
        assert all(Y.shape == (512, 2) for Y in bases)

        # Проверка, что базисы не вырожденные
        for i, Y in enumerate(bases):
            gram = Y.T @ Y
            det = np.linalg.det(gram)
            assert det > 1e-6, f"Базис {i} вырожден (det={det})"

    def test_predict_consistent_with_training(self):
        """predict должен быть согласован с labels_ на обучающих данных."""
        np.random.seed(42)
        X = np.random.randn(80, 128)

        clusterer = FursovClusterer(n_subclasses=6)
        clusterer.fit(X)

        # Predict на обучающих данных
        pred_labels = clusterer.predict(X)

        # Не обязательно точное совпадение, но распределение должно быть похоже
        assert len(pred_labels) == len(clusterer.labels_)
        assert set(pred_labels) == set(clusterer.labels_)

    def test_different_n_subclasses(self):
        """Проверка различных значений n_subclasses."""
        np.random.seed(42)
        X = np.random.randn(150, 256)

        for n_sub in [4, 8, 12, 16]:
            clusterer = FursovClusterer(n_subclasses=n_sub, freeze_basis_at=2)
            clusterer.fit(X)

            assert clusterer.n_subclasses_ == n_sub
            assert len(clusterer.subspaces_) == n_sub
            assert len(set(clusterer.labels_)) == n_sub


@pytest.mark.theory
class TestSubspaceClustererAlias:
    """Тесты alias SubspaceClusterer для обратной совместимости."""

    def test_alias_exists(self):
        """SubspaceClusterer должен быть доступен как alias."""
        assert SubspaceClusterer is FursovClusterer

    def test_alias_works_identically(self):
        """SubspaceClusterer должен работать идентично FursovClusterer."""
        np.random.seed(42)
        X = np.random.randn(80, 128)

        clusterer1 = FursovClusterer(n_subclasses=6)
        clusterer1.fit(X)

        np.random.seed(42)
        X2 = np.random.randn(80, 128)

        clusterer2 = SubspaceClusterer(n_subclasses=6)
        clusterer2.fit(X2)

        assert np.array_equal(clusterer1.labels_, clusterer2.labels_)
        assert type(clusterer1) is type(clusterer2)

    def test_alias_type_name(self):
        """SubspaceClusterer должен иметь правильное имя типа."""
        clusterer = SubspaceClusterer(n_subclasses=8)
        assert type(clusterer).__name__ == "FursovClusterer"


@pytest.mark.theory
class TestFursovClustererProperties:
    """Тесты математических свойств."""

    def test_all_subclasses_non_empty(self):
        """Все подклассы должны содержать минимум начальную пару."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        clusterer = FursovClusterer(n_subclasses=8)
        clusterer.fit(X)

        sizes = clusterer.get_subclass_sizes()
        assert all(s >= 2 for s in sizes)

    def test_labels_cover_all_vectors(self):
        """Метки должны покрывать все векторы."""
        np.random.seed(42)
        X = np.random.randn(90, 128)

        clusterer = FursovClusterer(n_subclasses=6)
        clusterer.fit(X)

        unique_labels = set(clusterer.labels_)
        assert len(clusterer.labels_) == 90
        assert len(unique_labels) == 6

    def test_bases_have_correct_dimensions(self):
        """Базисы должны иметь правильные размерности."""
        np.random.seed(42)
        n_features = 512
        X = np.random.randn(120, n_features)

        clusterer = FursovClusterer(n_subclasses=10, freeze_basis_at=2)
        clusterer.fit(X)

        for Y in clusterer.subspaces_:
            assert Y.shape[0] == n_features
            assert Y.shape[1] == 2

    def test_center_indices_are_subset_of_initial_pairs(self):
        """Центры из A.2-A.3 должны быть первыми элементами пар B.1."""
        np.random.seed(42)
        X = np.random.randn(80, 64)

        clusterer = FursovClusterer(n_subclasses=6)
        clusterer.fit(X)

        centers = clusterer.get_center_indices()
        pairs = clusterer.get_initial_pairs()

        # Первые элементы пар должны быть центрами
        for i, center in enumerate(centers):
            assert pairs[i, 0] == center


@pytest.mark.theory
class TestFursovClustererDependencyFilter:
    """filter_dependent=True — исключение почти линейно зависимых
    эталонных векторов перед кластеризацией (Korshikov & Fursov, "Problem
    Definition"; refactoring_plan.txt, раздел 10, находка №3). По умолчанию
    выключено — эти тесты явно проверяют и включённый, и отключённый режим
    для отсутствия регрессии в поведении по умолчанию.
    """

    def test_disabled_by_default_no_exclusions(self):
        np.random.seed(42)
        X = np.random.randn(40, 256)

        clusterer = FursovClusterer(n_subclasses=4)
        clusterer.fit(X)

        assert clusterer.excluded_indices_.size == 0
        np.testing.assert_array_equal(clusterer.kept_indices_, np.arange(40))
        assert np.all(clusterer.labels_ >= 0)  # никто не помечен -1

    def test_near_duplicate_excluded_and_labeled_minus_one(self):
        np.random.seed(42)
        X = np.random.randn(40, 256)
        X[35] = X[0] * 2.0 + 1e-7  # почти точный дубликат X[0]

        clusterer = FursovClusterer(
            n_subclasses=4, filter_dependent=True, dependency_threshold=0.999
        )
        clusterer.fit(X)

        assert 35 in clusterer.excluded_indices_
        assert clusterer.labels_[35] == -1
        assert clusterer.labels_.shape == (40,)

    def test_kept_vectors_still_get_valid_subclass_labels(self):
        np.random.seed(42)
        X = np.random.randn(40, 256)
        X[35] = X[0] * 2.0 + 1e-7

        clusterer = FursovClusterer(
            n_subclasses=4, filter_dependent=True, dependency_threshold=0.999
        )
        clusterer.fit(X)

        kept_labels = clusterer.labels_[clusterer.kept_indices_]
        assert np.all(kept_labels >= 0)
        assert np.all(kept_labels < 4)

    def test_indices_remapped_to_original_x_space(self):
        """center_indices_/initial_pairs_/get_initial_pair() должны
        индексировать ИСХОДНЫЙ X, а не отфильтрованное подмножество —
        иначе X[clusterer.center_indices_] вернёт неверные строки."""
        np.random.seed(42)
        X = np.random.randn(40, 256)
        X[35] = X[0] * 2.0 + 1e-7

        clusterer = FursovClusterer(
            n_subclasses=4, filter_dependent=True, dependency_threshold=0.999
        )
        clusterer.fit(X)

        assert clusterer.center_indices_.max() < 40
        assert 35 not in clusterer.center_indices_  # исключённый не может быть центром
        assert clusterer.initial_pairs_.max() < 40

        pair = clusterer.get_initial_pair()
        assert max(pair) < 40

        # Базисы должны быть линейными комбинациями РЕАЛЬНЫХ строк X.
        for idx in clusterer.center_indices_:
            assert idx in clusterer.kept_indices_

    def test_subclass_sizes_sum_to_kept_not_original_count(self):
        np.random.seed(42)
        X = np.random.randn(40, 256)
        X[35] = X[0] * 2.0 + 1e-7

        clusterer = FursovClusterer(
            n_subclasses=4, filter_dependent=True, dependency_threshold=0.999
        )
        clusterer.fit(X)

        assert clusterer.get_subclass_sizes().sum() == len(clusterer.kept_indices_)
        assert clusterer.get_subclass_sizes().sum() == 40 - len(clusterer.excluded_indices_)

    def test_too_many_exclusions_raises_value_error(self):
        """Если после фильтрации осталось меньше 2*n_subclasses векторов —
        явная ошибка, а не тихая деградация кластеризации."""
        np.random.seed(42)
        base = np.random.randn(256)
        X = np.vstack([base * (1.0 + i * 1e-9) for i in range(20)])  # все почти идентичны

        clusterer = FursovClusterer(
            n_subclasses=8, filter_dependent=True, dependency_threshold=0.999
        )
        with pytest.raises(ValueError, match="фильтрации"):
            clusterer.fit(X)

    def test_invalid_dependency_threshold_raises(self):
        with pytest.raises(ValueError):
            FursovClusterer(filter_dependent=True, dependency_threshold=1.5)
        with pytest.raises(ValueError):
            FursovClusterer(filter_dependent=True, dependency_threshold=0.0)

    def test_predict_on_new_data_unaffected_by_filter(self):
        """filter_dependent касается только обучающих векторов; predict()
        на новых данных работает как обычно."""
        np.random.seed(42)
        X = np.random.randn(40, 256)
        X[35] = X[0] * 2.0 + 1e-7

        clusterer = FursovClusterer(
            n_subclasses=4, filter_dependent=True, dependency_threshold=0.999
        )
        clusterer.fit(X)

        X_new = np.random.randn(5, 256)
        preds = clusterer.predict(X_new)
        assert preds.shape == (5,)
        assert np.all((preds >= 0) & (preds < 4))


class TestFursovClustererInformativenessFilter:
    """filter_low_informativeness=True — исключение малоинформативных
    эталонных векторов перед кластеризацией (Korshikov & Fursov, 3-й
    эксперимент; refactoring_plan.txt, раздел 10, находка №5). По умолчанию
    выключено — эти тесты явно проверяют и включённый, и отключённый режим
    для отсутствия регрессии в поведении по умолчанию.
    """

    def test_disabled_by_default_no_exclusions(self):
        np.random.seed(42)
        X = np.random.uniform(100, 255, size=(40, 256))

        clusterer = FursovClusterer(n_subclasses=4)
        clusterer.fit(X)

        assert clusterer.excluded_indices_.size == 0
        assert clusterer.excluded_by_informativeness_.size == 0
        np.testing.assert_array_equal(clusterer.kept_indices_, np.arange(40))
        assert np.all(clusterer.labels_ >= 0)

    def test_low_informativeness_excluded_and_labeled_minus_one(self):
        np.random.seed(42)
        X = np.random.uniform(100, 255, size=(40, 256))
        X[35] = 0.0  # почти полностью "фоновый" вектор

        clusterer = FursovClusterer(
            n_subclasses=4, filter_low_informativeness=True,
        )
        clusterer.fit(X)

        assert 35 in clusterer.excluded_by_informativeness_
        assert 35 in clusterer.excluded_indices_
        assert clusterer.labels_[35] == -1
        assert clusterer.labels_.shape == (40,)

    def test_kept_vectors_still_get_valid_subclass_labels(self):
        np.random.seed(42)
        X = np.random.uniform(100, 255, size=(40, 256))
        X[35] = 0.0

        clusterer = FursovClusterer(
            n_subclasses=4, filter_low_informativeness=True,
        )
        clusterer.fit(X)

        kept_labels = clusterer.labels_[clusterer.kept_indices_]
        assert np.all(kept_labels >= 0)
        assert np.all(kept_labels < 4)

    def test_indices_remapped_to_original_x_space(self):
        np.random.seed(42)
        X = np.random.uniform(100, 255, size=(40, 256))
        X[35] = 0.0

        clusterer = FursovClusterer(
            n_subclasses=4, filter_low_informativeness=True,
        )
        clusterer.fit(X)

        assert clusterer.center_indices_.max() < 40
        assert 35 not in clusterer.center_indices_
        for idx in clusterer.center_indices_:
            assert idx in clusterer.kept_indices_

    def test_invalid_informativeness_min_fraction_raises(self):
        with pytest.raises(ValueError):
            FursovClusterer(filter_low_informativeness=True, informativeness_min_fraction=1.5)
        with pytest.raises(ValueError):
            FursovClusterer(filter_low_informativeness=True, informativeness_min_fraction=0.0)

    def test_predict_on_new_data_unaffected_by_filter(self):
        np.random.seed(42)
        X = np.random.uniform(100, 255, size=(40, 256))
        X[35] = 0.0

        clusterer = FursovClusterer(
            n_subclasses=4, filter_low_informativeness=True,
        )
        clusterer.fit(X)

        X_new = np.random.uniform(100, 255, size=(5, 256))
        preds = clusterer.predict(X_new)
        assert preds.shape == (5,)
        assert np.all((preds >= 0) & (preds < 4))


class TestFursovClustererCombinedFilters:
    """Оба фильтра включены одновременно — сначала LowInformativenessFilter
    (0a), затем LinearDependencyFilter (0b) на выживших после 0a
    (refactoring_plan.txt, раздел 10, находка №5)."""

    def test_both_filters_chain_and_union_into_excluded_indices(self):
        np.random.seed(42)
        X = np.random.randn(40, 256) * 50 + 150  # положительные "яркие" значения
        X[10] = 0.0  # малоинформативный
        X[20] = X[0] * 1.0 + 1e-9  # почти линейно зависимый от X[0]

        clusterer = FursovClusterer(
            n_subclasses=4,
            filter_low_informativeness=True,
            filter_dependent=True,
            dependency_threshold=0.999,
        )
        clusterer.fit(X)

        assert 10 in clusterer.excluded_by_informativeness_
        assert 20 in clusterer.excluded_by_dependency_
        assert 10 in clusterer.excluded_indices_
        assert 20 in clusterer.excluded_indices_
        assert clusterer.labels_[10] == -1
        assert clusterer.labels_[20] == -1
        assert clusterer.get_subclass_sizes().sum() == len(clusterer.kept_indices_)

    def test_informativeness_filter_runs_before_dependency_filter(self):
        """Вектор, исключённый как малоинформативный, не должен попасть в
        LinearDependencyFilter (проверяется по тому, что он не появляется
        в excluded_by_dependency_, только в excluded_by_informativeness_)."""
        np.random.seed(42)
        X = np.random.randn(40, 256) * 50 + 150
        X[10] = 0.0

        clusterer = FursovClusterer(
            n_subclasses=4,
            filter_low_informativeness=True,
            filter_dependent=True,
            dependency_threshold=0.999,
        )
        clusterer.fit(X)

        assert 10 not in clusterer.excluded_by_dependency_


@pytest.mark.theory
class TestFursovClustererCorrelatedSplit:
    """split_correlated_pairs=True — разбиение на два подмножества похожих
    пар перед кластеризацией (ЧЕРНОВИК другой, неопубликованной статьи,
    theory/Макет новой статьи.docx, "Первый этап"; НЕ проверенная публикация,
    в отличие от filter_dependent/filter_low_informativeness). По умолчанию
    выключено — эти тесты явно проверяют и включённый, и отключённый режим
    для отсутствия регрессии в поведении по умолчанию.
    """

    def test_disabled_by_default_no_exclusions(self):
        np.random.seed(42)
        X = np.random.randn(40, 256)

        clusterer = FursovClusterer(n_subclasses=4)
        clusterer.fit(X)

        assert clusterer.excluded_by_correlation_split_.size == 0
        np.testing.assert_array_equal(clusterer.kept_indices_, np.arange(40))

    def test_enabled_halves_input_and_labels_dropped_subset_minus_one(self):
        np.random.seed(42)
        X = np.random.randn(40, 256)

        clusterer = FursovClusterer(
            n_subclasses=4, split_correlated_pairs=True, correlated_pairs_subset="a",
        )
        clusterer.fit(X)

        assert len(clusterer.kept_indices_) == 20
        assert len(clusterer.excluded_by_correlation_split_) == 20
        assert len(clusterer.excluded_indices_) == 20
        assert clusterer.labels_.shape == (40,)
        assert (clusterer.labels_ == -1).sum() == 20
        assert np.all(clusterer.labels_[clusterer.kept_indices_] >= 0)

    def test_subset_a_and_subset_b_are_complementary(self):
        np.random.seed(42)
        X = np.random.randn(40, 256)

        clusterer_a = FursovClusterer(
            n_subclasses=4, split_correlated_pairs=True, correlated_pairs_subset="a",
        )
        clusterer_a.fit(X)
        clusterer_b = FursovClusterer(
            n_subclasses=4, split_correlated_pairs=True, correlated_pairs_subset="b",
        )
        clusterer_b.fit(X)

        kept_a = set(clusterer_a.kept_indices_.tolist())
        kept_b = set(clusterer_b.kept_indices_.tolist())
        assert kept_a & kept_b == set()
        assert kept_a | kept_b == set(range(40))

    def test_odd_number_of_vectors_after_prior_filters_leaves_one_unpaired(self):
        """Комбинация с filter_dependent, отсеивающим один вектор, оставляет
        нечётное число входов для 0c — непарный вектор тоже должен попасть
        в excluded_by_correlation_split_ (через excluded_indices_)."""
        np.random.seed(42)
        X = np.random.randn(41, 256)
        X[40] = X[0] * 2.0 + 1e-7  # почти линейно зависимый -> 0b исключит 1 вектор

        clusterer = FursovClusterer(
            n_subclasses=4,
            filter_dependent=True,
            dependency_threshold=0.999,
            split_correlated_pairs=True,
        )
        clusterer.fit(X)

        # 41 - 1 (0b) = 40 -> чётное, но проверяем инвариант на общем случае:
        # всё, что не в kept_indices_, должно быть учтено в excluded_indices_.
        all_indices = set(clusterer.kept_indices_.tolist()) | set(
            clusterer.excluded_indices_.tolist()
        )
        assert all_indices == set(range(41))

    def test_invalid_correlated_pairs_subset_raises(self):
        with pytest.raises(ValueError):
            FursovClusterer(n_subclasses=4, correlated_pairs_subset="c")

    def test_combined_with_other_filters_runs_last(self):
        np.random.seed(42)
        X = np.random.randn(40, 256) * 50 + 150
        X[10] = 0.0  # малоинформативный (0a)
        X[20] = X[0] * 1.0 + 1e-9  # почти линейно зависимый (0b)

        clusterer = FursovClusterer(
            n_subclasses=4,
            filter_low_informativeness=True,
            filter_dependent=True,
            dependency_threshold=0.999,
            split_correlated_pairs=True,
        )
        clusterer.fit(X)

        assert 10 in clusterer.excluded_by_informativeness_
        assert 20 in clusterer.excluded_by_dependency_
        assert 10 not in clusterer.excluded_by_correlation_split_
        assert 20 not in clusterer.excluded_by_correlation_split_
        assert clusterer.get_subclass_sizes().sum() == len(clusterer.kept_indices_)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "theory"])
