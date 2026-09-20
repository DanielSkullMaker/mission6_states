"""Тесты SubspaceConjugacyClassifier (Фаза C теории, NB8).

Все тесты используют синтетические данные (fixtures из tests/conftest.py) —
это theory-тесты, проверяющие инварианты Фазы C независимо от конкретного
датасета. Проверка на реальных МРТ-данных — в tests/test_parity/.
"""

import numpy as np
import pytest

from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier

N_SUBCLASSES = 4


@pytest.fixture
def fitted_classifier(three_class_dataset) -> SubspaceConjugacyClassifier:
    X, y = three_class_dataset
    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at=2)
    clf.fit(X, y)
    return clf


@pytest.mark.theory
class TestFitUsesCanonicalClusterer:
    """Классификатор обязан использовать FursovClusterer (канон), не legacy."""

    def test_fit_produces_freeze_basis_at_2_by_default(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
        clf.fit(X, y)

        for cls in clf.classes_:
            for Y in clf.subspaces_[cls]:
                assert Y.shape == (X.shape[1], 2)

    def test_fit_matches_standalone_fursov_clusterer(self, three_class_dataset):
        """subspaces_[cls] должны совпадать с независимым FursovClusterer.fit()."""
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(
            n_subclasses=N_SUBCLASSES, freeze_basis_at=2, reg_param=1e-8
        )
        clf.fit(X, y)

        for cls in clf.classes_:
            X_cls = X[y == cls]
            reference = FursovClusterer(
                n_subclasses=N_SUBCLASSES, freeze_basis_at=2, reg_param=1e-8
            )
            reference.fit(X_cls)

            for Y_clf, Y_ref in zip(clf.subspaces_[cls], reference.subspaces_):
                np.testing.assert_array_equal(Y_clf, Y_ref)

    def test_freeze_basis_at_is_configurable(self, three_class_dataset):
        """freeze_basis_at ограничивает МАКСИМУМ размера базиса, а не задаёт
        его фиксированно для всех подклассов: B.2 — жадный argmax R, поэтому
        подклассы растут неравномерно (см. ConjugacyClusterGrowth) — часть
        подклассов может остаться с исходной парой (k=2), если ни один
        оставшийся вектор не набрал по ним максимум R ни на одной итерации.
        """
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at=3)
        clf.fit(X, y)

        for cls in clf.classes_:
            basis_sizes = [Y.shape[1] for Y in clf.subspaces_[cls]]
            assert all(2 <= size <= 3 for size in basis_sizes)
            assert max(basis_sizes) == 3  # freeze_basis_at реально применяется


@pytest.mark.theory
class TestPerClassNSubclasses:
    """n_subclasses может быть int (одинаково для всех классов, поведение по
    умолчанию) или Mapping[class_label, int] (индивидуально на класс) —
    статья Korshikov & Fursov отмечает, что оптимальное число подклассов
    обычно различается между классами патологий.
    """

    def test_dict_gives_different_subspace_counts_per_class(self, three_class_dataset):
        X, y = three_class_dataset
        n_subclasses = {0: 2, 1: 3, 2: 4}
        clf = SubspaceConjugacyClassifier(n_subclasses=n_subclasses)
        clf.fit(X, y)

        for cls, expected in n_subclasses.items():
            assert len(clf.subspaces_[cls]) == expected

    def test_n_subclasses_by_class_attribute_reflects_dict(self, three_class_dataset):
        X, y = three_class_dataset
        n_subclasses = {0: 2, 1: 3, 2: 4}
        clf = SubspaceConjugacyClassifier(n_subclasses=n_subclasses)
        clf.fit(X, y)

        assert clf.n_subclasses_by_class_ == n_subclasses

    def test_n_subclasses_by_class_attribute_reflects_scalar(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
        clf.fit(X, y)

        assert clf.n_subclasses_by_class_ == {cls: N_SUBCLASSES for cls in clf.classes_}

    def test_predict_still_works_with_per_class_dict(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses={0: 2, 1: 3, 2: 4})
        clf.fit(X, y)

        preds = clf.predict(X)
        assert set(preds).issubset({0, 1, 2})
        # total_subclasses = flat_subclass_labels_ длина = 2+3+4 = 9.
        assert len(clf.flat_subclass_labels_) == 9

    def test_missing_class_in_dict_raises_value_error(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses={0: 2, 1: 3})  # класс 2 отсутствует

        with pytest.raises(ValueError, match="2"):
            clf.fit(X, y)

    def test_extra_class_in_dict_is_ignored_not_an_error(self, three_class_dataset):
        X, y = three_class_dataset
        # "unused_class" не встречается в y — не должно ломать fit().
        clf = SubspaceConjugacyClassifier(n_subclasses={0: 2, 1: 3, 2: 4, "unused_class": 99})
        clf.fit(X, y)

        assert clf.is_fitted_
        assert "unused_class" not in clf.n_subclasses_by_class_

    def test_empty_dict_raises_value_error(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses={})

        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_non_positive_value_in_dict_raises_value_error(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses={0: 4, 1: 0, 2: 4})

        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_fit_from_subclass_bases_infers_n_subclasses_by_class(self):
        rng = np.random.default_rng(0)
        subspaces_by_class = {
            "a": [rng.standard_normal((16, 2)) for _ in range(3)],
            "b": [rng.standard_normal((16, 2)) for _ in range(5)],
        }
        clf = SubspaceConjugacyClassifier()
        clf.fit_from_subclass_bases(subspaces_by_class)

        assert clf.n_subclasses_by_class_ == {"a": 3, "b": 5}

    def test_sklearn_get_params_set_params_clone_with_dict(self, three_class_dataset):
        """SubspaceConjugacyClassifier должен оставаться sklearn-совместимым
        (get_params/set_params/clone), даже когда n_subclasses — словарь, а
        не int — это то, что позволяет grid_search_classifier/
        random_search_classifier перебирать словарные значения как обычную
        точку в param_grid (см. model_selection/search.py)."""
        from sklearn.base import clone

        n_subclasses = {0: 2, 1: 3, 2: 4}
        clf = SubspaceConjugacyClassifier(n_subclasses=n_subclasses)

        assert clf.get_params()["n_subclasses"] == n_subclasses

        cloned = clone(clf)
        assert cloned.get_params()["n_subclasses"] == n_subclasses

        X, y = three_class_dataset
        cloned.fit(X, y)
        assert cloned.n_subclasses_by_class_ == n_subclasses


@pytest.mark.theory
class TestFreezeBasisAtAuto:
    """freeze_basis_at="auto" — не отбрасывает рост B.2: каждый класс растёт
    без ограничения, затем все подпространства усекаются до общего
    минимального фактически достигнутого k (equalize_subspace_bases) —
    буквальный рецепт статьи Korshikov & Fursov (refactoring_plan.txt,
    раздел 10, находка №2), в отличие от freeze_basis_at=int (по умолчанию
    2), который отбрасывает всё, кроме исходной пары из фазы B.1.
    """

    def test_auto_produces_equal_sized_bases_across_all_classes(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at="auto")
        clf.fit(X, y)

        all_sizes = {
            Y.shape[1] for bases in clf.subspaces_.values() for Y in bases
        }
        assert len(all_sizes) == 1  # все базисы одного размера после равнения
        assert clf.equalized_basis_size_ == next(iter(all_sizes))

    def test_auto_basis_size_is_at_least_default_two(self, three_class_dataset):
        """При freeze_basis_at="auto" минимальный итоговый размер не может
        быть меньше 2 — это исходная пара из фазы B.1, до которой рост
        ниоткуда не может опуститься."""
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at="auto")
        clf.fit(X, y)

        assert clf.equalized_basis_size_ >= 2

    def test_equalized_basis_size_is_none_for_int_freeze_basis_at(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at=2)
        clf.fit(X, y)

        assert clf.equalized_basis_size_ is None

    def test_auto_matches_independent_growth_minimum(self, three_class_dataset):
        """Регрессия на находку №2: freeze_basis_at=2 всегда даёт ровно 2
        (исходную пару B.1, отбрасывая рост B.2). freeze_basis_at="auto"
        должен ТОЧНО воспроизводить: (a) каждый класс растится независимо
        без ограничения (как FursovClusterer(freeze_basis_at=None)), (b)
        итоговый equalized_basis_size_ — глобальный минимум среди
        фактически достигнутых размеров по ВСЕМ классам и подклассам сразу
        — не привязан к конкретному числу (может совпасть и с 2, если рост
        неравномерный), поэтому проверяем не неравенство, а точное
        совпадение с независимо посчитанным эталоном."""
        X, y = three_class_dataset

        expected_sizes = []
        for cls in np.unique(y):
            reference = FursovClusterer(n_subclasses=N_SUBCLASSES, freeze_basis_at=None)
            reference.fit(X[y == cls])
            expected_sizes.extend(Y.shape[1] for Y in reference.subspaces_)
        expected_min = min(expected_sizes)

        clf_auto = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at="auto")
        clf_auto.fit(X, y)

        assert clf_auto.equalized_basis_size_ == expected_min
        assert max(expected_sizes) > expected_min, (
            "Тест неинформативен: рост во всех подклассах остановился на "
            "одинаковом размере — увеличьте N_SUBCLASSES/объём данных."
        )

    def test_predict_and_predict_proba_work_with_auto(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at="auto")
        clf.fit(X, y)

        preds = clf.predict(X)
        assert set(preds).issubset(set(clf.classes_))

        probs = clf.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)

    def test_invalid_freeze_basis_at_raises(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=2, freeze_basis_at="not_a_valid_value")

        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_freeze_basis_at_int_below_two_raises(self, three_class_dataset):
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=2, freeze_basis_at=1)

        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_none_grows_freely_without_cross_class_equalization(self, three_class_dataset):
        """freeze_basis_at=None (в отличие от "auto") НЕ равняет размеры
        между классами — оставляет естественно разные k, как обычный
        FursovClusterer(freeze_basis_at=None)."""
        X, y = three_class_dataset
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at=None)
        clf.fit(X, y)

        assert clf.equalized_basis_size_ is None
        assert clf.is_fitted_


class TestFitFromSubclassBasesEqualize:
    """fit_from_subclass_bases(equalize=True) — тот же механизм равнения,
    что и fit(freeze_basis_at="auto"), но для уже готовых базисов
    (например, из FursovPipeline.build_classifier(equalize=True))."""

    def test_equalize_true_truncates_unequal_bases(self):
        rng = np.random.default_rng(0)
        subspaces_by_class = {
            "a": [rng.standard_normal((16, 5)), rng.standard_normal((16, 3))],
            "b": [rng.standard_normal((16, 7))],
        }
        clf = SubspaceConjugacyClassifier()
        clf.fit_from_subclass_bases(subspaces_by_class, equalize=True)

        assert clf.equalized_basis_size_ == 3
        assert all(Y.shape[1] == 3 for bases in clf.subspaces_.values() for Y in bases)

    def test_equalize_false_keeps_bases_as_is(self):
        rng = np.random.default_rng(0)
        subspaces_by_class = {
            "a": [rng.standard_normal((16, 5))],
            "b": [rng.standard_normal((16, 3))],
        }
        clf = SubspaceConjugacyClassifier()
        clf.fit_from_subclass_bases(subspaces_by_class, equalize=False)

        assert clf.equalized_basis_size_ is None
        assert clf.subspaces_["a"][0].shape[1] == 5
        assert clf.subspaces_["b"][0].shape[1] == 3


@pytest.mark.theory
class TestFilterDependent:
    """filter_dependent=True — исключение почти линейно зависимых
    эталонных векторов НЕЗАВИСИМО в каждом классе перед его кластеризацией
    (Korshikov & Fursov, "Problem Definition"; refactoring_plan.txt,
    раздел 10, находка №3). N существенно больше M на класс — как в
    реальном сценарии, чтобы не задевать естественное насыщение фильтра
    (см. Notes в algorithms/reference_filter.py)."""

    @staticmethod
    def _three_class_data_with_duplicates():
        rng = np.random.default_rng(0)
        n_features = 256
        X_list, y_list = [], []
        for idx, label in enumerate(["glioma", "meningioma", "pituitary"]):
            X_cls = rng.standard_normal((20, n_features)) + idx * 4.0
            X_list.append(X_cls)
            y_list.append(np.full(20, label))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        # Почти-дубликат внутри "glioma" (индекс 5 внутри класса, глобально 5)
        # и внутри "pituitary" (индекс 5 внутри класса, глобально 45).
        X[5] = X[0] * 2.0 + 1e-7
        X[45] = X[40] * 1.7 + 1e-7
        return X, y

    def test_disabled_by_default_no_exclusions(self):
        X, y = self._three_class_data_with_duplicates()
        clf = SubspaceConjugacyClassifier(n_subclasses=4)
        clf.fit(X, y)

        assert clf.excluded_indices_by_class_ is None

    def test_excluded_indices_mapped_to_global_x_indices(self):
        X, y = self._three_class_data_with_duplicates()
        clf = SubspaceConjugacyClassifier(
            n_subclasses=4, filter_dependent=True, dependency_threshold=0.999
        )
        clf.fit(X, y)

        assert 5 in clf.excluded_indices_by_class_["glioma"]
        assert 45 in clf.excluded_indices_by_class_["pituitary"]
        # meningioma не трогали дубликатами -> обычно ничего не исключено.
        assert set(clf.excluded_indices_by_class_.keys()) == {
            "glioma", "meningioma", "pituitary",
        }

    def test_predict_and_predict_proba_still_work(self):
        X, y = self._three_class_data_with_duplicates()
        clf = SubspaceConjugacyClassifier(
            n_subclasses=4, filter_dependent=True, dependency_threshold=0.999
        )
        clf.fit(X, y)

        preds = clf.predict(X)
        assert set(preds).issubset({"glioma", "meningioma", "pituitary"})

        probs = clf.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)

    def test_combined_with_freeze_basis_at_auto(self):
        """filter_dependent и freeze_basis_at="auto" — независимые оси,
        должны компоноваться без конфликтов."""
        X, y = self._three_class_data_with_duplicates()
        clf = SubspaceConjugacyClassifier(
            n_subclasses=4,
            freeze_basis_at="auto",
            filter_dependent=True,
            dependency_threshold=0.999,
        )
        clf.fit(X, y)

        assert clf.equalized_basis_size_ is not None
        assert 5 in clf.excluded_indices_by_class_["glioma"]
        all_sizes = {Y.shape[1] for bases in clf.subspaces_.values() for Y in bases}
        assert len(all_sizes) == 1


class TestFilterLowInformativeness:
    """filter_low_informativeness=True — исключение малоинформативных
    эталонных векторов НЕЗАВИСИМО в каждом классе перед его кластеризацией
    (Korshikov & Fursov, 3-й эксперимент; refactoring_plan.txt, раздел 10,
    находка №5)."""

    @staticmethod
    def _three_class_data_with_low_informativeness():
        rng = np.random.default_rng(0)
        n_features = 256
        X_list, y_list = [], []
        for idx, label in enumerate(["glioma", "meningioma", "pituitary"]):
            X_cls = rng.uniform(100, 255, size=(20, n_features))
            X_list.append(X_cls)
            y_list.append(np.full(20, label))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        # Малоинформативный (почти чёрный) вектор внутри "glioma" (глобально
        # индекс 5) и внутри "pituitary" (глобально индекс 45).
        X[5] = 0.0
        X[45] = 0.0
        return X, y

    def test_disabled_by_default_no_exclusions(self):
        X, y = self._three_class_data_with_low_informativeness()
        clf = SubspaceConjugacyClassifier(n_subclasses=4)
        clf.fit(X, y)

        assert clf.excluded_indices_by_class_ is None

    def test_excluded_indices_mapped_to_global_x_indices(self):
        X, y = self._three_class_data_with_low_informativeness()
        clf = SubspaceConjugacyClassifier(
            n_subclasses=4, filter_low_informativeness=True,
        )
        clf.fit(X, y)

        assert 5 in clf.excluded_indices_by_class_["glioma"]
        assert 45 in clf.excluded_indices_by_class_["pituitary"]
        assert set(clf.excluded_indices_by_class_.keys()) == {
            "glioma", "meningioma", "pituitary",
        }

    def test_predict_and_predict_proba_still_work(self):
        X, y = self._three_class_data_with_low_informativeness()
        clf = SubspaceConjugacyClassifier(
            n_subclasses=4, filter_low_informativeness=True,
        )
        clf.fit(X, y)

        preds = clf.predict(X)
        assert set(preds).issubset({"glioma", "meningioma", "pituitary"})

        probs = clf.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)

    def test_combined_with_filter_dependent(self):
        """filter_low_informativeness и filter_dependent — независимые оси,
        должны компоноваться без конфликтов (0a затем 0b, см.
        fursov_clusterer.py)."""
        X, y = self._three_class_data_with_low_informativeness()
        X[7] = X[6] * 1.0 + 1e-9  # почти линейно зависимый внутри glioma
        # (LinearDependencyFilter обрабатывает векторы по порядку индексов и
        # исключает ПОЗДНИЙ из почти-дублирующей пары, см. reference_filter.py)

        clf = SubspaceConjugacyClassifier(
            n_subclasses=4,
            filter_low_informativeness=True,
            filter_dependent=True,
            dependency_threshold=0.999,
        )
        clf.fit(X, y)

        assert 5 in clf.excluded_indices_by_class_["glioma"]
        assert 7 in clf.excluded_indices_by_class_["glioma"]

    def test_combined_with_freeze_basis_at_auto(self):
        X, y = self._three_class_data_with_low_informativeness()
        clf = SubspaceConjugacyClassifier(
            n_subclasses=4,
            freeze_basis_at="auto",
            filter_low_informativeness=True,
        )
        clf.fit(X, y)

        assert clf.equalized_basis_size_ is not None
        assert 5 in clf.excluded_indices_by_class_["glioma"]
        all_sizes = {Y.shape[1] for bases in clf.subspaces_.values() for Y in bases}
        assert len(all_sizes) == 1


@pytest.mark.theory
class TestSplitCorrelatedPairs:
    """split_correlated_pairs=True — разбиение эталонных векторов КАЖДОГО
    класса на два подмножества похожих пар перед его кластеризацией.
    Источник — ЧЕРНОВИК другой, неопубликованной статьи (theory/Макет новой
    статьи.docx, "Первый этап"), НЕ проверенная публикация Korshikov & Fursov
    (в отличие от filter_dependent/filter_low_informativeness)."""

    @staticmethod
    def _three_class_data():
        rng = np.random.default_rng(0)
        n_features = 256
        X_list, y_list = [], []
        for idx, label in enumerate(["glioma", "meningioma", "pituitary"]):
            X_cls = rng.standard_normal((20, n_features)) + idx * 4.0
            X_list.append(X_cls)
            y_list.append(np.full(20, label))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y

    def test_disabled_by_default_no_exclusions(self):
        X, y = self._three_class_data()
        clf = SubspaceConjugacyClassifier(n_subclasses=4)
        clf.fit(X, y)

        assert clf.excluded_indices_by_class_ is None

    def test_enabled_excludes_half_of_each_class(self):
        X, y = self._three_class_data()
        clf = SubspaceConjugacyClassifier(
            n_subclasses=4, split_correlated_pairs=True, correlated_pairs_subset="a",
        )
        clf.fit(X, y)

        assert set(clf.excluded_indices_by_class_.keys()) == {
            "glioma", "meningioma", "pituitary",
        }
        for cls in ("glioma", "meningioma", "pituitary"):
            assert len(clf.excluded_indices_by_class_[cls]) == 10

    def test_predict_and_predict_proba_still_work(self):
        X, y = self._three_class_data()
        clf = SubspaceConjugacyClassifier(
            n_subclasses=4, split_correlated_pairs=True,
        )
        clf.fit(X, y)

        preds = clf.predict(X)
        assert set(preds).issubset({"glioma", "meningioma", "pituitary"})

        probs = clf.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)

    def test_subset_a_and_subset_b_give_disjoint_exclusions(self):
        X, y = self._three_class_data()
        clf_a = SubspaceConjugacyClassifier(
            n_subclasses=4, split_correlated_pairs=True, correlated_pairs_subset="a",
        )
        clf_a.fit(X, y)
        clf_b = SubspaceConjugacyClassifier(
            n_subclasses=4, split_correlated_pairs=True, correlated_pairs_subset="b",
        )
        clf_b.fit(X, y)

        for cls in ("glioma", "meningioma", "pituitary"):
            excluded_a = set(clf_a.excluded_indices_by_class_[cls].tolist())
            excluded_b = set(clf_b.excluded_indices_by_class_[cls].tolist())
            assert excluded_a & excluded_b == set()

    def test_combined_with_other_filters(self):
        X, y = self._three_class_data()
        clf = SubspaceConjugacyClassifier(
            n_subclasses=4,
            filter_low_informativeness=False,
            filter_dependent=True,
            dependency_threshold=0.999,
            split_correlated_pairs=True,
        )
        clf.fit(X, y)
        assert clf.excluded_indices_by_class_ is not None


@pytest.mark.theory
class TestPhaseCFlatArgmax:
    """Теория C: R_{c,s} для 24 (n_classes*n_subclasses) базисов, flat argmax."""

    def test_flat_subclass_labels_length(self, fitted_classifier):
        clf = fitted_classifier
        n_classes = len(clf.classes_)
        assert len(clf.flat_subclass_labels_) == n_classes * N_SUBCLASSES

    def test_r_matrix_flat_shape(self, fitted_classifier, three_class_dataset):
        X, _ = three_class_dataset
        clf = fitted_classifier
        R_flat = clf.predict_r_matrix_flat(X)

        assert R_flat.shape == (X.shape[0], len(clf.flat_subclass_labels_))
        assert np.all(R_flat >= 0.0) and np.all(R_flat <= 1.0)

    def test_r_matrix_per_class_equals_max_of_flat(self, fitted_classifier, three_class_dataset):
        """predict_r_matrix (max по подклассам класса) должен совпасть с
        поклассовым максимумом predict_r_matrix_flat — это одна и та же
        Фаза C, посчитанная двумя разными агрегациями."""
        X, _ = three_class_dataset
        clf = fitted_classifier

        R_per_class = clf.predict_r_matrix(X)
        R_flat = clf.predict_r_matrix_flat(X)

        for cls_idx, cls in enumerate(clf.classes_):
            mask = clf.flat_subclass_labels_ == cls
            expected = np.max(R_flat[:, mask], axis=1)
            np.testing.assert_allclose(R_per_class[:, cls_idx], expected)

    def test_predict_subclass_owner_matches_predict(self, fitted_classifier, three_class_dataset):
        """Класс-владелец лучшего плоского подкласса == predict().

        Теория C: subclass* = argmax_{c,s} R_{c,s}; class* = c(subclass*).
        Это ДОЛЖНО совпадать с classifier.predict(), который берёт
        argmax по максимумам R внутри каждого класса — это два
        математически эквивалентных способа выразить один и тот же
        плоский argmax (argmax группового максимума == группа глобального
        максимума).
        """
        X, _ = three_class_dataset
        clf = fitted_classifier

        flat_idx = clf.predict_subclass(X)
        owner_classes = clf.flat_subclass_labels_[flat_idx]
        predicted_classes = clf.predict(X)

        np.testing.assert_array_equal(owner_classes, predicted_classes)

    def test_predict_subclass_range(self, fitted_classifier, three_class_dataset):
        X, _ = three_class_dataset
        clf = fitted_classifier
        flat_idx = clf.predict_subclass(X)

        assert np.all(flat_idx >= 0)
        assert np.all(flat_idx < len(clf.flat_subclass_labels_))


@pytest.mark.theory
class TestConfidenceRatio:
    """NB8: proportion = best_R / mean(R_others) - 1."""

    def test_confidence_ratio_matches_manual_formula(self, fitted_classifier, three_class_dataset):
        X, _ = three_class_dataset
        clf = fitted_classifier

        R_flat = clf.predict_r_matrix_flat(X)
        confidence = clf.predict_confidence_ratio(X)

        best = np.max(R_flat, axis=1)
        mean_others = (np.sum(R_flat, axis=1) - best) / (R_flat.shape[1] - 1)
        expected = best / mean_others - 1

        np.testing.assert_allclose(confidence, expected, rtol=1e-9)

    def test_confidence_ratio_shape(self, fitted_classifier, three_class_dataset):
        X, _ = three_class_dataset
        confidence = fitted_classifier.predict_confidence_ratio(X)
        assert confidence.shape == (X.shape[0],)

    def test_confidence_ratio_requires_multiple_subspaces(self):
        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
        clf.fit_from_subclass_bases({"only_class": [np.random.randn(16, 2)]})

        with pytest.raises(ValueError):
            clf.predict_confidence_ratio(np.random.randn(3, 16))


class TestFitFromSubclassBases:
    """fit_from_subclass_bases: сборка классификатора без повторной кластеризации."""

    def test_roundtrip_matches_fit(self, three_class_dataset):
        X, y = three_class_dataset

        clf_fit = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, freeze_basis_at=2)
        clf_fit.fit(X, y)

        clf_loaded = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
        clf_loaded.fit_from_subclass_bases(clf_fit.subspaces_)

        assert clf_loaded.is_fitted_
        assert set(clf_loaded.classes_) == set(clf_fit.classes_)
        np.testing.assert_array_equal(
            clf_fit.predict(X), clf_loaded.predict(X)
        )
        np.testing.assert_allclose(
            clf_fit.predict_r_matrix(X), clf_loaded.predict_r_matrix(X)
        )

    def test_rejects_empty_dict(self):
        clf = SubspaceConjugacyClassifier()
        with pytest.raises(ValueError):
            clf.fit_from_subclass_bases({})

    def test_rejects_mismatched_feature_dimension(self):
        clf = SubspaceConjugacyClassifier()
        with pytest.raises(ValueError):
            clf.fit_from_subclass_bases({
                "a": [np.random.randn(16, 2)],
                "b": [np.random.randn(32, 2)],
            })

    def test_sets_n_features_in(self):
        clf = SubspaceConjugacyClassifier()
        clf.fit_from_subclass_bases({
            "a": [np.random.randn(16, 2), np.random.randn(16, 2)],
            "b": [np.random.randn(16, 2), np.random.randn(16, 2)],
        })
        assert clf.n_features_in_ == 16


class TestPredictProba:
    def test_probabilities_sum_to_one(self, fitted_classifier, three_class_dataset):
        X, _ = three_class_dataset
        probs = fitted_classifier.predict_proba(X)

        assert probs.shape == (X.shape[0], len(fitted_classifier.classes_))
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-9)
        assert np.all(probs >= 0.0)


class TestValidationErrors:
    def test_fit_requires_at_least_two_classes(self):
        clf = SubspaceConjugacyClassifier(n_subclasses=2)
        X = np.random.randn(20, 16)
        y = np.full(20, "only_one_class")

        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_fit_requires_enough_samples_per_class(self):
        clf = SubspaceConjugacyClassifier(n_subclasses=8)
        X = np.random.randn(10, 16)
        y = np.array(["a"] * 5 + ["b"] * 5)

        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_predict_before_fit_raises(self):
        clf = SubspaceConjugacyClassifier(n_subclasses=4)
        with pytest.raises(RuntimeError):
            clf.predict(np.random.randn(3, 16))

    def test_predict_subclass_before_fit_raises(self):
        clf = SubspaceConjugacyClassifier(n_subclasses=4)
        with pytest.raises(RuntimeError):
            clf.predict_subclass(np.random.randn(3, 16))


@pytest.mark.theory
class TestClassificationAccuracyOnSeparatedClusters:
    """С хорошо разделёнными классами classifier должен предсказывать точно.

    Не количественный порог accuracy (это делает test_parity на реальных
    данных) — здесь просто sanity-check, что вся Фаза C работает end-to-end
    и не деградирует до случайного угадывания на тривиально разделимых
    данных.
    """

    def test_high_accuracy_on_well_separated_classes(self, random_seed):
        """Классы должны отличаться НАПРАВЛЕНИЕМ, а не только величиной сдвига.

        R(x, Y) масштабно-инвариантен (см. test_r_is_scale_invariant в
        test_clustering_theory.py): R(alpha*x, Y) == R(x, Y). Поэтому просто
        добавить разным классам разный константный сдвиг вдоль ОДНОГО и того
        же направления не создаёт разделимости для этого метода — нужны
        разные направления в пространстве признаков.
        """
        np.random.seed(random_seed)
        n_per_class = 40
        n_features = 32
        n_classes = 3

        class_directions = np.random.randn(n_classes, n_features)
        class_directions /= np.linalg.norm(class_directions, axis=1, keepdims=True)

        X_list, y_list = [], []
        for idx in range(n_classes):
            magnitudes = np.random.uniform(8.0, 12.0, size=(n_per_class, 1))
            noise = np.random.randn(n_per_class, n_features) * 0.5
            X_list.append(magnitudes * class_directions[idx] + noise)
            y_list.append(np.full(n_per_class, idx))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        clf = SubspaceConjugacyClassifier(n_subclasses=4, freeze_basis_at=2)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        accuracy = np.mean(y_pred == y)

        assert accuracy >= 0.9
