"""Тесты evaluation/paper_methodology.py — методология статьи (Table I/II, Fig. 5/7).

Реализует раздел "Analysis of Experimental Results" статьи Korshikov & Fursov
на синтетических данных (быстро, без реального датасета — см. tests/test_theory/
для конвенции). Отдельно проверяет защиту от утечки, мотивированную находкой
в theory/code/Run_200.m (см. docstring модуля paper_methodology.py).
"""

import numpy as np
import pytest

from subspace_conjugacy.evaluation.paper_methodology import (
    compare_informativeness_filter,
    find_best_n_subclasses,
    paired_correlation_split,
    paper_train_test_split,
)


def _separable_dataset(random_seed: int = 42, n_per_class: int = 60, n_features: int = 32):
    """3 класса, разделимых по НАПРАВЛЕНИЮ (не по константному сдвигу).

    R(x, Y) масштабно-инвариантен (test_r_is_scale_invariant,
    test_clustering_theory.py) — константный сдвиг вдоль одного направления
    не создаёт разделимости для этого метода (см.
    test_classifier.py::test_high_accuracy_on_well_separated_classes,
    тот же паттерн генерации данных использован здесь для консистентности).
    """
    rng = np.random.default_rng(random_seed)
    labels = ["glioma", "meningioma", "pituitary"]

    class_directions = rng.standard_normal((len(labels), n_features))
    class_directions /= np.linalg.norm(class_directions, axis=1, keepdims=True)

    X_list, y_list = [], []
    for idx, label in enumerate(labels):
        magnitudes = rng.uniform(8.0, 12.0, size=(n_per_class, 1))
        noise = rng.standard_normal((n_per_class, n_features)) * 0.5
        X_list.append(magnitudes * class_directions[idx] + noise)
        y_list.append(np.full(n_per_class, label))

    return np.vstack(X_list), np.concatenate(y_list)


@pytest.mark.theory
class TestPaperTrainTestSplit:
    def test_split_proportions(self):
        X, y = _separable_dataset()
        X_train, X_test, y_train, y_test = paper_train_test_split(
            X, y, train_fraction=0.8, random_state=0
        )
        total = len(y)
        assert len(y_train) + len(y_test) == total
        assert len(y_train) == pytest.approx(0.8 * total, abs=3)

    def test_stratified_by_class(self):
        X, y = _separable_dataset()
        X_train, X_test, y_train, y_test = paper_train_test_split(
            X, y, train_fraction=0.8, random_state=0
        )
        train_counts = {cls: int(np.sum(y_train == cls)) for cls in np.unique(y)}
        test_counts = {cls: int(np.sum(y_test == cls)) for cls in np.unique(y)}
        # Стратификация — примерно равные доли по классам (60 на класс, 80/20).
        for cls in np.unique(y):
            assert train_counts[cls] == pytest.approx(48, abs=2)
            assert test_counts[cls] == pytest.approx(12, abs=2)

    def test_no_index_overlap_between_train_and_test(self):
        """Прямая проверка защиты от утечки (мотивация — theory/code/Run_200.m):
        ни один X_test не должен быть строкой X_train."""
        X, y = _separable_dataset(n_per_class=20, n_features=8)
        X_train, X_test, y_train, y_test = paper_train_test_split(
            X, y, train_fraction=0.8, random_state=1
        )
        train_rows = {tuple(row) for row in X_train}
        test_rows = {tuple(row) for row in X_test}
        assert train_rows.isdisjoint(test_rows)

    def test_reproducible_with_random_state(self):
        X, y = _separable_dataset()
        split_a = paper_train_test_split(X, y, random_state=7)
        split_b = paper_train_test_split(X, y, random_state=7)
        for arr_a, arr_b in zip(split_a, split_b):
            np.testing.assert_array_equal(arr_a, arr_b)

    def test_different_train_fraction(self):
        X, y = _separable_dataset()
        X_train, X_test, y_train, y_test = paper_train_test_split(
            X, y, train_fraction=0.5, random_state=0
        )
        assert len(y_train) == pytest.approx(len(y_test), abs=3)


@pytest.mark.theory
class TestFindBestNSubclasses:
    def test_returns_expected_keys(self):
        X, y = _separable_dataset(n_per_class=60)
        X_train, X_test, y_train, y_test = paper_train_test_split(X, y, random_state=0)

        result = find_best_n_subclasses(
            X_train, y_train, X_test, y_test, n_subclasses_grid=(4, 8)
        )

        assert set(result.keys()) == {
            "best_n_subclasses",
            "best_mean_per_class_accuracy",
            "best_estimator_",
            "all_results",
        }
        assert len(result["all_results"]) == 2
        assert result["best_n_subclasses"] in (4, 8)

    def test_best_is_actually_the_maximum(self):
        X, y = _separable_dataset(n_per_class=60)
        X_train, X_test, y_train, y_test = paper_train_test_split(X, y, random_state=0)

        result = find_best_n_subclasses(
            X_train, y_train, X_test, y_test, n_subclasses_grid=(4, 6, 8)
        )

        max_mean = max(r["mean_per_class_accuracy"] for r in result["all_results"])
        assert result["best_mean_per_class_accuracy"] == pytest.approx(max_mean)

    def test_best_estimator_is_fitted_and_usable(self):
        X, y = _separable_dataset(n_per_class=60)
        X_train, X_test, y_train, y_test = paper_train_test_split(X, y, random_state=0)

        result = find_best_n_subclasses(
            X_train, y_train, X_test, y_test, n_subclasses_grid=(4, 8)
        )

        preds = result["best_estimator_"].predict(X_test)
        assert len(preds) == len(y_test)

    def test_classifier_kwargs_passthrough(self):
        X, y = _separable_dataset(n_per_class=60)
        X_train, X_test, y_train, y_test = paper_train_test_split(X, y, random_state=0)

        result = find_best_n_subclasses(
            X_train, y_train, X_test, y_test,
            n_subclasses_grid=(4,),
            classifier_kwargs={"freeze_basis_at": 3},
        )
        # freeze_basis_at=3 задаёт ВЕРХНЮЮ ГРАНИЦУ размера базиса — B.2
        # наполняет подклассы неравномерно (задокументированная особенность
        # алгоритма, см. README), поэтому не каждый подкласс обязательно
        # дорастёт до k=3 до усечения. Проверяем, что хотя бы один подкласс
        # достиг k=3 (подтверждает, что параметр реально применился, а не
        # остался дефолтным k=2).
        all_widths = [
            Y.shape[1]
            for bases in result["best_estimator_"].subspaces_.values()
            for Y in bases
        ]
        assert max(all_widths) == 3
        assert all(w <= 3 for w in all_widths)

    def test_accuracy_reasonable_on_well_separated_data(self):
        """На хорошо разделённых синтетических классах точность должна быть высокой."""
        X, y = _separable_dataset(n_per_class=60, n_features=32)
        X_train, X_test, y_train, y_test = paper_train_test_split(X, y, random_state=0)

        result = find_best_n_subclasses(
            X_train, y_train, X_test, y_test, n_subclasses_grid=(4, 8)
        )
        assert result["best_mean_per_class_accuracy"] > 0.7


@pytest.mark.theory
class TestCompareInformativenessFilter:
    def test_returns_both_tables(self):
        X, y = _separable_dataset(n_per_class=60)
        X_train, X_test, y_train, y_test = paper_train_test_split(X, y, random_state=0)

        comparison = compare_informativeness_filter(
            X_train, y_train, X_test, y_test, n_subclasses_grid=(4, 8)
        )

        assert set(comparison.keys()) == {"table_I_baseline", "table_II_filtered"}
        for table in comparison.values():
            assert "best_n_subclasses" in table
            assert "best_mean_per_class_accuracy" in table

    def test_rejects_conflicting_classifier_kwargs(self):
        X, y = _separable_dataset(n_per_class=60)
        X_train, X_test, y_train, y_test = paper_train_test_split(X, y, random_state=0)

        with pytest.raises(ValueError):
            compare_informativeness_filter(
                X_train, y_train, X_test, y_test,
                n_subclasses_grid=(4,),
                classifier_kwargs={"filter_low_informativeness": True},
            )


@pytest.mark.theory
class TestPairedCorrelationSplit:
    """Реализация theory/code/Вариант_2_1.docx, "Первый этап" (совет
    научного руководителя: пары наиболее скоррелированных изображений
    разносятся между train и test)."""

    def test_every_index_assigned_exactly_once(self):
        X, y = _separable_dataset(n_per_class=20, n_features=16)
        result = paired_correlation_split(X, y, random_state=0)

        all_assigned = sorted(
            result["train_indices"].tolist() + result["test_indices"].tolist()
        )
        assert all_assigned == list(range(len(y)))

    def test_no_overlap_between_train_and_test(self):
        X, y = _separable_dataset(n_per_class=20, n_features=16)
        result = paired_correlation_split(X, y, random_state=0)

        train_set = set(result["train_indices"].tolist())
        test_set = set(result["test_indices"].tolist())
        assert train_set.isdisjoint(test_set)

    def test_even_class_size_splits_roughly_in_half(self):
        X, y = _separable_dataset(n_per_class=20, n_features=16)  # чётный размер класса
        result = paired_correlation_split(X, y, random_state=0)

        for cls in np.unique(y):
            n_train_cls = int(np.sum(result["y_train"] == cls))
            n_test_cls = int(np.sum(result["y_test"] == cls))
            assert n_train_cls == 10
            assert n_test_cls == 10

    def test_odd_class_size_extra_goes_to_train(self):
        X, y = _separable_dataset(n_per_class=21, n_features=16)  # нечётный размер класса
        result = paired_correlation_split(X, y, random_state=0)

        for cls in np.unique(y):
            n_train_cls = int(np.sum(result["y_train"] == cls))
            n_test_cls = int(np.sum(result["y_test"] == cls))
            assert n_train_cls == 11
            assert n_test_cls == 10

    def test_pairs_are_disjoint_and_within_class(self):
        X, y = _separable_dataset(n_per_class=20, n_features=16)
        result = paired_correlation_split(X, y, random_state=0)

        seen = set()
        for train_idx, test_idx in result["pairs"]:
            assert train_idx not in seen and test_idx not in seen
            seen.add(train_idx)
            seen.add(test_idx)
            assert y[train_idx] == y[test_idx]  # пары формируются внутри класса

    def test_most_correlated_pair_is_split_across_train_and_test(self):
        """Ключевая проверка методологии: почти дубликат обязан оказаться
        по РАЗНЫЕ стороны сплита (train/test), а не вместе."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 32))
        X[7] = X[2] * 2.0 + rng.standard_normal(32) * 1e-6  # почти дубликат X[2]
        y = np.array(["glioma"] * 10)

        result = paired_correlation_split(X, y, random_state=0)

        in_train = 2 in result["train_indices"].tolist()
        in_test_7 = 7 in result["test_indices"].tolist()
        in_test = 2 in result["test_indices"].tolist()
        in_train_7 = 7 in result["train_indices"].tolist()
        # Ровно один из пары (2, 7) — в train, другой — в test.
        assert (in_train and in_test_7 and not in_test and not in_train_7) or (
            in_test and in_train_7 and not in_train and not in_test_7
        )

    def test_reproducible_with_random_state(self):
        X, y = _separable_dataset(n_per_class=20, n_features=16)
        result_a = paired_correlation_split(X, y, random_state=5)
        result_b = paired_correlation_split(X, y, random_state=5)

        np.testing.assert_array_equal(result_a["train_indices"], result_b["train_indices"])
        np.testing.assert_array_equal(result_a["test_indices"], result_b["test_indices"])
        assert result_a["pairs"] == result_b["pairs"]

    def test_returned_arrays_match_indices(self):
        X, y = _separable_dataset(n_per_class=20, n_features=16)
        result = paired_correlation_split(X, y, random_state=0)

        np.testing.assert_array_equal(result["X_train"], X[result["train_indices"]])
        np.testing.assert_array_equal(result["y_train"], y[result["train_indices"]])
        np.testing.assert_array_equal(result["X_test"], X[result["test_indices"]])
        np.testing.assert_array_equal(result["y_test"], y[result["test_indices"]])

    def test_inflates_accuracy_relative_to_independent_split(self):
        """Демонстрирует саму суть предупреждения из докстринга: accuracy на
        paired-сплите ожидаемо выше (или как минимум не хуже), чем на
        независимом paper_train_test_split, т.к. test специально подобран
        похожим на train."""
        X, y = _separable_dataset(n_per_class=60, n_features=32)

        paired = paired_correlation_split(X, y, random_state=0)
        honest_result = find_best_n_subclasses(
            paired["X_train"], paired["y_train"], paired["X_test"], paired["y_test"],
            n_subclasses_grid=(4,),
        )

        X_train_i, X_test_i, y_train_i, y_test_i = paper_train_test_split(
            X, y, train_fraction=0.5, random_state=0
        )
        independent_result = find_best_n_subclasses(
            X_train_i, y_train_i, X_test_i, y_test_i, n_subclasses_grid=(4,),
        )

        assert paired["X_train"].shape[0] == pytest.approx(X_train_i.shape[0], abs=3)
        # Не строгое неравенство (синтетика может дать потолок 1.0 в обоих
        # случаях) — но paired не должен быть хуже независимого сплита.
        assert (
            honest_result["best_mean_per_class_accuracy"]
            >= independent_result["best_mean_per_class_accuracy"] - 1e-9
        )
