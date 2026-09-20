"""Тесты algorithms/correlated_pair_splitter.py — CorrelatedPairSplitter.

Реализует "Первый этап" из ЧЕРНОВИКА другой (неопубликованной) статьи
(theory/Макет новой статьи.docx) — итеративное разбиение множества векторов
класса на два подмножества похожих (по косинусному сходству) пар. В отличие
от tests/test_algorithms/test_reference_filter.py (LinearDependencyFilter,
проверенная публикация), это экспериментальная возможность — тесты проверяют
структурные инварианты алгоритма, а не соответствие целевой метрике из
статьи (которой для этого черновика попросту нет).
"""

import numpy as np
import pytest

from subspace_conjugacy.algorithms.correlated_pair_splitter import CorrelatedPairSplitter


@pytest.mark.theory
class TestCorrelatedPairSplitterBasics:
    def test_even_input_splits_into_equal_halves(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 64))

        splitter = CorrelatedPairSplitter()
        splitter.fit(X)

        assert len(splitter.subset_a_indices_) == 10
        assert len(splitter.subset_b_indices_) == 10
        assert splitter.unpaired_index_ is None
        assert len(splitter.pairs_) == 10

    def test_subsets_partition_all_indices_without_overlap(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((16, 32))

        splitter = CorrelatedPairSplitter()
        splitter.fit(X)

        a = set(splitter.subset_a_indices_.tolist())
        b = set(splitter.subset_b_indices_.tolist())
        assert a & b == set()
        assert a | b == set(range(16))

    def test_pairs_match_subset_assignment(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((12, 32))

        splitter = CorrelatedPairSplitter()
        splitter.fit(X)

        pair_a = set(splitter.pairs_[:, 0].tolist())
        pair_b = set(splitter.pairs_[:, 1].tolist())
        assert pair_a == set(splitter.subset_a_indices_.tolist())
        assert pair_b == set(splitter.subset_b_indices_.tolist())

    def test_odd_input_leaves_one_vector_unpaired(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((15, 32))

        splitter = CorrelatedPairSplitter()
        splitter.fit(X)

        assert splitter.unpaired_index_ is not None
        assert len(splitter.subset_a_indices_) == 7
        assert len(splitter.subset_b_indices_) == 7

        all_used = (
            set(splitter.subset_a_indices_.tolist())
            | set(splitter.subset_b_indices_.tolist())
            | {splitter.unpaired_index_}
        )
        assert all_used == set(range(15))

    def test_most_similar_pair_is_correctly_identified(self):
        """Строим два вектора, максимально похожих между собой (почти
        коллинеарных), и проверяем, что сплиттер объединяет их в пару."""
        rng = np.random.default_rng(4)
        X = rng.standard_normal((10, 64))
        X[3] = X[7] * 2.0 + rng.standard_normal(64) * 1e-6  # почти коллинеарны

        splitter = CorrelatedPairSplitter()
        splitter.fit(X)

        pairs_as_sets = [set(p) for p in splitter.pairs_.tolist()]
        assert {3, 7} in pairs_as_sets

    def test_deterministic_given_fixed_input(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((14, 32))

        a = CorrelatedPairSplitter()
        a.fit(X)
        b = CorrelatedPairSplitter()
        b.fit(X)

        np.testing.assert_array_equal(a.subset_a_indices_, b.subset_a_indices_)
        np.testing.assert_array_equal(a.subset_b_indices_, b.subset_b_indices_)
        np.testing.assert_array_equal(a.pairs_, b.pairs_)


@pytest.mark.theory
class TestCorrelatedPairSplitterTransform:
    def test_transform_subset_a_returns_correct_rows(self):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((10, 32))

        splitter = CorrelatedPairSplitter()
        splitter.fit(X)
        X_a = splitter.transform(X, subset="a")

        np.testing.assert_array_equal(X_a, X[splitter.subset_a_indices_])

    def test_transform_subset_b_returns_correct_rows(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((10, 32))

        splitter = CorrelatedPairSplitter()
        splitter.fit(X)
        X_b = splitter.transform(X, subset="b")

        np.testing.assert_array_equal(X_b, X[splitter.subset_b_indices_])

    def test_transform_default_is_subset_a(self):
        rng = np.random.default_rng(8)
        X = rng.standard_normal((10, 32))

        splitter = CorrelatedPairSplitter()
        splitter.fit(X)

        np.testing.assert_array_equal(splitter.transform(X), splitter.transform(X, subset="a"))

    def test_transform_invalid_subset_raises(self):
        rng = np.random.default_rng(9)
        X = rng.standard_normal((10, 32))

        splitter = CorrelatedPairSplitter()
        splitter.fit(X)
        with pytest.raises(ValueError):
            splitter.transform(X, subset="c")

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((10, 32))

        a = CorrelatedPairSplitter()
        result_a = a.fit_transform(X, subset="b")

        b = CorrelatedPairSplitter()
        b.fit(X)
        result_b = b.transform(X, subset="b")

        np.testing.assert_array_equal(result_a, result_b)

    def test_transform_before_fit_raises(self):
        splitter = CorrelatedPairSplitter()
        with pytest.raises(RuntimeError):
            splitter.transform(np.random.randn(5, 10))


@pytest.mark.theory
class TestCorrelatedPairSplitterEdgeCases:
    def test_single_vector_input_is_entirely_unpaired(self):
        X = np.random.randn(1, 16)
        splitter = CorrelatedPairSplitter()
        splitter.fit(X)

        assert splitter.unpaired_index_ == 0
        assert len(splitter.subset_a_indices_) == 0
        assert len(splitter.subset_b_indices_) == 0
        assert len(splitter.pairs_) == 0

    def test_two_vector_input_forms_single_pair(self):
        X = np.random.randn(2, 16)
        splitter = CorrelatedPairSplitter()
        splitter.fit(X)

        assert splitter.unpaired_index_ is None
        assert len(splitter.subset_a_indices_) == 1
        assert len(splitter.subset_b_indices_) == 1

    def test_empty_input_raises(self):
        splitter = CorrelatedPairSplitter()
        with pytest.raises(ValueError):
            splitter.fit(np.zeros((0, 16)))

    def test_1d_input_raises(self):
        splitter = CorrelatedPairSplitter()
        with pytest.raises(ValueError):
            splitter.fit(np.zeros(16))

    def test_repr_before_and_after_fit(self):
        splitter = CorrelatedPairSplitter()
        assert "not fitted" in repr(splitter)

        splitter.fit(np.random.randn(10, 16))
        assert "pairs=" in repr(splitter)
