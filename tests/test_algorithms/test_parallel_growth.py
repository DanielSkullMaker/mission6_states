"""Theory tests для ParallelConjugacyClusterGrowth / ParallelFursovClusterer.

Проверяет, что векторизованная/параллельная альтернатива канону
(subclass_growth_parallel.py, fursov_clusterer_parallel.py) даёт ИДЕНТИЧНЫЙ
результат каноническим ConjugacyClusterGrowth/FursovClusterer при разных
n_jobs — расхождение между ними должно быть только в скорости, не в
корректности (это и есть требование, под которое писались эти модули:
"не затрагивая уже написанный код, далее он будет необходим для сравнения").
"""

import numpy as np
import pytest

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher
from subspace_conjugacy.algorithms.subclass_growth import ConjugacyClusterGrowth
from subspace_conjugacy.algorithms.subclass_growth_parallel import ParallelConjugacyClusterGrowth
from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer
from subspace_conjugacy.algorithms.fursov_clusterer_parallel import ParallelFursovClusterer


def _build_pairs(X: np.ndarray, n_subclasses: int):
    pf = GlobalMinCosinePairFinder()
    pf.fit(X)
    cb = ReferenceCenterBuilder(n_subclasses=n_subclasses)
    cb.fit(X, pf.pair_indices_)
    sa = CosineSecondVectorAttacher()
    sa.fit(X, cb.center_indices_)
    return sa.pairs_


@pytest.mark.theory
class TestParallelConjugacyClusterGrowthTheory:
    """ParallelConjugacyClusterGrowth должен совпадать с каноном при любом n_jobs."""

    @pytest.mark.parametrize("n_jobs", [1, 2, 4])
    def test_matches_canon_labels_and_bases(self, n_jobs):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 32)
        pairs = _build_pairs(X, n_subclasses=6)

        canon = ConjugacyClusterGrowth(freeze_basis_at=2)
        canon.fit(X, pairs)

        parallel = ParallelConjugacyClusterGrowth(freeze_basis_at=2, n_jobs=n_jobs)
        parallel.fit(X, pairs)

        assert np.array_equal(canon.labels_, parallel.labels_)
        for Y_canon, Y_parallel in zip(canon.subspace_bases_, parallel.subspace_bases_):
            assert np.allclose(Y_canon, Y_parallel)

    def test_predict_matches_canon(self):
        rng = np.random.RandomState(7)
        X = rng.randn(50, 20)
        pairs = _build_pairs(X, n_subclasses=5)

        canon = ConjugacyClusterGrowth(freeze_basis_at=2)
        canon.fit(X, pairs)
        parallel = ParallelConjugacyClusterGrowth(freeze_basis_at=2, n_jobs=2)
        parallel.fit(X, pairs)

        X_new = rng.randn(10, 20)
        assert np.array_equal(canon.predict(X_new), parallel.predict(X_new))

    def test_fit_time_recorded(self):
        rng = np.random.RandomState(3)
        X = rng.randn(40, 16)
        pairs = _build_pairs(X, n_subclasses=4)

        growth = ParallelConjugacyClusterGrowth(freeze_basis_at=2)
        growth.fit(X, pairs)
        assert growth.fit_time_s_ is not None and growth.fit_time_s_ >= 0


@pytest.mark.theory
class TestParallelFursovClustererTheory:
    """ParallelFursovClusterer должен совпадать с каноном FursovClusterer."""

    @pytest.mark.parametrize("n_jobs", [1, 3])
    def test_matches_canon(self, n_jobs):
        rng = np.random.RandomState(11)
        X = rng.randn(70, 24)

        canon = FursovClusterer(n_subclasses=6, freeze_basis_at=2)
        canon.fit(X)

        parallel = ParallelFursovClusterer(n_subclasses=6, freeze_basis_at=2, n_jobs=n_jobs)
        parallel.fit(X)

        assert np.array_equal(canon.labels_, parallel.labels_)
        assert np.array_equal(canon.kept_indices_, parallel.kept_indices_)
        for Y_canon, Y_parallel in zip(canon.subspaces_, parallel.subspaces_):
            assert np.allclose(Y_canon, Y_parallel)

    def test_insufficient_vectors_raises(self):
        rng = np.random.RandomState(5)
        X = rng.randn(4, 10)
        with pytest.raises(ValueError):
            ParallelFursovClusterer(n_subclasses=4).fit(X)
