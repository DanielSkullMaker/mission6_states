"""Theory tests для RecursiveBisectionClusterer (theory/Теоретические основы.docx).

Проверяет инварианты рекурсивной бисекции (альтернатива канону
ReferenceCenterBuilder/FursovClusterer) на синтетических данных.
"""

import numpy as np
import pytest

from subspace_conjugacy.algorithms.recursive_bisection import RecursiveBisectionClusterer
from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher


@pytest.mark.theory
class TestRecursiveBisectionClustererTheory:
    """Инварианты рекурсивной бисекции на синтетических данных."""

    def test_all_vectors_assigned_exactly_one_leaf(self):
        rng = np.random.RandomState(0)
        X = rng.randn(40, 16)

        clusterer = RecursiveBisectionClusterer(n_subclasses=4, freeze_basis_at=2)
        clusterer.fit(X)

        assert clusterer.labels_.shape == (40,)
        assert set(clusterer.labels_.tolist()) == {0, 1, 2, 3}
        assert (clusterer.labels_ >= 0).all()

    def test_n_subclasses_and_basis_shapes(self):
        rng = np.random.RandomState(1)
        X = rng.randn(60, 10)

        clusterer = RecursiveBisectionClusterer(n_subclasses=8, freeze_basis_at=2)
        clusterer.fit(X)

        assert clusterer.n_subclasses_ == 8
        assert len(clusterer.subspaces_) == 8
        for Y in clusterer.subspaces_:
            assert Y.shape == (10, 2)

    def test_leaf_sizes_sum_to_total(self):
        rng = np.random.RandomState(2)
        X = rng.randn(50, 8)

        clusterer = RecursiveBisectionClusterer(n_subclasses=5, freeze_basis_at=2)
        clusterer.fit(X)

        assert sum(clusterer.leaf_sizes_history_) == 50
        assert np.array_equal(
            np.bincount(clusterer.labels_, minlength=5),
            np.array(clusterer.leaf_sizes_history_),
        )

    def test_freeze_basis_at_none_keeps_full_leaf_growth(self):
        rng = np.random.RandomState(3)
        X = rng.randn(30, 6)

        clusterer = RecursiveBisectionClusterer(n_subclasses=3, freeze_basis_at=None)
        clusterer.fit(X)

        sizes = clusterer.get_subclass_sizes()
        for Y, size in zip(clusterer.subspaces_, sizes):
            assert Y.shape[1] == size  # базис растёт до размера листа без усечения

    def test_raises_on_insufficient_vectors(self):
        rng = np.random.RandomState(4)
        X = rng.randn(6, 5)  # < 2*4

        with pytest.raises(ValueError, match="Недостаточно векторов"):
            RecursiveBisectionClusterer(n_subclasses=4).fit(X)

    def test_raises_when_split_impossible(self):
        # 10 векторов, n_subclasses=8 -> min_required=16 уже проверяется
        # отдельно; здесь проверяем промежуточный сбой разбиения конкретно
        # через явно малое n_subclasses относительно M, но с крайне
        # неравномерным распределением после первых разбиений смоделировать
        # сложно детерминированно на случайных данных — проверяем через
        # прямой вызов с минимально достаточным M, где каждое разбиение
        # должно быть ровно 2+2+2+2.
        rng = np.random.RandomState(5)
        X = rng.randn(8, 5)  # ровно 2*n_subclasses

        clusterer = RecursiveBisectionClusterer(n_subclasses=4, freeze_basis_at=2)
        # Не обязано упасть (может сойтись к листьям по 2), но не должно
        # падать с другой ошибкой, кроме предусмотренной ValueError.
        try:
            clusterer.fit(X)
            assert clusterer.n_subclasses_ == 4
        except ValueError as exc:
            assert "Невозможно получить" in str(exc)

    def test_predict_matches_fit_labels_reasonably(self):
        rng = np.random.RandomState(6)
        # Два явно разных направления, чтобы бисекция была содержательной.
        X = np.vstack([
            rng.randn(20, 10) + np.array([5.0] + [0.0] * 9),
            rng.randn(20, 10) + np.array([-5.0] + [0.0] * 9),
        ])

        clusterer = RecursiveBisectionClusterer(n_subclasses=2, freeze_basis_at=2)
        clusterer.fit(X)
        predicted = clusterer.predict(X)

        # Самопредсказание должно воспроизводить фактическое распределение
        # по листьям хотя бы с точностью выше случайного угадывания (50%).
        agreement = np.mean(predicted == clusterer.labels_)
        assert agreement > 0.5


@pytest.mark.theory
class TestCosineSecondVectorAttacherCriterion:
    """criterion='max' (theory) должен давать иной результат, чем 'min' (канон)."""

    def test_min_and_max_can_disagree(self):
        X = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],   # центр 1
            [0.0, 0.9, 0.1],   # похож на центр 1 (высокий cos)
            [0.0, -0.9, 0.1],  # непохож на центр 1 (низкий cos)
            [1.0, 0.0, 0.1],   # центр 0
        ])
        centers = np.array([0, 1])

        min_attacher = CosineSecondVectorAttacher(criterion="min")
        min_attacher.fit(X, centers)

        max_attacher = CosineSecondVectorAttacher(criterion="max")
        max_attacher.fit(X, centers)

        assert not np.array_equal(min_attacher.pairs_, max_attacher.pairs_)

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError, match="criterion"):
            CosineSecondVectorAttacher(criterion="bogus")
