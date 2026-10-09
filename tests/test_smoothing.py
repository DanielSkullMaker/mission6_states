"""Тесты features/smoothing.py — smooth_adjacent_vectors.

Реализует находку №5 (theory/code/Recogn_VSPU_S1.m) — сглаживание соседних
ПО ИНДЕКСУ векторов признаков (не пикселей внутри изображения) перед
классификацией. НЕ часть опубликованной статьи Korshikov & Fursov.
"""

import numpy as np
import pytest

from subspace_conjugacy.features.smoothing import smooth_adjacent_vectors


@pytest.mark.theory
class TestSmoothAdjacentVectorsBasics:
    def test_boundaries_never_smoothed(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((12, 6))

        X_smoothed = smooth_adjacent_vectors(X)

        np.testing.assert_array_equal(X_smoothed[0], X[0])
        np.testing.assert_array_equal(X_smoothed[-1], X[-1])

    def test_output_shape_matches_input(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((9, 4))

        X_smoothed = smooth_adjacent_vectors(X)
        assert X_smoothed.shape == X.shape

    def test_position_1_uses_window_3(self):
        """MATLAB: i==2 (1-indexed) -> окно (i-1, i, i+1), размер 3."""
        X = np.arange(9 * 4, dtype=float).reshape(9, 4)
        X_smoothed = smooth_adjacent_vectors(X)

        expected = X[0:3].mean(axis=0)
        np.testing.assert_allclose(X_smoothed[1], expected)

    def test_position_2_uses_window_5(self):
        """MATLAB: i==3 -> окно (i-2..i+2), размер 5."""
        X = np.arange(9 * 4, dtype=float).reshape(9, 4)
        X_smoothed = smooth_adjacent_vectors(X)

        expected = X[0:5].mean(axis=0)
        np.testing.assert_allclose(X_smoothed[2], expected)

    def test_middle_position_uses_window_7(self):
        """MATLAB: 4 <= i <= size-3 -> окно (i-3..i+3), размер 7."""
        X = np.arange(9 * 4, dtype=float).reshape(9, 4)
        X_smoothed = smooth_adjacent_vectors(X)

        # Позиция 4 (0-indexed) — расстояние до границы min(4, 8-4)=4 >= 3 -> half_width=3.
        expected = X[1:8].mean(axis=0)
        np.testing.assert_allclose(X_smoothed[4], expected)

    def test_symmetric_positions_near_far_boundary_use_same_window_size(self):
        """distance = min(i, M-1-i) — симметрично у обеих границ, значит
        окна одинакового размера (но не обязательно одинаковых значений,
        т.к. данные вокруг позиций разные)."""
        X = np.arange(11 * 3, dtype=float).reshape(11, 3)
        X_smoothed = smooth_adjacent_vectors(X)

        # Позиция 1: расстояние 1 -> окно [0:3]. Позиция 9: расстояние
        # min(9, 11-1-9)=1 -> окно [8:11].
        np.testing.assert_allclose(X_smoothed[1], X[0:3].mean(axis=0))
        np.testing.assert_allclose(X_smoothed[9], X[8:11].mean(axis=0))

        # Позиция 2: расстояние 2 -> окно [0:5]. Позиция 8: расстояние
        # min(8, 11-1-8)=2 -> окно [6:11].
        np.testing.assert_allclose(X_smoothed[2], X[0:5].mean(axis=0))
        np.testing.assert_allclose(X_smoothed[8], X[6:11].mean(axis=0))


@pytest.mark.theory
class TestSmoothAdjacentVectorsCustomHalfWidths:
    def test_single_half_width_caps_all_interior_positions(self):
        X = np.arange(9 * 4, dtype=float).reshape(9, 4)
        X_smoothed = smooth_adjacent_vectors(X, half_widths=(1,))

        for i in range(1, 8):
            expected = X[i - 1 : i + 2].mean(axis=0)
            np.testing.assert_allclose(X_smoothed[i], expected)

    def test_window_too_wide_for_position_leaves_row_unchanged(self):
        """half_width, не помещающийся в границы X, не должен обрезаться
        асимметрично — строка остаётся без изменений."""
        X = np.arange(6 * 4, dtype=float).reshape(6, 4)
        X_smoothed = smooth_adjacent_vectors(X, half_widths=(5,))

        # Ни для одной internal-позиции окно ±5 не помещается в M=6.
        np.testing.assert_array_equal(X_smoothed, X)

    def test_empty_half_widths_raises(self):
        X = np.random.randn(9, 4)
        with pytest.raises(ValueError):
            smooth_adjacent_vectors(X, half_widths=())

    def test_non_positive_half_width_raises(self):
        X = np.random.randn(9, 4)
        with pytest.raises(ValueError):
            smooth_adjacent_vectors(X, half_widths=(1, 0, 3))
        with pytest.raises(ValueError):
            smooth_adjacent_vectors(X, half_widths=(-1,))


@pytest.mark.theory
class TestSmoothAdjacentVectorsEdgeCases:
    def test_fewer_than_3_rows_returns_unchanged_copy(self):
        X = np.random.randn(2, 4)
        X_smoothed = smooth_adjacent_vectors(X)

        np.testing.assert_array_equal(X_smoothed, X)
        assert X_smoothed is not X  # копия, не тот же объект

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            smooth_adjacent_vectors(np.zeros(10))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            smooth_adjacent_vectors(np.zeros((0, 4)))

    def test_does_not_mutate_input(self):
        X = np.arange(9 * 4, dtype=float).reshape(9, 4)
        X_copy = X.copy()

        smooth_adjacent_vectors(X)

        np.testing.assert_array_equal(X, X_copy)
