"""Тесты algorithms/informativeness_filter.py — LowInformativenessFilter.

Реализует шаг предобработки из статьи Korshikov & Fursov, 3-й эксперимент:
"images with the number of white pixels less than 50% of the average
number of white pixels in the images of the set are cut off"
(refactoring_plan.txt, раздел 10, находка №5).
"""

import numpy as np
import pytest

from subspace_conjugacy.algorithms.informativeness_filter import (
    DEFAULT_BRIGHTNESS_THRESHOLD,
    DEFAULT_MIN_FRACTION_OF_MEAN,
    LowInformativenessFilter,
)


@pytest.mark.theory
class TestLowInformativenessFilterBasics:
    def test_uniformly_bright_vectors_nothing_excluded(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(100, 255, size=(10, 64))

        filt = LowInformativenessFilter()
        filt.fit(X)

        assert len(filt.excluded_indices_) == 0
        np.testing.assert_array_equal(filt.kept_indices_, np.arange(10))

    def test_near_black_vector_is_excluded(self):
        X = np.full((10, 100), 200.0)
        X[5] = 0.0

        filt = LowInformativenessFilter()
        filt.fit(X)

        assert 5 in filt.excluded_indices_
        assert 5 not in filt.kept_indices_

    def test_mean_and_cutoff_computed_correctly(self):
        X = np.full((10, 100), 200.0)
        X[5] = 0.0

        filt = LowInformativenessFilter(brightness_threshold=10, min_fraction_of_mean=0.5)
        filt.fit(X)

        assert filt.mean_white_count_ == pytest.approx(90.0)
        assert filt.cutoff_ == pytest.approx(45.0)

    def test_kept_and_excluded_partition_all_indices(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(100, 255, size=(15, 64))
        X[3] = 0.0
        X[9] = 0.0

        filt = LowInformativenessFilter()
        filt.fit(X)

        all_indices = sorted(list(filt.kept_indices_) + list(filt.excluded_indices_))
        assert all_indices == list(range(15))


@pytest.mark.theory
class TestLowInformativenessFilterParams:
    def test_min_fraction_must_be_in_valid_range(self):
        with pytest.raises(ValueError):
            LowInformativenessFilter(min_fraction_of_mean=0.0)
        with pytest.raises(ValueError):
            LowInformativenessFilter(min_fraction_of_mean=1.5)
        with pytest.raises(ValueError):
            LowInformativenessFilter(min_fraction_of_mean=-0.1)

    def test_min_fraction_at_upper_bound_is_valid(self):
        LowInformativenessFilter(min_fraction_of_mean=1.0)  # не должно бросать

    def test_defaults_match_module_constants(self):
        filt = LowInformativenessFilter()
        assert filt.brightness_threshold == DEFAULT_BRIGHTNESS_THRESHOLD
        assert filt.min_fraction_of_mean == DEFAULT_MIN_FRACTION_OF_MEAN

    def test_lower_min_fraction_excludes_fewer(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(150, 255, size=(10, 64))
        X[4] = rng.uniform(0, 30, size=64)  # заметно темнее остальных, но не полностью чёрный

        strict = LowInformativenessFilter(min_fraction_of_mean=0.9)
        strict.fit(X)
        lenient = LowInformativenessFilter(min_fraction_of_mean=0.1)
        lenient.fit(X)

        assert len(lenient.excluded_indices_) <= len(strict.excluded_indices_)


@pytest.mark.theory
class TestLowInformativenessFilterTransform:
    def test_transform_returns_only_kept_rows(self):
        X = np.full((10, 64), 200.0)
        X[6] = 0.0

        filt = LowInformativenessFilter()
        filt.fit(X)
        X_filtered = filt.transform(X)

        assert X_filtered.shape[0] == len(filt.kept_indices_)
        np.testing.assert_array_equal(X_filtered, X[filt.kept_indices_])

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(100, 255, size=(10, 64))
        X[2] = 0.0

        a = LowInformativenessFilter()
        result_a = a.fit_transform(X)

        b = LowInformativenessFilter()
        b.fit(X)
        result_b = b.transform(X)

        np.testing.assert_array_equal(result_a, result_b)

    def test_transform_before_fit_raises(self):
        filt = LowInformativenessFilter()
        with pytest.raises(RuntimeError):
            filt.transform(np.random.randn(5, 10))


@pytest.mark.theory
class TestLowInformativenessFilterEdgeCases:
    def test_single_vector_input_always_kept(self):
        X = np.random.uniform(50, 255, size=(1, 16))
        filt = LowInformativenessFilter()
        filt.fit(X)

        assert list(filt.kept_indices_) == [0]
        assert len(filt.excluded_indices_) == 0

    def test_empty_input_raises(self):
        filt = LowInformativenessFilter()
        with pytest.raises(ValueError):
            filt.fit(np.zeros((0, 16)))

    def test_1d_input_raises(self):
        filt = LowInformativenessFilter()
        with pytest.raises(ValueError):
            filt.fit(np.zeros(16))

    def test_repr_before_and_after_fit(self):
        filt = LowInformativenessFilter()
        assert "not fitted" in repr(filt)

        filt.fit(np.random.uniform(100, 255, size=(5, 16)))
        assert "kept=" in repr(filt)
        assert "excluded=" in repr(filt)
