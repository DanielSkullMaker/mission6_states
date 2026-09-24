"""Тесты evaluation/ensemble_diagnostics.py — диагностика согласованности
ошибок нескольких классификаторов (статья 2, задача 1 методологии,
theory/article_plans/02_simbioz_arkhitektur.txt).
"""

import numpy as np
import pytest

from subspace_conjugacy.evaluation.ensemble_diagnostics import (
    compute_ensemble_diagnostics,
)


class TestComputeEnsembleDiagnosticsBasics:
    def test_raises_with_single_model(self):
        y_true = np.array(["a", "b"])
        with pytest.raises(ValueError, match="минимум 2"):
            compute_ensemble_diagnostics(y_true, {"only_one": np.array(["a", "b"])})

    def test_raises_with_mismatched_length(self):
        y_true = np.array(["a", "b", "c"])
        with pytest.raises(ValueError, match="предсказаний"):
            compute_ensemble_diagnostics(
                y_true,
                {"m1": np.array(["a", "b", "c"]), "m2": np.array(["a", "b"])},
            )

    def test_model_names_preserve_input_order(self):
        y_true = np.array(["a", "b"])
        diag = compute_ensemble_diagnostics(
            y_true, {"zeta": y_true.copy(), "alpha": y_true.copy()}
        )
        assert diag["model_names"] == ["zeta", "alpha"]


class TestAccuracyByModel:
    def test_perfect_and_imperfect_models(self):
        y_true = np.array(["a", "a", "b", "b"])
        preds = {
            "perfect": np.array(["a", "a", "b", "b"]),
            "half_wrong": np.array(["a", "b", "b", "a"]),
        }
        diag = compute_ensemble_diagnostics(y_true, preds)
        assert diag["accuracy_by_model"]["perfect"] == pytest.approx(1.0)
        assert diag["accuracy_by_model"]["half_wrong"] == pytest.approx(0.5)


class TestErrorOverlapCeilingAndFloor:
    def test_oracle_accuracy_is_one_when_at_least_one_model_always_right(self):
        y_true = np.array(["a", "b", "c", "a"])
        preds = {
            # каждая модель ошибается на своём объекте, но не на всех сразу
            "m1": np.array(["x", "b", "c", "a"]),
            "m2": np.array(["a", "x", "c", "a"]),
            "m3": np.array(["a", "b", "x", "a"]),
        }
        diag = compute_ensemble_diagnostics(y_true, preds)
        assert diag["oracle_accuracy"] == pytest.approx(1.0)
        assert diag["fraction_all_wrong"] == pytest.approx(0.0)

    def test_fraction_all_wrong_when_all_models_share_the_same_mistake(self):
        y_true = np.array(["a", "b"])
        preds = {
            "m1": np.array(["b", "a"]),  # оба неверны на обоих объектах
            "m2": np.array(["b", "a"]),
        }
        diag = compute_ensemble_diagnostics(y_true, preds)
        assert diag["fraction_all_wrong"] == pytest.approx(1.0)
        assert diag["oracle_accuracy"] == pytest.approx(0.0)
        assert diag["fraction_exactly_one_wrong"] == pytest.approx(0.0)

    def test_fraction_exactly_one_wrong(self):
        y_true = np.array(["a", "b", "c"])
        preds = {
            # объект 0: обе модели правы; объект 1: только m2 ошибается;
            # объект 2: обе ошибаются.
            "m1": np.array(["a", "b", "x"]),
            "m2": np.array(["a", "x", "y"]),
        }
        diag = compute_ensemble_diagnostics(y_true, preds)
        assert diag["fraction_exactly_one_wrong"] == pytest.approx(1 / 3)
        assert diag["fraction_all_wrong"] == pytest.approx(1 / 3)
        assert diag["oracle_accuracy"] == pytest.approx(2 / 3)

    def test_histogram_covers_all_counts_including_zero_occurrences(self):
        y_true = np.array(["a", "a"])
        preds = {
            "m1": np.array(["a", "a"]),
            "m2": np.array(["a", "a"]),
        }
        diag = compute_ensemble_diagnostics(y_true, preds)
        # 3 модели бы дали ключи 0..3 -- здесь 2 модели -> ключи 0..2, все
        # объекты у "0 моделей ошиблось".
        assert diag["n_models_wrong_histogram"] == {0: 2, 1: 0, 2: 0}


class TestPairwiseStatistics:
    def test_pairwise_agreement_full_when_identical_predictions(self):
        y_true = np.array(["a", "b", "c"])
        preds = {"m1": y_true.copy(), "m2": y_true.copy()}
        diag = compute_ensemble_diagnostics(y_true, preds)
        assert diag["pairwise_agreement"][("m1", "m2")] == pytest.approx(1.0)

    def test_pairwise_error_correlation_nan_when_one_model_never_wrong(self):
        y_true = np.array(["a", "b", "c", "a"])
        preds = {
            "always_right": y_true.copy(),
            "sometimes_wrong": np.array(["a", "x", "c", "a"]),
        }
        diag = compute_ensemble_diagnostics(y_true, preds)
        corr = diag["pairwise_error_correlation"][("always_right", "sometimes_wrong")]
        assert np.isnan(corr)

    def test_pairwise_error_correlation_high_when_models_fail_together(self):
        rng = np.random.default_rng(0)
        y_true = np.array([f"c{i % 3}" for i in range(60)])
        shared_errors = rng.random(60) < 0.4
        m1 = np.where(shared_errors, "wrong", y_true)
        m2 = np.where(shared_errors, "wrong", y_true)
        diag = compute_ensemble_diagnostics(y_true, {"m1": m1, "m2": m2})
        assert diag["pairwise_error_correlation"][("m1", "m2")] == pytest.approx(1.0)

    def test_three_models_produce_three_pairs(self):
        y_true = np.array(["a", "b"])
        preds = {"m1": y_true.copy(), "m2": y_true.copy(), "m3": y_true.copy()}
        diag = compute_ensemble_diagnostics(y_true, preds)
        assert set(diag["pairwise_agreement"].keys()) == {
            ("m1", "m2"), ("m1", "m3"), ("m2", "m3"),
        }
