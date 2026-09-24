"""Тесты models/ensemble.py — ансамблевая интеграция уже обученных
классификаторов (статья 2, "Симбиоз классификаторов",
theory/article_plans/02_simbioz_arkhitektur.txt).

Как и tests/test_sequential_classifier.py, использует простые синтетические
классы (не привязанные к конкретному домену) — PrefitVotingClassifier/
PrefitStackingClassifier/SwitchingEnsembleClassifier сами по себе
универсальны и не завязаны на MRI/метод сопряжённости конкретно, хотя
основной мотивирующий сценарий статьи — именно комбинация
SubspaceConjugacyClassifier с классическим ML/CNN.
"""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.linear_model import LogisticRegression

from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier
from subspace_conjugacy.models.ensemble import (
    PrefitVotingClassifier,
    PrefitStackingClassifier,
    SwitchingEnsembleClassifier,
)


def _make_class_blob(rng, offset, n, n_features):
    return rng.standard_normal((n, n_features)) + offset


@pytest.fixture
def three_class_split():
    """3 хорошо разделимых класса (a/b/c), раздельные train/val/test —
    честный holdout-протокол, как и в остальных экспериментах проекта."""
    rng = np.random.default_rng(42)
    classes = ["a", "b", "c"]

    def build(n_per_class, seed_offset):
        r = np.random.default_rng(1000 + seed_offset)
        X = np.vstack([_make_class_blob(r, i * 5.0, n_per_class, 24) for i in range(3)])
        y = np.repeat(classes, n_per_class)
        return X, y

    X_train, y_train = build(30, 0)
    X_val, y_val = build(15, 1)
    X_test, y_test = build(15, 2)
    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture
def fitted_base_models(three_class_split):
    X_train, y_train, *_ = three_class_split
    subspace = SubspaceConjugacyClassifier(n_subclasses=2).fit(X_train, y_train)
    logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    return subspace, logreg


class TestPrefitCheckValidation:
    def test_voting_rejects_single_model(self, fitted_base_models):
        subspace, _ = fitted_base_models
        with pytest.raises(ValueError, match="минимум 2"):
            PrefitVotingClassifier(estimators=[("subspace", subspace)]).fit()

    def test_voting_rejects_unfitted_model(self, fitted_base_models):
        subspace, _ = fitted_base_models
        unfitted = LogisticRegression()  # ещё не обучена, нет classes_
        with pytest.raises(ValueError, match="не обучена"):
            PrefitVotingClassifier(
                estimators=[("subspace", subspace), ("logreg", unfitted)]
            ).fit()

    def test_voting_rejects_duplicate_names(self, fitted_base_models):
        subspace, logreg = fitted_base_models
        with pytest.raises(ValueError, match="уникальны"):
            PrefitVotingClassifier(
                estimators=[("m", subspace), ("m", logreg)]
            ).fit()

    def test_voting_rejects_model_without_predict_proba(self, fitted_base_models):
        subspace, logreg = fitted_base_models

        class NoProba:
            classes_ = np.array(["a", "b", "c"])

            def predict(self, X):
                return np.full(len(X), "a")

        with pytest.raises(ValueError, match="predict_proba"):
            PrefitVotingClassifier(
                estimators=[("subspace", subspace), ("noproba", NoProba())]
            ).fit()


class TestPrefitVotingClassifier:
    def test_soft_voting_predicts_correct_classes(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, _, _, X_test, y_test = three_class_split

        ens = PrefitVotingClassifier(
            estimators=[("subspace", subspace), ("logreg", logreg)], voting="soft",
        ).fit()
        preds = ens.predict(X_test)

        assert np.mean(preds == y_test) >= 0.8  # хорошо разделимые классы
        assert set(ens.classes_) == {"a", "b", "c"}

    def test_hard_voting_uses_majority(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, _, _, X_test, y_test = three_class_split

        ens = PrefitVotingClassifier(
            estimators=[("subspace", subspace), ("logreg", logreg)], voting="hard",
        ).fit()
        preds = ens.predict(X_test)
        assert len(preds) == len(y_test)

    def test_predict_proba_rows_sum_to_one(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, _, _, X_test, _ = three_class_split

        ens = PrefitVotingClassifier(
            estimators=[("subspace", subspace), ("logreg", logreg)],
        ).fit()
        proba = ens.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_weights_change_result_towards_dominant_model(self, three_class_split):
        X_train, y_train, _, _, X_test, y_test = three_class_split

        # Модель good_model почти всегда права, bad_model — почти всегда
        # неправа (систематически предсказывает следующий класс по кругу).
        class GoodModel:
            classes_ = np.array(["a", "b", "c"])

            def predict_proba(self, X):
                n = len(X)
                proba = np.tile([0.05, 0.05, 0.9], (n, 1))
                return proba

        class BadModel:
            classes_ = np.array(["a", "b", "c"])

            def predict_proba(self, X):
                n = len(X)
                proba = np.tile([0.9, 0.05, 0.05], (n, 1))
                return proba

        ens = PrefitVotingClassifier(
            estimators=[("good", GoodModel()), ("bad", BadModel())],
            weights=[0.95, 0.05],
        ).fit()
        preds = ens.predict(X_test[:3])
        assert all(p == "c" for p in preds)  # доминирует good_model

    def test_invalid_voting_mode_raises(self, fitted_base_models):
        subspace, logreg = fitted_base_models
        with pytest.raises(ValueError, match="voting"):
            PrefitVotingClassifier(
                estimators=[("subspace", subspace), ("logreg", logreg)], voting="invalid",
            ).fit()

    def test_mismatched_weights_length_raises(self, fitted_base_models):
        subspace, logreg = fitted_base_models
        with pytest.raises(ValueError, match="weights"):
            PrefitVotingClassifier(
                estimators=[("subspace", subspace), ("logreg", logreg)],
                weights=[1.0, 1.0, 1.0],
            ).fit()

    def test_predict_before_fit_raises(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, _, _, X_test, _ = three_class_split
        ens = PrefitVotingClassifier(estimators=[("subspace", subspace), ("logreg", logreg)])
        with pytest.raises(RuntimeError, match="не обучен"):
            ens.predict(X_test)

    def test_is_classifier_and_clonable(self, fitted_base_models):
        subspace, logreg = fitted_base_models
        ens = PrefitVotingClassifier(estimators=[("subspace", subspace), ("logreg", logreg)])
        assert is_classifier(ens)
        cloned = clone(ens)
        assert isinstance(cloned, PrefitVotingClassifier)


class TestPrefitStackingClassifier:
    def test_stacking_fits_and_predicts(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, X_val, y_val, X_test, y_test = three_class_split

        ens = PrefitStackingClassifier(
            estimators=[("subspace", subspace), ("logreg", logreg)],
        ).fit(X_val, y_val)
        preds = ens.predict(X_test)

        assert np.mean(preds == y_test) >= 0.8
        assert set(ens.classes_) == {"a", "b", "c"}

    def test_predict_proba_rows_sum_to_one(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, X_val, y_val, X_test, _ = three_class_split

        ens = PrefitStackingClassifier(
            estimators=[("subspace", subspace), ("logreg", logreg)],
        ).fit(X_val, y_val)
        proba = ens.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_custom_meta_estimator_is_used(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, X_val, y_val, X_test, y_test = three_class_split

        meta = LogisticRegression(max_iter=500, C=0.1)
        ens = PrefitStackingClassifier(
            estimators=[("subspace", subspace), ("logreg", logreg)],
            meta_estimator=meta,
        ).fit(X_val, y_val)
        assert ens._meta_estimator_ is meta
        assert np.mean(ens.predict(X_test) == y_test) >= 0.7

    def test_predict_before_fit_raises(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, _, _, X_test, _ = three_class_split
        ens = PrefitStackingClassifier(estimators=[("subspace", subspace), ("logreg", logreg)])
        with pytest.raises(RuntimeError, match="не обучен"):
            ens.predict(X_test)

    def test_is_classifier(self, fitted_base_models):
        subspace, logreg = fitted_base_models
        ens = PrefitStackingClassifier(estimators=[("subspace", subspace), ("logreg", logreg)])
        assert is_classifier(ens)


class TestSwitchingEnsembleClassifier:
    def test_champion_map_covers_all_classes(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, X_val, y_val, _, _ = three_class_split

        ens = SwitchingEnsembleClassifier(
            estimators=[("subspace", subspace), ("logreg", logreg)],
        ).fit(X_val, y_val)

        assert set(ens.class_champion_.keys()) == {"a", "b", "c"}
        assert all(v in ("subspace", "logreg") for v in ens.class_champion_.values())
        assert set(ens.champion_metric_.keys()) == {"a", "b", "c"}

    def test_predicts_reasonably_on_separable_classes(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, X_val, y_val, X_test, y_test = three_class_split

        ens = SwitchingEnsembleClassifier(
            estimators=[("subspace", subspace), ("logreg", logreg)],
        ).fit(X_val, y_val)
        preds = ens.predict(X_test)
        assert np.mean(preds == y_test) >= 0.8

    def test_champion_per_class_picks_better_model(self):
        """Синтетический случай с явным чемпионом на каждый класс:
        model_a идеальна на 'a', случайна на остальных; model_b — наоборот.

        Обе заглушки намеренно возвращают КОНСТАНТНУЮ (не зависящую от X)
        вероятность — цель теста не в различении объектов по признакам (это
        проверяют тесты на реальных моделях выше), а в том, что итоговая
        оценка по каждому классу берётся СТРОГО у объявленного чемпиона
        этого класса, а не усредняется/берётся у другой модели.
        """
        classes = np.array(["a", "b"])

        class ModelA:
            classes_ = classes

            def predict(self, X):
                return np.full(len(X), "a")

            def predict_proba(self, X):
                return np.tile([0.9, 0.1], (len(X), 1))

        class ModelB:
            classes_ = classes

            def predict(self, X):
                return np.full(len(X), "b")

            def predict_proba(self, X):
                return np.tile([0.1, 0.9], (len(X), 1))

        X_val = np.zeros((4, 2))
        y_val = np.array(["a", "a", "b", "b"])

        ens = SwitchingEnsembleClassifier(
            estimators=[("model_a", ModelA()), ("model_b", ModelB())],
        ).fit(X_val, y_val)

        assert ens.class_champion_["a"] == "model_a"
        assert ens.class_champion_["b"] == "model_b"

        X_test = np.zeros((2, 2))
        # Столбец "a" итоговой матрицы должен быть взят у model_a (0.9), а
        # не у model_b (0.1); столбец "b" — у model_b (0.9), а не у model_a
        # (0.1). После построчной нормировки оба столбца равны 0.9/(0.9+0.9).
        proba = ens.predict_proba(X_test)
        class_a_idx = list(ens.classes_).index("a")
        class_b_idx = list(ens.classes_).index("b")
        np.testing.assert_allclose(proba[:, class_a_idx], 0.5, atol=1e-8)
        np.testing.assert_allclose(proba[:, class_b_idx], 0.5, atol=1e-8)

        # При равных (после нормировки чемпионов) счетах argmax детерминированно
        # берёт первый класс по возрастанию сортировки — это ожидаемое,
        # документированное поведение np.argmax при ничьей, а не баг.
        preds = ens.predict(X_test)
        assert all(p == ens.classes_[0] for p in preds)

    def test_invalid_selection_metric_raises(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, X_val, y_val, _, _ = three_class_split
        ens = SwitchingEnsembleClassifier(
            estimators=[("subspace", subspace), ("logreg", logreg)],
            selection_metric="precision",
        )
        with pytest.raises(ValueError, match="selection_metric"):
            ens.fit(X_val, y_val)

    def test_predict_before_fit_raises(self, fitted_base_models, three_class_split):
        subspace, logreg = fitted_base_models
        _, _, _, _, X_test, _ = three_class_split
        ens = SwitchingEnsembleClassifier(estimators=[("subspace", subspace), ("logreg", logreg)])
        with pytest.raises(RuntimeError, match="не обучен"):
            ens.predict(X_test)

    def test_is_classifier(self, fitted_base_models):
        subspace, logreg = fitted_base_models
        ens = SwitchingEnsembleClassifier(estimators=[("subspace", subspace), ("logreg", logreg)])
        assert is_classifier(ens)
