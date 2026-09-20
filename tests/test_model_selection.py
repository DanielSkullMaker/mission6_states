"""Тесты model_selection/ — grid search и random search для SubspaceConjugacyClassifier."""

import numpy as np
import pytest
from scipy.stats import loguniform
from sklearn.base import is_classifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from subspace_conjugacy.model_selection import (
    DEFAULT_PARAM_DISTRIBUTIONS,
    DEFAULT_PARAM_GRID,
    grid_search_classifier,
    random_search_classifier,
    search_hyperparameters,
    summarize_search_results,
)
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier


@pytest.fixture
def three_class_data():
    """Синтетический трёхклассовый датасет — достаточно большой для cv=3
    с небольшими n_subclasses без вырождения фолдов."""
    rng = np.random.default_rng(42)
    n_per_class = 24
    n_features = 32
    X_list, y_list = [], []
    for idx, label in enumerate(["glioma", "meningioma", "pituitary"]):
        X_list.append(rng.standard_normal((n_per_class, n_features)) + idx * 4.0)
        y_list.append(np.full(n_per_class, label))
    return np.vstack(X_list), np.concatenate(y_list)


class TestIsClassifierRegression:
    """Регрессионный тест на баг с порядком ClassifierMixin/BaseSubspaceEstimator.

    is_classifier(...) == False приводил к тому, что GridSearchCV с целым cv
    использовал обычный KFold вместо StratifiedKFold и на данных,
    сгруппированных по классам, давал accuracy≡0 без единой ошибки — самая
    незаметная форма поломки для конкретно этой фичи (поиск гиперпараметров).
    """

    def test_is_classifier_true(self):
        assert is_classifier(SubspaceConjugacyClassifier())

    def test_cross_val_score_with_int_cv_is_not_degenerate(self, three_class_data):
        """С целым cv sklearn должен стратифицировать по классам сам."""
        from sklearn.model_selection import cross_val_score

        X, y = three_class_data
        clf = SubspaceConjugacyClassifier(n_subclasses=3)
        scores = cross_val_score(clf, X, y, cv=3)

        # До фикса is_classifier()==False заставлял sklearn использовать
        # обычный (не стратифицированный) KFold: на отсортированных по
        # классам данных train-фолд целиком терял один класс, и accuracy
        # была РОВНО 0.0 на всех трёх фолдах. Порог 0.4 — заметно выше
        # чистого угадывания (1/3 для 3 классов), но не требует высокого
        # абсолютного качества метода на маленьком синтетическом датасете.
        assert scores.mean() > 0.4
        assert not np.any(scores == 0.0)


class TestGridSearchClassifier:
    def test_returns_fitted_grid_search_cv(self, three_class_data):
        X, y = three_class_data
        search = grid_search_classifier(
            X, y,
            param_grid={"n_subclasses": [2, 3], "reg_param": [1e-8, 1e-6]},
            cv=3,
        )

        assert isinstance(search, GridSearchCV)
        assert search.best_params_ is not None
        assert 0.0 <= search.best_score_ <= 1.0
        assert search.best_estimator_.is_fitted_

    def test_best_estimator_can_predict(self, three_class_data):
        X, y = three_class_data
        search = grid_search_classifier(
            X, y, param_grid={"n_subclasses": [2, 3]}, cv=3,
        )

        preds = search.best_estimator_.predict(X[:5])
        assert len(preds) == 5
        assert set(preds).issubset({"glioma", "meningioma", "pituitary"})

    def test_default_param_grid_used_when_none(self, three_class_data):
        X, y = three_class_data
        # DEFAULT_PARAM_GRID включает n_subclasses до 12 — на 16 объектах на
        # класс в train-фолде (24 * 2/3) часть комбинаций может провалиться
        # (error_score=nan), но сам вызов не должен бросать исключение.
        search = grid_search_classifier(X, y, cv=3)
        assert search.cv_results_["params"][0].keys() == DEFAULT_PARAM_GRID.keys()

    def test_refit_false_skips_best_estimator(self, three_class_data):
        X, y = three_class_data
        search = grid_search_classifier(
            X, y, param_grid={"n_subclasses": [2, 3]}, cv=3, refit=False,
        )
        with pytest.raises(AttributeError):
            _ = search.best_estimator_

    def test_custom_base_estimator_growth_strategy_fixed(self, three_class_data):
        X, y = three_class_data
        base = SubspaceConjugacyClassifier(growth_strategy="master")
        search = grid_search_classifier(
            X, y,
            param_grid={"n_subclasses": [2, 3]},
            cv=3,
            base_estimator=base,
        )
        assert search.best_estimator_.growth_strategy == "master"

    def test_can_grid_search_over_per_class_n_subclasses_dicts(self, three_class_data):
        """n_subclasses per-class (Mapping) — обычное значение для точки
        param_grid: GridSearchCV не делает ничего специального с типом
        значения, просто clone().set_params(n_subclasses=<dict>). Это
        разблокирует тюнинг числа подклассов независимо на класс, как
        рекомендует статья Korshikov & Fursov."""
        X, y = three_class_data
        candidate_grids = [
            {"glioma": 2, "meningioma": 2, "pituitary": 2},
            {"glioma": 3, "meningioma": 2, "pituitary": 4},
        ]
        search = grid_search_classifier(
            X, y,
            param_grid={"n_subclasses": candidate_grids},
            cv=3,
        )

        assert search.best_params_["n_subclasses"] in candidate_grids
        assert search.best_estimator_.n_subclasses_by_class_ == search.best_params_["n_subclasses"]


class TestRandomSearchClassifier:
    def test_returns_fitted_randomized_search_cv(self, three_class_data):
        X, y = three_class_data
        search = random_search_classifier(
            X, y,
            param_distributions={"n_subclasses": [2, 3], "reg_param": loguniform(1e-10, 1e-4)},
            n_iter=4,
            cv=3,
            random_state=0,
        )

        assert isinstance(search, RandomizedSearchCV)
        assert len(search.cv_results_["params"]) == 4
        assert 0.0 <= search.best_score_ <= 1.0

    def test_reproducible_with_same_random_state(self, three_class_data):
        X, y = three_class_data
        kwargs = dict(
            param_distributions={"n_subclasses": [2, 3, 4], "reg_param": loguniform(1e-10, 1e-4)},
            n_iter=5,
            cv=3,
            random_state=7,
        )
        search_a = random_search_classifier(X, y, **kwargs)
        search_b = random_search_classifier(X, y, **kwargs)

        assert search_a.cv_results_["params"] == search_b.cv_results_["params"]

    def test_default_param_distributions_used_when_none(self, three_class_data):
        X, y = three_class_data
        search = random_search_classifier(X, y, n_iter=3, cv=3, random_state=0)
        assert search.cv_results_["params"][0].keys() == DEFAULT_PARAM_DISTRIBUTIONS.keys()


class TestSearchHyperparametersDispatcher:
    def test_dispatches_to_grid(self, three_class_data):
        X, y = three_class_data
        search = search_hyperparameters(
            X, y, method="grid", param_grid={"n_subclasses": [2, 3]}, cv=3,
        )
        assert isinstance(search, GridSearchCV)

    def test_dispatches_to_random(self, three_class_data):
        X, y = three_class_data
        search = search_hyperparameters(
            X, y, method="random",
            param_distributions={"n_subclasses": [2, 3]},
            n_iter=3, cv=3, random_state=0,
        )
        assert isinstance(search, RandomizedSearchCV)

    def test_unknown_method_raises(self, three_class_data):
        X, y = three_class_data
        with pytest.raises(ValueError):
            search_hyperparameters(X, y, method="bayesian")


class TestSummarizeSearchResults:
    def test_returns_sorted_rows(self, three_class_data):
        X, y = three_class_data
        search = grid_search_classifier(
            X, y,
            param_grid={"n_subclasses": [2, 3], "reg_param": [1e-8, 1e-6]},
            cv=3,
        )

        rows = summarize_search_results(search, top_n=3)

        assert len(rows) == 3
        ranks = [row["rank"] for row in rows]
        assert ranks == sorted(ranks)
        for row in rows:
            assert set(row.keys()) == {"rank", "mean_test_score", "std_test_score", "params"}

    def test_top_n_larger_than_results_returns_all(self, three_class_data):
        X, y = three_class_data
        search = grid_search_classifier(
            X, y, param_grid={"n_subclasses": [2, 3]}, cv=3,
        )
        rows = summarize_search_results(search, top_n=100)
        assert len(rows) == 2


class TestPublicApiExport:
    def test_importable_from_top_level_package(self):
        import subspace_conjugacy

        assert subspace_conjugacy.grid_search_classifier is grid_search_classifier
        assert subspace_conjugacy.random_search_classifier is random_search_classifier
        assert subspace_conjugacy.search_hyperparameters is search_hyperparameters
