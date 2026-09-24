"""Ансамблевая интеграция УЖЕ ОБУЧЕННЫХ классификаторов (статья 2,
"Симбиоз классификаторов: ансамблевая интеграция метода подпространственной
сопряжённости с классическими моделями машинного обучения и свёрточными
нейронными сетями", theory/article_plans/02_simbioz_arkhitektur.txt).

НАХОДКА (аналогично находке №4 model_selection, про is_classifier): проверено
экспериментально — SubspaceConjugacyClassifier (ClassifierMixin +
BaseEstimator, sklearn-совместимые fit/predict/predict_proba/classes_) уже
работает БЕЗ ИЗМЕНЕНИЙ внутри стандартных sklearn.ensemble.VotingClassifier
и sklearn.ensemble.StackingClassifier — clone()/fit() отрабатывают корректно.
Поэтому для сценария "все базовые модели можно (пере)обучить средствами
sklearn на одних и тех же X, y" этот модуль НИЧЕГО не переизобретает —
используйте sklearn напрямую:

    from sklearn.ensemble import VotingClassifier
    from subspace_conjugacy import SubspaceConjugacyClassifier
    ens = VotingClassifier(
        estimators=[("subspace", SubspaceConjugacyClassifier()), ("logreg", LogisticRegression())],
        voting="soft",
    )
    ens.fit(X_train, y_train)

Пробел, который закрывает ЭТОТ модуль — сценарий статьи 2 (методология,
Шаг 1-2): три модели проекта (метод сопряжённости, классический ML, CNN)
обучаются РАЗНЫМИ, несовместимыми путями (CNN — отдельным циклом обучения
на PyTorch вне scikit-learn, main.py::train_cnn_model — sklearn.clone() не
умеет её пересоздать и переобучить). sklearn.ensemble.VotingClassifier/
StackingClassifier ВСЕГДА клонируют и переобучают базовые модели в fit() —
для UЖЕ обученной вне sklearn модели (как эта CNN) это не подходит. Три
класса ниже комбинируют ПРЕДСКАЗАНИЯ уже обученных моделей напрямую, без
повторного обучения кого-либо из них:

  - PrefitVotingClassifier — мягкое (по вероятностям) или жёсткое (по
    меткам) голосование уже обученных моделей.
  - PrefitStackingClassifier — метамодель поверх вероятностей уже обученных
    моделей, обучаемая на ОТДЕЛЬНОЙ валидационной выборке (не участвовавшей
    в обучении базовых моделей — тот же принцип честного holdout, что и
    везде в проекте, см. main.py::run_tuned_comparison_experiment).
  - SwitchingEnsembleClassifier — "селективное переключение" (план,
    задача 2): для каждого класса по отдельности выбирается "чемпион" —
    та из базовых моделей, что лучше всего (по валидационной выборке)
    распознаёт именно этот класс; итоговое решающее правило берёт для
    каждого класса оценку СВОЕГО чемпиона, а не усредняет всех.

Все три эстиматора ожидают на вход уже обученные объекты с атрибутом
classes_ и методом predict_proba(X) (SubspaceConjugacyClassifier и любой
классификатор scikit-learn — из коробки; сторонняя модель вроде CNN —
единственное требование к обёртке пользователя, специальный класс-обёртка
под конкретный фреймворк в эту библиотеку намеренно не включён, т.к. PyTorch
не входит в зависимости subspace_conjugacy, см. pyproject.toml, extra
"cnn-experiment").

Диагностика согласованности ошибок нескольких моделей (задача 1 плана,
"потенциал ансамбля" / "жёсткий потолок") — evaluation/ensemble_diagnostics.py.
"""

import logging
from typing import Any, List, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from subspace_conjugacy.evaluation.metrics import per_class_accuracy

logger = logging.getLogger(__name__)


def _check_prefit(estimators: Sequence[Tuple[str, Any]], caller: str) -> None:
    """Проверяет, что все базовые модели уже обучены (имеют classes_)."""
    if len(estimators) < 2:
        logger.error("%s: передано %d моделей, требуется минимум 2.", caller, len(estimators))
        raise ValueError(f"{caller} требует минимум 2 базовых модели, получено {len(estimators)}.")
    names = [name for name, _ in estimators]
    if len(set(names)) != len(names):
        logger.error("%s: дублирующиеся имена моделей %s.", caller, names)
        raise ValueError(f"Имена моделей в estimators должны быть уникальны, получено {names}.")
    for name, est in estimators:
        if not hasattr(est, "classes_"):
            logger.error(
                "%s: модель '%s' (%s) не обучена — отсутствует classes_. "
                "Prefit-ансамбли ожидают УЖЕ обученные модели, обучите их "
                "заранее через est.fit(X, y).", caller, name, type(est).__name__,
            )
            raise ValueError(
                f"Модель '{name}' ({type(est).__name__}) ещё не обучена "
                f"(нет атрибута classes_). {caller} комбинирует предсказания "
                f"УЖЕ обученных моделей — вызовите est.fit(X, y) для каждой "
                f"модели перед передачей сюда."
            )
        if not hasattr(est, "predict_proba"):
            logger.error(
                "%s: модель '%s' (%s) не имеет predict_proba.",
                caller, name, type(est).__name__,
            )
            raise ValueError(
                f"Модель '{name}' ({type(est).__name__}) не имеет predict_proba — "
                f"все базовые модели должны его поддерживать (voting='hard' "
                f"также требует predict_proba для непротиворечивости API; "
                f"используйте sklearn.ensemble.VotingClassifier(voting='hard') "
                f"с переобучением, если нужна работа только через predict())."
            )


def _common_classes(estimators: Sequence[Tuple[str, Any]]) -> np.ndarray:
    """Объединение classes_ всех моделей в один отсортированный массив."""
    all_classes = set()
    for _, est in estimators:
        all_classes.update(np.asarray(est.classes_).tolist())
    return np.array(sorted(all_classes))


def _aligned_proba(estimator: Any, X: np.ndarray, common_classes: np.ndarray) -> np.ndarray:
    """predict_proba(X) модели, репроецированный в столбцы common_classes.

    Классы, которые эта конкретная модель не видела при обучении, получают
    столбец из нулей — сумма строки при этом остаётся <= 1 (обычно = 1,
    если common_classes совпадают с classes_ модели, что в проекте — типичный
    случай: все базовые модели обучены на одном и том же наборе классов).
    """
    proba = estimator.predict_proba(X)
    est_classes = np.asarray(estimator.classes_)
    aligned = np.zeros((proba.shape[0], len(common_classes)), dtype=np.float64)
    positions = np.searchsorted(common_classes, est_classes)
    aligned[:, positions] = proba
    return aligned


class PrefitVotingClassifier(ClassifierMixin, BaseEstimator):
    """Голосование уже обученных моделей — без повторного обучения.

    Parameters
    ----------
    estimators : list of (str, object)
        Пары (имя, УЖЕ ОБУЧЕННАЯ модель). Каждая модель должна иметь
        classes_ и predict_proba(X) (sklearn-совместимые классификаторы,
        включая SubspaceConjugacyClassifier, подходят без адаптации).
    voting : {"soft", "hard"}, default="soft"
        "soft" — усредняются вероятности (predict_proba) с весами weights;
        "hard" — большинство голосов по predict() каждой модели, при
        равенстве голосов решает суммарный вес проголосовавших моделей, при
        полном равенстве — модель с наименьшим индексом в estimators.
    weights : Sequence[float], optional
        Веса моделей в том же порядке, что и estimators. None — равные веса.
        Нормируются к сумме 1 автоматически.

    Attributes
    ----------
    classes_ : np.ndarray or None
        Объединение classes_ всех моделей (после fit()).
    is_fitted_ : bool

    Notes
    -----
    fit(X=None, y=None) НЕ обучает базовые модели (они уже обучены) — лишь
    проверяет их состояние и собирает classes_. X, y можно не передавать.
    Именно поэтому этот класс НЕ предназначен для sklearn.model_selection.
    GridSearchCV/cross_val_score (они клонируют и переобучают эстиматор
    через clone() на каждом фолде — для prefit-моделей это бессмысленно:
    клон получит НЕобученные копии базовых моделей). Для честной оценки
    качества ансамбля используйте отдельный held-out test (как и в
    остальных экспериментах проекта, main.py::prepare_centered_split).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from subspace_conjugacy import SubspaceConjugacyClassifier
    >>> from subspace_conjugacy.models.ensemble import PrefitVotingClassifier
    >>> np.random.seed(0)
    >>> X_train = np.random.randn(60, 32)
    >>> y_train = np.array(["a", "b", "c"] * 20)
    >>> subspace = SubspaceConjugacyClassifier(n_subclasses=2).fit(X_train, y_train)
    >>> logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    >>> ens = PrefitVotingClassifier(
    ...     estimators=[("subspace", subspace), ("logreg", logreg)], voting="soft",
    ... ).fit()
    >>> preds = ens.predict(X_train[:5])
    """

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        voting: Literal["soft", "hard"] = "soft",
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

        self.classes_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> "PrefitVotingClassifier":
        if self.voting not in ("soft", "hard"):
            raise ValueError(f"voting должен быть 'soft' или 'hard', получено {self.voting!r}.")
        _check_prefit(self.estimators, "PrefitVotingClassifier")

        n_est = len(self.estimators)
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=np.float64)
            if len(w) != n_est:
                raise ValueError(
                    f"weights должен иметь длину {n_est} (по числу моделей), получено {len(w)}."
                )
        else:
            w = np.ones(n_est, dtype=np.float64)
        self._weights_ = w / w.sum()

        self.classes_ = _common_classes(self.estimators)
        self.is_fitted_ = True
        logger.info(
            "PrefitVotingClassifier.fit: %d моделей (%s), voting=%s, классы=%s.",
            n_est, [name for name, _ in self.estimators], self.voting, list(self.classes_),
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        proba_sum = np.zeros((np.asarray(X).shape[0], len(self.classes_)), dtype=np.float64)
        for w, (name, est) in zip(self._weights_, self.estimators):
            proba_sum += w * _aligned_proba(est, X, self.classes_)
        return proba_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        if self.voting == "soft":
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

        # hard voting: считаем взвешенные голоса по predict() каждой модели
        n_samples = np.asarray(X).shape[0]
        votes = np.zeros((n_samples, len(self.classes_)), dtype=np.float64)
        class_index = {cls: i for i, cls in enumerate(self.classes_)}
        for w, (name, est) in zip(self._weights_, self.estimators):
            pred = est.predict(X)
            for i, label in enumerate(pred):
                votes[i, class_index[label]] += w
        return self.classes_[np.argmax(votes, axis=1)]

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("PrefitVotingClassifier не обучен — вызовите fit() перед predict().")


class PrefitStackingClassifier(ClassifierMixin, BaseEstimator):
    """Метамодель (стекинг) поверх вероятностей уже обученных моделей.

    Parameters
    ----------
    estimators : list of (str, object)
        Пары (имя, УЖЕ ОБУЧЕННАЯ модель), как в PrefitVotingClassifier.
    meta_estimator : object, optional
        Классификатор, обучаемый на конкатенации вероятностей базовых
        моделей. По умолчанию — sklearn.linear_model.LogisticRegression
        (max_iter=1000) — простая, быстро обучаемая модель, достаточная для
        входа размерности "число моделей x число классов", без риска
        переобучения на небольшой валидационной выборке.

    Attributes
    ----------
    classes_ : np.ndarray or None
        classes_ обученной meta_estimator (после fit()).
    is_fitted_ : bool

    Notes
    -----
    fit(X_meta, y_meta) обучает ТОЛЬКО meta_estimator — базовые модели не
    трогаются. X_meta, y_meta ДОЛЖНЫ быть данными, НЕ использованными для
    обучения базовых моделей (отдельная валидационная выборка) — иначе
    метамодель обучится на переобученных, оптимистично смещённых
    вероятностях базовых моделей и её оценка на новом тесте будет
    завышенной. Тот же принцип честного holdout, что и в
    main.py::run_tuned_comparison_experiment (20% train используется
    ТОЛЬКО для подбора гиперпараметров/метамодели, финальный test —
    для оценки).

    Examples
    --------
    >>> ens = PrefitStackingClassifier(
    ...     estimators=[("subspace", subspace), ("logreg", logreg)],
    ... ).fit(X_val, y_val)
    >>> preds = ens.predict(X_test)
    """

    def __init__(
        self, estimators: List[Tuple[str, Any]], meta_estimator: Optional[Any] = None,
    ) -> None:
        self.estimators = estimators
        self.meta_estimator = meta_estimator

        self.classes_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrefitStackingClassifier":
        _check_prefit(self.estimators, "PrefitStackingClassifier")
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        self._common_classes_ = _common_classes(self.estimators)
        meta_features = np.hstack([
            _aligned_proba(est, X_arr, self._common_classes_) for _, est in self.estimators
        ])

        self._meta_estimator_ = (
            self.meta_estimator if self.meta_estimator is not None
            else LogisticRegression(max_iter=1000)
        )
        self._meta_estimator_.fit(meta_features, y_arr)
        self.classes_ = np.asarray(self._meta_estimator_.classes_)
        self.is_fitted_ = True
        logger.info(
            "PrefitStackingClassifier.fit: %d моделей, метамодель=%s, "
            "meta_features.shape=%s, классы=%s.",
            len(self.estimators), type(self._meta_estimator_).__name__,
            meta_features.shape, list(self.classes_),
        )
        return self

    def _meta_features(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([
            _aligned_proba(est, X, self._common_classes_) for _, est in self.estimators
        ])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self._meta_estimator_.predict_proba(self._meta_features(np.asarray(X, dtype=np.float64)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self._meta_estimator_.predict(self._meta_features(np.asarray(X, dtype=np.float64)))

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("PrefitStackingClassifier не обучен — вызовите fit(X_val, y_val) перед predict().")


class SwitchingEnsembleClassifier(ClassifierMixin, BaseEstimator):
    """Селективное переключение: для каждого класса — свой "чемпион" (план
    статьи 2, задача 2: "использовать сопряжённость только для классов, где
    она сильна").

    В отличие от голосования (где все модели участвуют в решении по КАЖДОМУ
    классу) и стекинга (где метамодель учится комбинировать вероятности
    произвольным образом), здесь для каждого класса c заранее (по
    валидационной выборке) выбирается ОДНА модель — та, что лучше всего
    (по метрике selection_metric) распознаёт именно класс c. Итоговая
    оценка объекта по классу c берётся ИСКЛЮЧИТЕЛЬНО у чемпиона этого
    класса, а не усредняется по всем моделям.

    Parameters
    ----------
    estimators : list of (str, object)
        Пары (имя, УЖЕ ОБУЧЕННАЯ модель), как в PrefitVotingClassifier.
    selection_metric : {"recall"}, default="recall"
        Метрика выбора чемпиона класса. Пока поддерживается только
        "recall" (= accuracy, ограниченная объектами этого класса,
        evaluation.metrics.per_class_accuracy) — доля объектов класса c,
        которых модель распознаёт верно. Параметр зарезервирован для
        будущих метрик (например, precision) без изменения сигнатуры.

    Attributes
    ----------
    classes_ : np.ndarray or None
        Классы, встретившиеся в y при fit().
    class_champion_ : Dict[Any, str] or None
        {класс: имя модели-чемпиона} — какая модель отвечает за каждый
        класс. Доступен после fit() для анализа/отчётности (например,
        чтобы явно показать в статье, какая модель "выиграла" каждый
        класс).
    champion_metric_ : Dict[Any, float] or None
        {класс: значение selection_metric чемпиона на валидационной
        выборке} — насколько уверенно чемпион выиграл у остальных моделей.
    is_fitted_ : bool

    Notes
    -----
    Как и PrefitStackingClassifier, fit(X_val, y_val) ожидает отдельную
    валидационную выборку, не использованную для обучения базовых моделей.

    Examples
    --------
    >>> ens = SwitchingEnsembleClassifier(
    ...     estimators=[("subspace", subspace), ("cnn", cnn_wrapper)],
    ... ).fit(X_val, y_val)
    >>> ens.class_champion_
    {'glioma': 'cnn', 'meningioma': 'subspace', 'pituitary': 'cnn'}
    """

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        selection_metric: Literal["recall"] = "recall",
    ) -> None:
        self.estimators = estimators
        self.selection_metric = selection_metric

        self.classes_: Optional[np.ndarray] = None
        self.class_champion_: Optional[dict] = None
        self.champion_metric_: Optional[dict] = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SwitchingEnsembleClassifier":
        if self.selection_metric != "recall":
            raise ValueError(
                f"selection_metric должен быть 'recall', получено {self.selection_metric!r}."
            )
        _check_prefit(self.estimators, "SwitchingEnsembleClassifier")
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        self._common_classes_ = _common_classes(self.estimators)
        self.classes_ = np.unique(y_arr)

        per_class_metric_by_model = {}
        for name, est in self.estimators:
            y_pred_val = est.predict(X_arr)
            per_class_metric_by_model[name] = per_class_accuracy(y_arr, y_pred_val)

        class_champion = {}
        champion_metric = {}
        for cls in self.classes_:
            best_name, best_score = None, -1.0
            for name, _ in self.estimators:
                score = per_class_metric_by_model[name].get(cls, 0.0)
                if score > best_score:
                    best_name, best_score = name, score
            class_champion[cls] = best_name
            champion_metric[cls] = best_score

        self.class_champion_ = class_champion
        self.champion_metric_ = champion_metric
        self.is_fitted_ = True
        logger.info(
            "SwitchingEnsembleClassifier.fit: %d классов, чемпионы=%s.",
            len(self.classes_), class_champion,
        )
        return self

    def _champion_combined_proba(self, X: np.ndarray) -> np.ndarray:
        """Матрица (M, len(classes_)): столбец класса c — из проекции
        predict_proba чемпиона класса c на common_classes_."""
        aligned_by_name = {
            name: _aligned_proba(est, X, self._common_classes_) for name, est in self.estimators
        }
        combined = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for col_idx, cls in enumerate(self.classes_):
            champion_name = self.class_champion_[cls]
            common_col_idx = int(np.searchsorted(self._common_classes_, cls))
            combined[:, col_idx] = aligned_by_name[champion_name][:, common_col_idx]
        return combined

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        combined = self._champion_combined_proba(np.asarray(X, dtype=np.float64))
        # Оценки разных чемпионов по разным классам в общем случае не
        # суммируются в 1 по строке (это не единая вероятностная модель) —
        # нормируем построчно для валидного API predict_proba; на predict()
        # (argmax) нормировка не влияет.
        row_sums = combined.sum(axis=1, keepdims=True)
        row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
        return combined / row_sums_safe

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        combined = self._champion_combined_proba(np.asarray(X, dtype=np.float64))
        return self.classes_[np.argmax(combined, axis=1)]

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "SwitchingEnsembleClassifier не обучен — вызовите fit(X_val, y_val) перед predict()."
            )
