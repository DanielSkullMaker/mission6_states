"""Гибридизация нескольких векторных представлений одних и тех же объектов
(статья 3, "Мультипредставительная гибридизация признакового пространства в
методе сопряжённости: горизонтальная и вертикальная развёртка изображения",
theory/article_plans/03_multipredstavitelnaya_gibridizatsiya.txt).

Направление развёртки изображения в вектор (горизонтальная — построчно,
вертикальная — по столбцам, features/vectorization.py) в библиотеке всегда
было зафиксировано (horizontal) без обоснования выбора. Задача плана
(раздел 5, шаг 1-2) — из ДВУХ (или более) представлений одного и того же
набора изображений построить ОДНО совместное решающее правило и сравнить его
с каждым представлением по отдельности.

План явно выделяет 2-3 способа гибридизации (раздел 2, задача 1), и только
ОДИН из них требует нового кода в библиотеке:

  1. Конкатенация векторов (N=2*65536) — НЕ требует нового класса: это просто
     ``np.hstack([X_horizontal, X_vertical])`` перед обычным
     ``SubspaceConjugacyClassifier.fit()``. Собирается в эксперименте.
  2. Раздельные подпространства с объединением критериев сопряжённости
     (R_hor(x, Y_hor) и R_ver(x, Y_ver) в одно решающее правило) — НЕТ
     готового механизма нигде в проекте: SubspaceConjugacyClassifier строит
     R-матрицу для ОДНОГО представления, а models/ensemble.py комбинирует
     уже готовые predict_proba (после softmax/argmax внутри каждой модели),
     а не сырые R ДО softmax. Это и есть пробел, который закрывает этот
     модуль — MultiRepresentationConjugacyClassifier.
  3. Позднее слияние (ансамбль двух НЕЗАВИСИМО обученных
     SubspaceConjugacyClassifier, каждый на своём представлении) — тоже НЕ
     требует нового кода: обе модели sklearn-совместимы (ClassifierMixin +
     BaseEstimator), поэтому подходит либо sklearn.ensemble.VotingClassifier/
     StackingClassifier напрямую, либо models.ensemble.PrefitVotingClassifier/
     PrefitStackingClassifier (если модели уже обучены и повторное обучение
     нежелательно) — пересечение со статьёй 2, models/ensemble.py.
"""

import logging
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from subspace_conjugacy.models.classifier import ClassLabel, SubspaceConjugacyClassifier

logger = logging.getLogger(__name__)

FusionRule = Literal["mean", "sum", "max", "weighted", "geometric_mean"]
_VALID_FUSION_RULES = {"mean", "sum", "max", "weighted", "geometric_mean"}
_GEOMETRIC_MEAN_EPS = 1e-12


# ClassifierMixin ДОЛЖЕН идти первым — та же причина, что и в
# SubspaceConjugacyClassifier (см. classifier.py, комментарий над классом):
# is_classifier() в современном sklearn резолвит __sklearn_tags__ через MRO,
# обратный порядок баз молча ломает StratifiedKFold в model_selection/.
class MultiRepresentationConjugacyClassifier(ClassifierMixin, BaseEstimator):
    """Объединяет показатели сопряжённости НЕСКОЛЬКИХ представлений объекта.

    Для каждого представления (например, "horizontal" и "vertical"
    развёртка одного изображения) обучается собственный независимый
    ``SubspaceConjugacyClassifier`` со своими подпространствами Y_{c,s}.
    Итоговое решающее правило объединяет матрицы R ПО ПРЕДСТАВЛЕНИЯМ ДО
    softmax/argmax:

        R_combined[i, c] = fuse( R_repr1[i, c], R_repr2[i, c], ... )
        y_pred[i] = argmax_c R_combined[i, c]

    В отличие от конкатенации признаков (единое подпространство строится на
    N1+N2 признаках сразу — теряется информация о том, какое представление
    "проголосовало") и от ансамбля предсказаний (models.ensemble —
    объединяются уже нормализованные вероятности ПОСЛЕ argmax внутри каждой
    модели), здесь у каждого представления остаётся собственное
    подпространство, а объединяется сырой, ещё не нормализованный критерий
    R — это позволяет посмотреть на вклад каждого представления в отдельности
    (predict_r_matrix_by_representation) и выбрать разные правила слияния.

    Parameters
    ----------
    fusion : {"mean", "sum", "max", "weighted", "geometric_mean"}, default="mean"
        Правило объединения R(x, Y) по представлениям (поэлементно, для
        каждого объекта и каждого класса отдельно):

        - "mean" — среднее арифметическое. Даёт тот же argmax, что и "sum"
          (совпадают с точностью до постоянного множителя = число
          представлений) — оставлены оба варианта только для читаемости
          промежуточных значений в диагностике/отчётах.
        - "sum" — сумма без нормировки.
        - "max" — максимум по представлениям: класс побеждает, если ХОТЯ БЫ
          ОДНО представление уверенно его выделяет, даже если остальные
          представления путаются. Годится, если представления ожидаемо
          "специализируются" на разных классах (план, раздел 2, задача 3:
          "если разные патологии по-разному выражены геометрически, эффект
          гибридизации должен быть неравномерным по классам").
        - "weighted" — взвешенная сумма с весами `weights` (по умолчанию,
          если `weights` не задан, совпадает с "mean"/"sum" — все веса
          равны). Полезно, если одно представление заведомо надёжнее
          (например, по итогам предварительной оценки на валидации).
        - "geometric_mean" — среднее геометрическое. В отличие от "mean",
          чувствительно к САМОМУ СЛАБОМУ представлению: если хотя бы одно
          представление даёт R≈0 для класса, итоговое значение тоже близко
          к 0, даже когда остальные представления уверены — модель "не
          доверяет" классу, в котором согласны не все представления.
    weights : Mapping[str, float], optional
        Веса представлений при fusion="weighted" — ключи должны совпадать с
        именами представлений, переданными в fit()/predict() (X_by_representation).
        Нормируются автоматически (сумма весов не обязана быть равна 1).
        Игнорируется при других значениях fusion.
    n_subclasses : int or Mapping[ClassLabel, int], default=8
        Передаётся в SubspaceConjugacyClassifier КАЖДОГО представления без
        изменений (см. classifier.py). Если представлениям требуются разные
        гиперпараметры — обучите SubspaceConjugacyClassifier отдельно для
        каждого представления и соберите этот класс через
        `from_fitted_estimators`, а не через fit().
    freeze_basis_at : int or "auto" or None, default=2
        Передаётся в SubspaceConjugacyClassifier каждого представления без
        изменений. ⚠ При freeze_basis_at="auto" равнение размера базиса
        (equalize_subspace_bases) выполняется НЕЗАВИСИМО внутри каждого
        представления — итоговый k может отличаться между представлениями
        (например, k_horizontal=3, k_vertical=5), см. Notes.
    growth_strategy : {"default", "master"}, default="default"
        Передаётся в SubspaceConjugacyClassifier каждого представления.
    reg_param : float, default=1e-8
        Передаётся в SubspaceConjugacyClassifier каждого представления.

    Attributes
    ----------
    classes_ : np.ndarray
        Массив уникальных меток классов (берётся с любого представления —
        после fit() совпадает у всех, т.к. все обучены на одном и том же y).
    representations_ : List[str]
        Имена представлений в порядке, зафиксированном при fit() (или
        порядке ключей словаря, переданного в from_fitted_estimators).
    estimators_by_representation_ : Dict[str, SubspaceConjugacyClassifier]
        Обученная модель для каждого представления — доступна для
        индивидуальной диагностики (например, estimators_by_representation_
        ["vertical"].predict(X_vertical) — что предсказало бы ТОЛЬКО
        вертикальное представление).

    Notes
    -----
    R(x, Y) не масштабируется по k (числу столбцов базиса) — тот же нюанс,
    что и в SubspaceConjugacyClassifier.predict_r_matrix (см. classifier.py).
    Если freeze_basis_at="auto" и разные представления "выросли" до разного
    k (например, горизонтальная развёртка оказалась более разделимой и
    остановилась на большем базисе, чем вертикальная), их R-шкалы не
    гарантированно сопоставимы — fusion="max" в этом случае может
    систематически предпочитать представление с бОльшим k, а не то,
    которое действительно увереннее. Чтобы исключить этот эффект, задайте
    freeze_basis_at целым числом (одинаковый k у всех представлений) либо
    сравните equalized_basis_size_ обученных моделей в
    estimators_by_representation_.

    Examples
    --------
    >>> from subspace_conjugacy.features import vectorize_batch
    >>> X_hor = vectorize_batch(images, method="horizontal")
    >>> X_ver = vectorize_batch(images, method="vertical")
    >>> clf = MultiRepresentationConjugacyClassifier(fusion="mean", n_subclasses=8)
    >>> clf.fit({"horizontal": X_hor_train, "vertical": X_ver_train}, y_train)
    >>> y_pred = clf.predict({"horizontal": X_hor_test, "vertical": X_ver_test})
    """

    def __init__(
        self,
        fusion: FusionRule = "mean",
        weights: Optional[Mapping[str, float]] = None,
        n_subclasses: Union[int, Mapping[ClassLabel, int]] = 8,
        freeze_basis_at: Union[int, str, None] = 2,
        growth_strategy: str = "default",
        reg_param: float = 1e-8,
    ) -> None:
        self.fusion = fusion
        self.weights = weights
        self.n_subclasses = n_subclasses
        self.freeze_basis_at = freeze_basis_at
        self.growth_strategy = growth_strategy
        self.reg_param = reg_param
        self.classes_: Optional[np.ndarray] = None
        self.representations_: Optional[List[str]] = None
        self.estimators_by_representation_: Dict[str, SubspaceConjugacyClassifier] = {}
        self.is_fitted_: bool = False

    def _validate_fusion(self) -> None:
        if self.fusion not in _VALID_FUSION_RULES:
            logger.error(
                "MultiRepresentationConjugacyClassifier: fusion=%r недопустим.",
                self.fusion,
            )
            raise ValueError(
                f"fusion должен быть одним из {sorted(_VALID_FUSION_RULES)}, "
                f"получено {self.fusion!r}."
            )

    def _make_base_estimator(self) -> SubspaceConjugacyClassifier:
        return SubspaceConjugacyClassifier(
            n_subclasses=self.n_subclasses,
            freeze_basis_at=self.freeze_basis_at,
            growth_strategy=self.growth_strategy,
            reg_param=self.reg_param,
        )

    def fit(
        self, X_by_representation: Mapping[str, np.ndarray], y: np.ndarray
    ) -> "MultiRepresentationConjugacyClassifier":
        """Обучает по одному SubspaceConjugacyClassifier на представление.

        Parameters
        ----------
        X_by_representation : Mapping[str, np.ndarray]
            Словарь {имя_представления: X}, минимум 2 представления. Все
            матрицы X должны иметь одинаковое число строк M (объекты в
            одинаковом порядке — строка i в каждом представлении должна
            соответствовать ОДНОМУ И ТОМУ ЖЕ объекту); число признаков N
            может отличаться между представлениями.
        y : np.ndarray
            Вектор меток классов размерности (M,), общий для всех
            представлений.

        Returns
        -------
        self : MultiRepresentationConjugacyClassifier

        Raises
        ------
        ValueError
            Если передано меньше 2 представлений, или число объектов (строк)
            не совпадает между представлениями/y.
        """
        self._validate_fusion()
        if len(X_by_representation) < 2:
            logger.error(
                "MultiRepresentationConjugacyClassifier.fit: передано %d "
                "представлений, требуется минимум 2.", len(X_by_representation),
            )
            raise ValueError(
                "X_by_representation должен содержать минимум 2 представления, "
                f"получено {len(X_by_representation)}."
            )

        y_arr = np.asarray(y)
        n_samples_by_repr = {
            name: np.asarray(X).shape[0] for name, X in X_by_representation.items()
        }
        mismatched = {
            name: n for name, n in n_samples_by_repr.items() if n != y_arr.shape[0]
        }
        if mismatched:
            logger.error(
                "MultiRepresentationConjugacyClassifier.fit: несовпадение числа "
                "объектов — y содержит %d, представления: %s.",
                y_arr.shape[0], mismatched,
            )
            raise ValueError(
                f"Число объектов должно совпадать во всех представлениях и в y "
                f"(y: {y_arr.shape[0]}), получены несовпадения: {mismatched}."
            )

        representations = list(X_by_representation.keys())
        logger.info(
            "MultiRepresentationConjugacyClassifier.fit: старт, представления=%s, "
            "fusion=%s, %d объектов.", representations, self.fusion, y_arr.shape[0],
        )

        estimators: Dict[str, SubspaceConjugacyClassifier] = {}
        for name in representations:
            logger.info(
                "MultiRepresentationConjugacyClassifier.fit: обучение "
                "представления '%s' (N=%d признаков).",
                name, np.asarray(X_by_representation[name]).shape[1],
            )
            estimator = self._make_base_estimator()
            estimator.fit(X_by_representation[name], y_arr)
            estimators[name] = estimator

        self.representations_ = representations
        self.estimators_by_representation_ = estimators
        self.classes_ = estimators[representations[0]].classes_
        self.is_fitted_ = True
        logger.info(
            "MultiRepresentationConjugacyClassifier.fit: готово, %d представлений, "
            "%d классов.", len(representations), len(self.classes_),
        )
        return self

    @classmethod
    def from_fitted_estimators(
        cls,
        estimators_by_representation: Mapping[str, SubspaceConjugacyClassifier],
        fusion: FusionRule = "mean",
        weights: Optional[Mapping[str, float]] = None,
    ) -> "MultiRepresentationConjugacyClassifier":
        """Собирает модель из уже обученных SubspaceConjugacyClassifier.

        Полезно, если представлениям нужны РАЗНЫЕ гиперпараметры (например,
        разное n_subclasses на представление) — fit() применяет ОДНИ и те же
        параметры конструктора ко всем представлениям, а этот метод
        позволяет их обучить независимо и объединить постфактум.

        Parameters
        ----------
        estimators_by_representation : Mapping[str, SubspaceConjugacyClassifier]
            Словарь {имя_представления: уже обученный SubspaceConjugacyClassifier},
            минимум 2 представления. Все модели должны иметь одинаковый
            набор классов (classes_).
        fusion : {"mean", "sum", "max", "weighted", "geometric_mean"}, default="mean"
            См. docstring класса.
        weights : Mapping[str, float], optional
            См. docstring класса.

        Returns
        -------
        instance : MultiRepresentationConjugacyClassifier

        Raises
        ------
        ValueError
            Если передано меньше 2 представлений, какая-то модель не
            обучена (нет classes_), или наборы классов у моделей различаются.
        """
        if len(estimators_by_representation) < 2:
            logger.error(
                "MultiRepresentationConjugacyClassifier.from_fitted_estimators: "
                "передано %d моделей, требуется минимум 2.",
                len(estimators_by_representation),
            )
            raise ValueError(
                "estimators_by_representation должен содержать минимум 2 "
                f"представления, получено {len(estimators_by_representation)}."
            )

        classes_by_repr = {}
        for name, estimator in estimators_by_representation.items():
            if not getattr(estimator, "is_fitted_", False):
                logger.error(
                    "MultiRepresentationConjugacyClassifier.from_fitted_estimators: "
                    "представление '%s' не обучено.", name,
                )
                raise ValueError(
                    f"Модель представления '{name}' ещё не обучена — вызовите "
                    f"её fit() перед передачей сюда."
                )
            classes_by_repr[name] = tuple(np.asarray(estimator.classes_).tolist())

        distinct_class_sets = set(classes_by_repr.values())
        if len(distinct_class_sets) != 1:
            logger.error(
                "MultiRepresentationConjugacyClassifier.from_fitted_estimators: "
                "разные наборы классов между представлениями: %s.", classes_by_repr,
            )
            raise ValueError(
                "Все представления должны быть обучены на одном и том же "
                f"наборе классов, получено: {classes_by_repr}."
            )

        instance = cls(fusion=fusion, weights=weights)
        instance._validate_fusion()
        instance.representations_ = list(estimators_by_representation.keys())
        instance.estimators_by_representation_ = dict(estimators_by_representation)
        first = next(iter(estimators_by_representation.values()))
        instance.classes_ = first.classes_
        instance.is_fitted_ = True
        logger.info(
            "MultiRepresentationConjugacyClassifier.from_fitted_estimators: собрано "
            "из %d уже обученных представлений: %s.",
            len(instance.representations_), instance.representations_,
        )
        return instance

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            logger.error(
                "MultiRepresentationConjugacyClassifier: обращение к "
                "предсказаниям до fit()."
            )
            raise RuntimeError(
                "Экземпляр модели ещё не обучен. Вызовите fit() (или "
                "from_fitted_estimators()) перед предсказаниями."
            )

    def _check_representations(self, X_by_representation: Mapping[str, Any]) -> None:
        missing = [r for r in self.representations_ if r not in X_by_representation]
        if missing:
            logger.error(
                "MultiRepresentationConjugacyClassifier: в X_by_representation "
                "отсутствуют представления %s (ожидались %s).",
                missing, self.representations_,
            )
            raise ValueError(
                f"X_by_representation не содержит представлений {missing}, "
                f"ожидались (из fit()): {self.representations_}."
            )

    def _resolve_weights(self) -> Dict[str, float]:
        if self.weights is None:
            return {name: 1.0 for name in self.representations_}
        missing = [name for name in self.representations_ if name not in self.weights]
        if missing:
            logger.error(
                "MultiRepresentationConjugacyClassifier: weights не содержит "
                "значений для представлений %s.", missing,
            )
            raise ValueError(
                f"fusion='weighted' требует веса для всех представлений "
                f"{self.representations_}, отсутствуют: {missing}."
            )
        return {name: float(self.weights[name]) for name in self.representations_}

    def predict_r_matrix_by_representation(
        self, X_by_representation: Mapping[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """R-матрица (Фаза C) КАЖДОГО представления по отдельности, без слияния.

        Полезно для диагностики: сравнить, что предсказало бы каждое
        представление само по себе (план, задача 3 — "разбить результат по
        классам, проверить гипотезу о неравномерном эффекте").

        Parameters
        ----------
        X_by_representation : Mapping[str, np.ndarray]
            Словарь {имя_представления: X}, должен содержать все
            представления, известные модели после fit() (self.representations_).

        Returns
        -------
        r_by_representation : Dict[str, np.ndarray]
            {имя_представления: R_matrix (M, n_classes)}.
        """
        self._check_is_fitted()
        self._check_representations(X_by_representation)
        return {
            name: self.estimators_by_representation_[name].predict_r_matrix(
                X_by_representation[name]
            )
            for name in self.representations_
        }

    def predict_r_matrix(self, X_by_representation: Mapping[str, np.ndarray]) -> np.ndarray:
        """Объединённая (fused) R-матрица по всем представлениям.

        Parameters
        ----------
        X_by_representation : Mapping[str, np.ndarray]
            Словарь {имя_представления: X}, см. predict_r_matrix_by_representation.

        Returns
        -------
        R_combined : np.ndarray
            Матрица размерности (M, n_classes) — результат применения
            правила fusion к R-матрицам всех представлений.
        """
        r_by_repr = self.predict_r_matrix_by_representation(X_by_representation)
        stacked = np.stack(
            [r_by_repr[name] for name in self.representations_], axis=0
        )  # (n_representations, M, n_classes)

        if self.fusion == "mean":
            combined = np.mean(stacked, axis=0)
        elif self.fusion == "sum":
            combined = np.sum(stacked, axis=0)
        elif self.fusion == "max":
            combined = np.max(stacked, axis=0)
        elif self.fusion == "geometric_mean":
            clipped = np.clip(stacked, _GEOMETRIC_MEAN_EPS, None)
            combined = np.exp(np.mean(np.log(clipped), axis=0))
        elif self.fusion == "weighted":
            weights = self._resolve_weights()
            w = np.array(
                [weights[name] for name in self.representations_], dtype=np.float64
            ).reshape(-1, 1, 1)
            weight_sum = np.sum(w)
            if weight_sum <= 0:
                logger.error(
                    "MultiRepresentationConjugacyClassifier.predict_r_matrix: "
                    "сумма весов <= 0 (%s).", weights,
                )
                raise ValueError(
                    f"Сумма весов представлений должна быть положительной, "
                    f"получено {weights}."
                )
            combined = np.sum(stacked * w, axis=0) / weight_sum
        else:  # pragma: no cover - перехвачено _validate_fusion при fit()
            raise ValueError(f"Неизвестное правило fusion: {self.fusion!r}.")

        logger.debug(
            "MultiRepresentationConjugacyClassifier.predict_r_matrix: "
            "%d представлений (fusion=%s) -> R_combined.shape=%s.",
            len(self.representations_), self.fusion, combined.shape,
        )
        return combined

    def predict_proba(self, X_by_representation: Mapping[str, np.ndarray]) -> np.ndarray:
        """Softmax объединённой R-матрицы в вероятности по классам."""
        R_combined = self.predict_r_matrix(X_by_representation)
        exp_r = np.exp(R_combined - np.max(R_combined, axis=1, keepdims=True))
        return exp_r / np.sum(exp_r, axis=1, keepdims=True)

    def predict(self, X_by_representation: Mapping[str, np.ndarray]) -> np.ndarray:
        """Предсказывает класс с наивысшим объединённым показателем сопряжённости."""
        R_combined = self.predict_r_matrix(X_by_representation)
        best_indices = np.argmax(R_combined, axis=1)
        y_pred = self.classes_[best_indices]
        logger.info(
            "MultiRepresentationConjugacyClassifier.predict: %d объектов -> "
            "распределение по классам %s.",
            len(y_pred), dict(zip(*np.unique(y_pred, return_counts=True))),
        )
        return y_pred
