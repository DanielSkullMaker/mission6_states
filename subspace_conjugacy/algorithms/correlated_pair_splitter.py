"""Correlated Pair Splitter — "Первый этап" из черновика новой статьи.

Источник (НЕ опубликованная статья, черновик — theory/Макет новой статьи.docx,
раздел "Формирование обучающих подпространств...", "Первый этап"):

  "Пусть X — исходное множество размеченных векторов, соответствующих
  изображениям некоторого класса (m четно). Для произвольно заданного
  начального вектора... на этом множестве определяется вектор, для которого
  нормированный коэффициент корреляции [максимален]. Каждый из этой пары
  векторов заносится в «своё» подмножество (A, B). Процедура повторяется на
  множестве оставшихся доступных векторов до их исчерпания. В результате
  доступное множество будет разделено на два подмножества (A, B) «похожих»
  векторов. Любое из этих подмножеств может использоваться для формирования
  обучающего подпространства данного класса."

В отличие от опубликованной статьи (theory/VI_Korshikov_VA_Fursov_...docx) и
уже реализованного LinearDependencyFilter (algorithms/reference_filter.py,
находка №3 в refactoring_plan.txt), этот алгоритм — из ДРУГОГО, незавершённого
черновика (про КТ грудной клетки, а не МРТ мозга) и использует ИНОЙ механизм:

  - LinearDependencyFilter: последовательно накапливает "принятое"
    подпространство Y_accepted и ИСКЛЮЧАЕТ вектор, если R(x, Y_accepted) >=
    threshold — однонаправленная фильтрация (keep/exclude), результат
    зависит от порогового значения.
  - CorrelatedPairSplitter (этот модуль): итеративно ищет пары НАИБОЛЕЕ
    похожих (а не непохожих, как в фазе A.1) векторов по косинусному
    сходству и делит каждую пару пополам между двумя подмножествами — без
    порога, без отбрасывания информации о классе, результат — два
    равноценных (по докладу автора) подмножества размера ~M/2 каждое.

Решения, принятые при реализации неоднозначностей черновика:

  1. Формула (1) черновика ("нормированный коэффициент корреляции") в
     извлечённом тексте отсутствует (встроенный объект формулы, не
     распознанный при извлечении текста из .docx) — по терминологии,
     совпадающей с формулой (3) опубликованной статьи ("normalized
     correlation coefficient" = cosine similarity) и с первым шагом
     документа "Кластеризация по критерию сопряжённости.docx" (тот же
     "косинусный критерий"), здесь используется cosine_similarity_matrix
     из core/metrics.py — та же метрика, что и в фазе A.1 канона, но ищется
     МАКСИМУМ (похожесть), а не минимум (непохожесть).
  2. "Произвольно заданный начальный вектор" — черновик не уточняет способ
     выбора. Здесь используется детерминированный порядок: на каждой
     итерации начальным берётся вектор с наименьшим ещё не использованным
     исходным индексом — воспроизводимость важнее буквального соответствия
     недоопределённому "произвольно".
  3. "m чётно" — черновик явно предполагает чётное число векторов. При
     нечётном M один вектор остаётся без пары; он не включается ни в одно
     из двух подмножеств (доступен для диагностики в unpaired_index_), а не
     произвольно приписывается к одному из них.
  4. "Любое из подмножеств может использоваться" — оставлено на усмотрение
     вызывающего кода: fit() строит оба (subset_a_indices_/
     subset_b_indices_), transform(X, subset="a"|"b") выбирает нужное.

Связь с остальной библиотекой (опционально, по умолчанию выключено):
  FursovClusterer(split_correlated_pairs=True, correlated_pairs_subset="a")
  применяет этот сплиттер как необязательную ФАЗУ 0c — ПОСЛЕ
  LowInformativenessFilter (0a) и LinearDependencyFilter (0b), ПЕРЕД фазой
  A.1 — на том, что осталось после 0a/0b (см. fursov_clusterer.py).
"""

import logging
from typing import List, Optional
import numpy as np

from subspace_conjugacy.core.metrics import cosine_similarity_matrix

logger = logging.getLogger(__name__)


class CorrelatedPairSplitter:
    """Делит множество векторов на два подмножества "похожих" пар.

    Итеративно (в порядке возрастания ещё не использованного индекса)
    находит для очередного вектора наиболее похожий (максимум косинусного
    сходства) среди оставшихся, заносит первый в подмножество A, второй — в
    подмножество B, и убирает оба из рассмотрения. Повторяет до исчерпания
    множества (при нечётном M один вектор остаётся без пары).

    Parameters
    ----------
    None

    Attributes
    ----------
    pairs_ : np.ndarray or None
        Массив пар (n_pairs, 2) — (i, j) в порядке нахождения, i из
        подмножества A, j — из подмножества B.
    subset_a_indices_ : np.ndarray or None
        Индексы (в исходном X, отсортированы по возрастанию) первого
        подмножества, длина = n_pairs.
    subset_b_indices_ : np.ndarray or None
        Индексы (в исходном X, отсортированы по возрастанию) второго
        подмножества, длина = n_pairs.
    unpaired_index_ : int or None
        Индекс вектора, оставшегося без пары при нечётном M. None, если M
        чётно (или M == 0, что невозможно — пустой X отклоняется в fit()).
    is_fitted_ : bool
        Флаг, указывающий, был ли выполнен fit().

    Examples
    --------
    >>> import numpy as np
    >>> from subspace_conjugacy.algorithms.correlated_pair_splitter import (
    ...     CorrelatedPairSplitter,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 64))
    >>> splitter = CorrelatedPairSplitter()
    >>> splitter.fit(X)
    >>> len(splitter.subset_a_indices_) == len(splitter.subset_b_indices_) == 10
    True
    >>> X_a = splitter.transform(X, subset="a")
    """

    def __init__(self) -> None:
        self.pairs_: Optional[np.ndarray] = None
        self.subset_a_indices_: Optional[np.ndarray] = None
        self.subset_b_indices_: Optional[np.ndarray] = None
        self.unpaired_index_: Optional[int] = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray) -> "CorrelatedPairSplitter":
        """Разбивает X на два подмножества похожих пар.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N), M >= 1.

        Returns
        -------
        self : CorrelatedPairSplitter
            Возвращает экземпляр самого себя.

        Raises
        ------
        ValueError
            Если X не 2D или пуст.
        """
        X_arr = self._validate_input(X)
        n_samples = X_arr.shape[0]
        logger.info(
            "CorrelatedPairSplitter.fit: старт, %d векторов, N=%d.",
            n_samples, X_arr.shape[1],
        )

        sim_matrix = cosine_similarity_matrix(X_arr)

        used = np.zeros(n_samples, dtype=bool)
        pairs: List[tuple] = []
        subset_a: List[int] = []
        subset_b: List[int] = []

        for i in range(n_samples):
            if used[i]:
                continue

            candidate_mask = ~used
            candidate_mask[i] = False
            if not candidate_mask.any():
                # Последний, непарный вектор при нечётном M — не включаем
                # ни в одно из подмножеств (used[i] остаётся False).
                continue

            sims = np.where(candidate_mask, sim_matrix[i], -np.inf)
            j = int(np.argmax(sims))

            used[i] = True
            used[j] = True
            pairs.append((i, j))
            subset_a.append(i)
            subset_b.append(j)
            logger.debug(
                "CorrelatedPairSplitter.fit: пара (%d, %d), cos=%.6f.",
                i, j, sim_matrix[i, j],
            )

        leftover = np.where(~used)[0]
        unpaired_index = int(leftover[0]) if leftover.size > 0 else None

        self.pairs_ = np.array(pairs, dtype=int) if pairs else np.zeros((0, 2), dtype=int)
        self.subset_a_indices_ = np.sort(np.array(subset_a, dtype=int))
        self.subset_b_indices_ = np.sort(np.array(subset_b, dtype=int))
        self.unpaired_index_ = unpaired_index
        self.is_fitted_ = True

        logger.info(
            "CorrelatedPairSplitter.fit: готово, %d пар (%d + %d векторов)%s.",
            len(pairs), len(subset_a), len(subset_b),
            f", непарный вектор: {unpaired_index}" if unpaired_index is not None else "",
        )
        return self

    def transform(self, X: np.ndarray, subset: str = "a") -> np.ndarray:
        """Возвращает выбранное подмножество векторов.

        Parameters
        ----------
        X : np.ndarray
            Матрица (M, N), на которой был выполнен fit() (или с тем же
            порядком строк).
        subset : {"a", "b"}, default="a"
            Какое из двух подмножеств вернуть — обе равноценны согласно
            черновику ("любое из этих подмножеств может использоваться").

        Returns
        -------
        X_subset : np.ndarray
            Подматрица (n_pairs, N).

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        ValueError
            Если subset не "a" и не "b".
        """
        self._check_is_fitted()
        if subset not in ("a", "b"):
            raise ValueError(f"subset должен быть 'a' или 'b', получено {subset!r}.")

        X_arr = np.asarray(X, dtype=np.float64)
        indices = self.subset_a_indices_ if subset == "a" else self.subset_b_indices_
        return X_arr[indices]

    def fit_transform(self, X: np.ndarray, subset: str = "a") -> np.ndarray:
        """Эквивалент fit(X).transform(X, subset) за один проход.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов (M, N).
        subset : {"a", "b"}, default="a"
            См. transform().

        Returns
        -------
        X_subset : np.ndarray
            Подматрица выбранного подмножества.
        """
        self.fit(X)
        return self.transform(X, subset=subset)

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Валидирует входную матрицу X."""
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim != 2:
            raise ValueError(
                f"Ожидалась 2D матрица векторов, получена {X_arr.ndim}D."
            )

        if X_arr.shape[0] == 0:
            raise ValueError("Передана пустая матрица X.")

        return X_arr

    def _check_is_fitted(self) -> None:
        """Проверяет, был ли вызван fit()."""
        if not self.is_fitted_:
            logger.error("CorrelatedPairSplitter: обращение к результатам до fit().")
            raise RuntimeError(
                "Модель не обучена. Вызовите fit(X) перед использованием."
            )

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"CorrelatedPairSplitter(pairs={len(self.pairs_)}, "
                f"unpaired={self.unpaired_index_})"
            )
        return "CorrelatedPairSplitter(not fitted)"
