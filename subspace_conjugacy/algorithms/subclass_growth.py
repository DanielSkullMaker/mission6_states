"""Canonical Algorithm B.2: Conjugacy Cluster Growth.

Теория (секция 1.3, шаг B.2 из refactoring_plan.txt):
  После формирования начальных пар (фаза B.1), оставшиеся векторы
  последовательно присоединяются к подклассам на основе максимума
  показателя сопряжённости R(x, Y_s).

  ШАГ B.2 — Conjugacy growth (наполнение кластеров):
    Условие входа: в каждом подклассе уже ≥ 2 вектора (после B.1 выполнено).
    Пока remaining не пуст:
      1. Для каждого x ∈ remaining и каждого подкласса s:
           вычислить R(x, Y_s), где Y_s — текущая базисная матрица (N×k_s).
      2. Найти пару (x*, s*) с МАКСИМАЛЬНЫМ R(x*, Y_{s*}).
      3. Присоединить x* к подклассу s*, обновить Y_{s*}.
      4. Удалить x* из remaining.

    Критично: присоединять СТРОГО ПО ОДНОМУ вектору за итерацию (глобальный argmax
    по всем remaining × всем подклассам). Нельзя bulk-assign множество в один кластер.

Связь с ноутбуками:
  NB7 (7_2_Fursov_fulfilling_new_subclasses): схожая логика, но с ratio к среднему.

  ⚠ strategy="master" — это КАНОН-СОВМЕСТИМАЯ АППРОКСИМАЦИЯ идеи NB7
  (ratio R(x,s)/mean(R(x, others)) при отборе), а НЕ побитовая реплика NB7.
  Настоящий NB7 (7_2_..., cell 11) на каждом проходе ``for idea in
  range(count_num)`` назначает КАЖДОМУ подклассу СВОЙ лучший вектор
  одновременно (S присвоений за один "idea"-шаг, без учёта векторов,
  занятых другими подклассами в этом же шаге). Здесь же, чтобы не нарушать
  инвариант канона B.2 ("строго по одному вектору за итерацию" — см. выше),
  _find_argmax_master ищет ОДНО глобальное (x*, s*) по ratio среди ВСЕХ
  remaining×подклассов за одну итерацию while. Порядок и итоговое
  распределение векторов по подклассам поэтому могут отличаться от
  настоящего NB7 — за побитовым воспроизведением NB7 обращайтесь к
  algorithms/legacy/notebook_pipeline.py, а не к strategy="master".

Связь с другими модулями:
  Требует pairs из CosineSecondVectorAttacher (фаза B.1).
  Выход используется в FursovClusterer и экспортируется для classifier.
"""

import logging
from typing import List, Literal, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    from subspace_conjugacy.core.metrics import conjugate_criterion
except ImportError:
    # Fallback для автономного запуска
    def conjugate_criterion(x: np.ndarray, Y: np.ndarray, reg_param: float = 1e-8) -> float:
        """Вычисляет R(x, Y) = (x^T Y (Y^T Y)^{-1} Y^T x) / (x^T x)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        gram = Y.T @ Y
        gram_reg = gram + reg_param * np.eye(gram.shape[0])
        gram_inv = np.linalg.inv(gram_reg)

        projection = Y @ gram_inv @ Y.T @ x.T
        R = np.sum(x * projection.T, axis=1) / np.sum(x * x, axis=1)
        return np.clip(R, 0.0, 1.0)


class ConjugacyClusterGrowth:
    """Последовательное наполнение кластеров через максимум R (B.2).

    Этот алгоритм распределяет оставшиеся векторы по подклассам,
    присоединяя на каждой итерации ровно один вектор к подклассу
    с максимальным показателем сопряжённости.

    Parameters
    ----------
    freeze_basis_at : int or None, default=2
        Ограничение размерности базисов для финального экспорта.
        Если 2 — после fit базисы будут содержать только первые 2 столбца.
        Если None — базисы растут без ограничений.
    strategy : {"default", "master"}, default="default"
        Стратегия выбора вектора:
        - "default": простой argmax R(x, Y_s)
        - "master": ratio R(x, s*) / mean(R(x, others)) — канон-совместимая
          аппроксимация идеи NB7 (см. предупреждение в docstring модуля),
          НЕ побитовая реплика NB7
    reg_param : float, default=1e-8
        Параметр регуляризации Тихонова для (Y^T Y)^{-1}.
    early_stopping : {"relative_drop", "fixed_fraction"} or None, default=None
        Критерий ранней остановки итерационного роста (статья "Итерационное
        наращивание опорных подпространств...", theory/article_plans/01_
        iterativnyi_algoritm.txt, задача 4). Векторы, не успевшие
        присоединиться к моменту остановки, остаются с меткой -1 (см.
        excluded_by_early_stopping_) — как и у остальных фильтров
        библиотеки (LinearDependencyFilter, LowInformativenessFilter).
        - None (по умолчанию): рост идёт до исчерпания remaining — текущее
          поведение канона B.2, без изменений.
        - "relative_drop": остановка, когда наилучшее R текущей итерации
          падает ниже ``early_stopping_threshold * R_первой_итерации`` —
          сигнал, что легко сопряжённые векторы уже присоединены и
          начинаются «дорогие» присоединения с падающим качеством.
        - "fixed_fraction": остановка после присоединения доли
          ``early_stopping_threshold`` от общего числа векторов,
          подлежащих распределению (простой бюджетный критерий).
    early_stopping_threshold : float, default=0.1
        Порог для выбранного early_stopping. Для "relative_drop" — доля от
        R первой итерации (0.1 = остановиться, когда R упадёт ниже 10% от
        начального значения). Для "fixed_fraction" — доля от общего числа
        векторов, подлежащих распределению (0.1 = присоединить только 10%
        и остановиться). Игнорируется, если early_stopping is None.
        Должен быть в (0, 1].

    Attributes
    ----------
    labels_ : np.ndarray or None
        Метки подклассов для каждого вектора (M,). Векторы, оставшиеся
        неразмеченными из-за ранней остановки, имеют метку -1.
    subspace_bases_ : list[np.ndarray] or None
        Финальные базисы подпространств (N, k).
        Если freeze_basis_at задан, k = freeze_basis_at.
    n_subclasses_ : int or None
        Количество подклассов.
    growth_history_ : list[dict] or None
        История роста по итерациям — заполняется, только если
        fit(..., store_history=True). Каждая запись:
        {"iteration", "vector_index", "subclass_index", "r_value",
        "basis_size", "gram_condition_number", "n_remaining_after"}.
        Прямой источник кривых R(k) и обусловленности (Y^T Y) для анализа
        сходимости (см. get_growth_curve()).
    excluded_by_early_stopping_ : np.ndarray or None
        Индексы векторов, оставшихся неразмеченными из-за ранней
        остановки. Пустой массив, если early_stopping is None либо
        критерий не сработал (рост дошёл до конца естественным образом).
    stopped_early_ : bool
        True, если рост был прерван критерием ранней остановки до
        исчерпания remaining.
    is_fitted_ : bool
        Флаг, указывающий, был ли выполнен fit().

    Examples
    --------
    >>> import numpy as np
    >>> from subspace_conjugacy.algorithms import (
    ...     GlobalMinCosinePairFinder,
    ...     ReferenceCenterBuilder,
    ...     CosineSecondVectorAttacher,
    ...     ConjugacyClusterGrowth,
    ... )
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 256)
    >>>
    >>> # Фазы A.1 → A.3 → B.1
    >>> pair_finder = GlobalMinCosinePairFinder()
    >>> pair_finder.fit(X)
    >>>
    >>> builder = ReferenceCenterBuilder(n_subclasses=8)
    >>> builder.fit(X, pair_finder.pair_indices_)
    >>>
    >>> attacher = CosineSecondVectorAttacher()
    >>> attacher.fit(X, builder.center_indices_)
    >>> pairs = attacher.pairs_
    >>>
    >>> # Фаза B.2: наполнение кластеров
    >>> growth = ConjugacyClusterGrowth(freeze_basis_at=2)
    >>> growth.fit(X, pairs)
    >>> labels = growth.labels_
    >>> bases = growth.subspace_bases_
    >>> print(f"Все векторы распределены: {len(set(labels)) == 8}")
    >>> print(f"Базисы для классификации: {all(Y.shape[1] == 2 for Y in bases)}")
    """

    def __init__(
        self,
        freeze_basis_at: Optional[int] = 2,
        strategy: Literal["default", "master"] = "default",
        reg_param: float = 1e-8,
        early_stopping: Optional[Literal["relative_drop", "fixed_fraction"]] = None,
        early_stopping_threshold: float = 0.1,
    ) -> None:
        if freeze_basis_at is not None and freeze_basis_at < 2:
            raise ValueError(f"freeze_basis_at должен быть >= 2, получено {freeze_basis_at}")

        if strategy not in ("default", "master"):
            raise ValueError(f"strategy должен быть 'default' или 'master', получено {strategy}")

        if early_stopping is not None and early_stopping not in ("relative_drop", "fixed_fraction"):
            raise ValueError(
                f"early_stopping должен быть None, 'relative_drop' или "
                f"'fixed_fraction', получено {early_stopping!r}."
            )

        if early_stopping is not None and not (0.0 < early_stopping_threshold <= 1.0):
            raise ValueError(
                f"early_stopping_threshold должен быть в (0, 1], получено "
                f"{early_stopping_threshold}."
            )

        self.freeze_basis_at = freeze_basis_at
        self.strategy = strategy
        self.reg_param = reg_param
        self.early_stopping = early_stopping
        self.early_stopping_threshold = early_stopping_threshold

        self.labels_: Optional[np.ndarray] = None
        self.subspace_bases_: Optional[List[np.ndarray]] = None
        self.n_subclasses_: Optional[int] = None
        self.growth_history_: Optional[List[dict]] = None
        self.excluded_by_early_stopping_: Optional[np.ndarray] = None
        self.stopped_early_: bool = False
        self.is_fitted_: bool = False

    def fit(
        self,
        X: np.ndarray,
        pairs: np.ndarray,
        store_history: bool = False,
    ) -> "ConjugacyClusterGrowth":
        """Распределяет векторы по подклассам через последовательный рост.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N).
        pairs : np.ndarray
            Начальные пары индексов размерности (S, 2) из CosineSecondVectorAttacher.
            pairs[s] = [center_index, second_vector_index] для подкласса s.
        store_history : bool, default=False
            Сохранять ли историю роста (значение R, размер базиса и число
            обусловленности матрицы Грама на каждой итерации) в
            growth_history_. По умолчанию выключено — вычисление числа
            обусловленности на каждом шаге даёт заметный оверхед только при
            явном запросе (тот же принцип, что у ReferenceCenterBuilder.
            fit(..., store_history=...)).

        Returns
        -------
        self : ConjugacyClusterGrowth
            Возвращает экземпляр самого себя.

        Raises
        ------
        ValueError
            Если недостаточно векторов или некорректные пары.
        """
        X_arr = self._validate_input(X)
        pairs_arr = self._validate_pairs(pairs, X_arr.shape[0])

        n_samples, n_features = X_arr.shape
        n_subclasses = len(pairs_arr)

        # Инициализация меток и базисов
        labels = np.full(n_samples, -1, dtype=int)
        Y_bases = []

        # Заполняем начальные базисы из пар и помечаем векторы
        for s, (idx1, idx2) in enumerate(pairs_arr):
            Y_bases.append(X_arr[[idx1, idx2]].T)  # (N, 2)
            labels[idx1] = s
            labels[idx2] = s

        # Оставшиеся векторы для распределения
        remaining = np.where(labels == -1)[0].tolist()
        total_to_assign = len(remaining)
        logger.info(
            "ConjugacyClusterGrowth.fit: B.2 старт (strategy=%s), %d подклассов, "
            "%d векторов уже в парах, %d осталось распределить.",
            self.strategy, n_subclasses, n_samples - total_to_assign, total_to_assign,
        )

        # Основной цикл: присоединяем по одному вектору
        iteration = 0
        # Логируем прогресс на INFO примерно раз в 10% итераций, чтобы не
        # заваливать вывод на больших датасетах — детальный разбор каждой
        # итерации доступен на DEBUG.
        progress_step = max(1, total_to_assign // 10)
        growth_history: Optional[List[dict]] = [] if store_history else None
        first_best_R: Optional[float] = None
        stopped_early = False

        while remaining:
            # 1. Вычисляем R_matrix (len(remaining), n_subclasses)
            R_matrix = self._compute_conjugacy_matrix(
                X_arr, remaining, Y_bases
            )

            # 2. Находим глобальный максимум
            if self.strategy == "default":
                r_idx, s_idx = self._find_argmax_default(R_matrix)
            else:  # master
                r_idx, s_idx = self._find_argmax_master(R_matrix)

            best_R = float(R_matrix[r_idx, s_idx])

            # 3. Присоединяем вектор к подклассу
            vector_idx = remaining[r_idx]
            labels[vector_idx] = s_idx
            logger.debug(
                "ConjugacyClusterGrowth.fit: итерация %d/%d — вектор %d -> "
                "подкласс %d (R=%.6f, базис Y теперь k=%d, remaining=%d).",
                iteration + 1, total_to_assign, vector_idx, s_idx,
                best_R, Y_bases[s_idx].shape[1] + 1, len(remaining) - 1,
            )

            # 4. Обновляем базис подкласса
            Y_bases[s_idx] = self._append_to_basis(
                Y_bases[s_idx], X_arr[vector_idx]
            )

            if store_history:
                gram = Y_bases[s_idx].T @ Y_bases[s_idx]
                try:
                    cond = float(np.linalg.cond(gram))
                except np.linalg.LinAlgError:
                    cond = float("inf")
                growth_history.append({
                    "iteration": iteration + 1,
                    "vector_index": int(vector_idx),
                    "subclass_index": int(s_idx),
                    "r_value": best_R,
                    "basis_size": int(Y_bases[s_idx].shape[1]),
                    "gram_condition_number": cond,
                    "n_remaining_after": len(remaining) - 1,
                })

            # 5. Удаляем вектор из remaining
            remaining.pop(r_idx)
            iteration += 1

            if iteration % progress_step == 0 or not remaining:
                logger.info(
                    "ConjugacyClusterGrowth.fit: прогресс %d/%d векторов распределено.",
                    iteration, total_to_assign,
                )

            # 6. Критерий ранней остановки (theory/article_plans/01_
            #    iterativnyi_algoritm.txt) — проверяем ПОСЛЕ присоединения
            #    текущего вектора, и только если ещё есть что распределять:
            #    иначе на последней итерации естественное завершение (когда
            #    remaining уже пуст) ошибочно засчиталось бы как "ранняя"
            #    остановка. Первая итерация только фиксирует точку отсчёта
            #    R_первой_итерации для "relative_drop" — останов возможен не
            #    раньше второй.
            if not remaining:
                pass
            elif self.early_stopping == "relative_drop":
                if first_best_R is None:
                    first_best_R = best_R
                elif first_best_R > 0 and best_R < self.early_stopping_threshold * first_best_R:
                    logger.info(
                        "ConjugacyClusterGrowth.fit: ранняя остановка "
                        "(relative_drop) на итерации %d/%d — R=%.6f < "
                        "%.2f * R_first=%.6f, %d векторов останутся "
                        "неразмеченными.",
                        iteration, total_to_assign, best_R,
                        self.early_stopping_threshold, first_best_R,
                        len(remaining),
                    )
                    stopped_early = True
                    break
            elif self.early_stopping == "fixed_fraction":
                if iteration >= self.early_stopping_threshold * total_to_assign:
                    logger.info(
                        "ConjugacyClusterGrowth.fit: ранняя остановка "
                        "(fixed_fraction) на итерации %d/%d (порог=%.0f%%), "
                        "%d векторов останутся неразмеченными.",
                        iteration, total_to_assign,
                        self.early_stopping_threshold * 100, len(remaining),
                    )
                    stopped_early = True
                    break

        # Финализация: freeze базисов если требуется
        if self.freeze_basis_at is not None:
            Y_bases = [Y[:, :self.freeze_basis_at] for Y in Y_bases]
            logger.debug(
                "ConjugacyClusterGrowth.fit: базисы заморожены на k=%d.",
                self.freeze_basis_at,
            )

        self.labels_ = labels
        self.subspace_bases_ = Y_bases
        self.n_subclasses_ = n_subclasses
        self.growth_history_ = growth_history
        self.excluded_by_early_stopping_ = np.array(remaining, dtype=int)
        self.stopped_early_ = stopped_early
        self.is_fitted_ = True
        logger.info(
            "ConjugacyClusterGrowth.fit: B.2 готово, размеры подклассов=%s%s.",
            np.bincount(labels[labels >= 0], minlength=n_subclasses).tolist(),
            f", {len(remaining)} векторов не распределено (ранняя остановка)"
            if stopped_early else "",
        )

        return self

    def _compute_conjugacy_matrix(
        self,
        X: np.ndarray,
        remaining: List[int],
        Y_bases: List[np.ndarray],
    ) -> np.ndarray:
        """Вычисляет матрицу R для remaining × subclasses."""
        n_remaining = len(remaining)
        n_subclasses = len(Y_bases)
        R_matrix = np.zeros((n_remaining, n_subclasses))

        for r_idx, vector_idx in enumerate(remaining):
            x = X[vector_idx]
            for s_idx, Y in enumerate(Y_bases):
                R_matrix[r_idx, s_idx] = conjugate_criterion(
                    x.reshape(1, -1), Y, self.reg_param
                )[0]

        return R_matrix

    def _find_argmax_default(self, R_matrix: np.ndarray) -> Tuple[int, int]:
        """Находит глобальный argmax в R_matrix (default strategy)."""
        flat_idx = np.argmax(R_matrix)
        r_idx, s_idx = np.unravel_index(flat_idx, R_matrix.shape)
        return int(r_idx), int(s_idx)

    def _find_argmax_master(self, R_matrix: np.ndarray) -> Tuple[int, int]:
        """Находит ОДНО глобальное (x*, s*) через ratio к среднему.

        Аппроксимация идеи NB7 в рамках канона B.2 (один вектор за
        итерацию) — не литеральная реплика NB7, который назначает по
        одному вектору КАЖДОМУ подклассу одновременно за один "idea"-шаг.
        См. предупреждение в docstring модуля.
        """
        n_remaining, n_subclasses = R_matrix.shape

        best_ratio = -np.inf
        best_r_idx = 0
        best_s_idx = 0

        for r_idx in range(n_remaining):
            R_row = R_matrix[r_idx]

            # Для каждого подкласса: ratio = R_s / mean(R_others)
            for s_idx in range(n_subclasses):
                R_s = R_row[s_idx]
                R_others = np.concatenate([R_row[:s_idx], R_row[s_idx+1:]])
                mean_others = R_others.mean() if len(R_others) > 0 else 1.0

                if mean_others <= 1e-10:
                    logger.debug(
                        "ConjugacyClusterGrowth._find_argmax_master: "
                        "mean_others~0 для кандидата %d/подкласса %d — "
                        "используем R_s напрямую вместо деления.", r_idx, s_idx,
                    )
                    ratio = R_s
                else:
                    ratio = R_s / mean_others

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_r_idx = r_idx
                    best_s_idx = s_idx

        return best_r_idx, best_s_idx

    def _append_to_basis(self, Y: np.ndarray, new_vector: np.ndarray) -> np.ndarray:
        """Добавляет новый вектор как столбец к базису."""
        return np.column_stack([Y, new_vector])

    def get_subclass_sizes(self) -> np.ndarray:
        """Возвращает количество векторов в каждом подклассе.

        Векторы, оставшиеся неразмеченными из-за ранней остановки
        (метка -1, см. excluded_by_early_stopping_), в подсчёт не входят.

        Returns
        -------
        sizes : np.ndarray
            Массив размера (n_subclasses,) с количеством векторов.

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        return np.bincount(self.labels_[self.labels_ >= 0], minlength=self.n_subclasses_)

    def get_growth_curve(self) -> np.ndarray:
        """Возвращает последовательность R-значений в порядке присоединения.

        Прямой источник кривой R(k) для анализа сходимости (theory/
        article_plans/01_iterativnyi_algoritm.txt, задача 2) — R_matrix[k]
        здесь соответствует значению R, с которым к своему подклассу был
        присоединён k-й по счёту вектор (глобальный порядок итераций, не
        порядок внутри одного подкласса). Для разбивки по подклассам или
        доступа к числу обусловленности матрицы Грама на каждом шаге
        используйте growth_history_ напрямую.

        Returns
        -------
        r_values : np.ndarray
            Массив R-значений длины len(growth_history_).

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван, либо был вызван без
            store_history=True.
        """
        self._check_is_fitted()
        if self.growth_history_ is None:
            raise RuntimeError(
                "История роста не сохранена — вызовите fit(X, pairs, "
                "store_history=True) перед get_growth_curve()."
            )
        return np.array([record["r_value"] for record in self.growth_history_])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает подкласс для новых векторов.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N).

        Returns
        -------
        labels : np.ndarray
            Предсказанные метки подклассов (M,).

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        X_arr = self._validate_input(X)

        n_samples = X_arr.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        logger.info(
            "ConjugacyClusterGrowth.predict: %d векторов, %d подклассов.",
            n_samples, self.n_subclasses_,
        )

        for i in range(n_samples):
            x = X_arr[i]
            R_values = np.zeros(self.n_subclasses_)

            for s, Y in enumerate(self.subspace_bases_):
                R_values[s] = conjugate_criterion(
                    x.reshape(1, -1), Y, self.reg_param
                )[0]

            labels[i] = np.argmax(R_values)
            logger.debug(
                "ConjugacyClusterGrowth.predict: вектор %d -> подкласс %d (R=%.6f).",
                i, labels[i], float(R_values[labels[i]]),
            )

        logger.info(
            "ConjugacyClusterGrowth.predict: готово, распределение по подклассам=%s.",
            np.bincount(labels, minlength=self.n_subclasses_).tolist(),
        )
        return labels

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Валидирует входную матрицу X."""
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim != 2:
            raise ValueError(
                f"Ожидалась 2D матрица векторов, получена {X_arr.ndim}D."
            )

        if X_arr.size == 0:
            raise ValueError("Передана пустая матрица X.")

        return X_arr

    def _validate_pairs(self, pairs: np.ndarray, n_samples: int) -> np.ndarray:
        """Валидирует массив пар."""
        pairs_arr = np.asarray(pairs, dtype=int)

        if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
            raise ValueError(
                f"pairs должен иметь форму (n_subclasses, 2), получено {pairs_arr.shape}."
            )

        if len(pairs_arr) == 0:
            raise ValueError("pairs не может быть пустым.")

        # Проверка диапазона индексов
        all_indices = pairs_arr.flatten()
        if not np.all((all_indices >= 0) & (all_indices < n_samples)):
            raise ValueError(
                f"Некоторые индексы в pairs вне диапазона [0, {n_samples})."
            )

        # Проверка уникальности
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("pairs содержит дублирующиеся индексы.")

        return pairs_arr

    def _check_is_fitted(self) -> None:
        """Проверяет, был ли вызван fit()."""
        if not self.is_fitted_:
            logger.error("ConjugacyClusterGrowth: обращение к результатам до fit().")
            raise RuntimeError(
                "Модель не обучена. Вызовите fit(X, pairs) перед использованием."
            )

    def __repr__(self) -> str:
        es = (
            f", early_stopping='{self.early_stopping}'"
            f"(threshold={self.early_stopping_threshold})"
            if self.early_stopping is not None else ""
        )
        if self.is_fitted_:
            stopped = f", stopped_early={self.stopped_early_}" if self.early_stopping else ""
            return (
                f"ConjugacyClusterGrowth(n_subclasses={self.n_subclasses_}, "
                f"freeze_basis_at={self.freeze_basis_at}, strategy='{self.strategy}'"
                f"{es}{stopped}, fitted=True)"
            )
        return (
            f"ConjugacyClusterGrowth(freeze_basis_at={self.freeze_basis_at}, "
            f"strategy='{self.strategy}'{es}, fitted=False)"
        )
