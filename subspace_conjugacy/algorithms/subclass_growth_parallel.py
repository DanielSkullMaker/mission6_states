"""Параллельная/оптимизированная альтернатива Algorithm B.2 (ConjugacyClusterGrowth).

Контекст: при анализе `algorithms/subclass_growth.py::ConjugacyClusterGrowth`
(канон, см. обсуждение в этом чате) обнаружена алгоритмическая неэффективность
— `_compute_conjugacy_matrix` пересчитывает обратную матрицу Грама
`(Y^T Y)^{-1}` ОТДЕЛЬНО НА КАЖДОГО кандидата в Python-цикле, хотя базис
`Y_s` внутри одной итерации роста не меняется (один и тот же результат
пересчитывается `len(remaining)` раз вместо одного).

По явной просьбе — "не затрагивая уже написанный код, далее он будет
необходим для сравнения" — канон (`ConjugacyClusterGrowth`) НЕ изменён.
Этот модуль — САМОСТОЯТЕЛЬНАЯ альтернатива с идентичным публичным
интерфейсом (fit/labels_/subspace_bases_/predict/get_subclass_sizes),
взаимозаменяемая с каноном для последующего бенчмарка (сравнение
скорости/корректности "до/после" — материал для Статьи 5 диссертационного
плана, применимость метода в реальном времени).

Два НЕЗАВИСИМЫХ источника ускорения, оба уместны для бенчмарка отдельно:

  1. ВЕКТОРИЗАЦИЯ (всегда включена, не параметр): `(Y^T Y)^{-1}` считается
     ОДИН раз на подкласс за итерацию (не на кандидата), после чего
     R(x, Y_s) для ВСЕХ remaining-кандидатов вычисляется одним матричным
     умножением — устраняет саму причину квадратичного роста времени по
     числу кандидатов на итерацию.
  2. ПАРАЛЛЕЛИЗАЦИЯ (n_jobs, см. ниже): после векторизации по кандидатам
     остаётся цикл по n_subclasses подпространствам (один R-столбец на
     подкласс) — выполняется через `joblib.Parallel`. backend="threading",
     т.к. тяжёлая часть каждой задачи — BLAS-матричные умножения numpy,
     которые отпускают GIL; процессный backend добавил бы накладные расходы
     на копирование X между процессами без выигрыша для такой гранулярности
     задачи.

n_jobs=1 (по умолчанию) включает ТОЛЬКО векторизацию (без накладных расходов
joblib) — честная база для сравнения "эффект векторизации отдельно от
эффекта параллелизма отдельно".
"""

import logging
import time
from typing import List, Optional, Tuple
import numpy as np
from joblib import Parallel, delayed

from subspace_conjugacy.core.metrics import compute_gram_inverse

logger = logging.getLogger(__name__)


def _r_column(X_remaining: np.ndarray, Y: np.ndarray, reg_param: float) -> np.ndarray:
    """R(x, Y) для всех строк X_remaining сразу — (Y^T Y)^{-1} считается один раз."""
    inv_gram = compute_gram_inverse(Y, reg_param=reg_param)
    x_norms_sq = np.maximum(np.sum(X_remaining ** 2, axis=1), reg_param)
    projections = X_remaining @ Y
    numerator = np.sum((projections @ inv_gram) * projections, axis=1)
    return np.clip(numerator / x_norms_sq, 0.0, 1.0)


class ParallelConjugacyClusterGrowth:
    """Векторизованная/параллельная альтернатива ConjugacyClusterGrowth (B.2).

    Реализует только strategy="default" (argmax R(x, Y_s)) канона —
    "master"-стратегия (ratio к среднему, approximation NB7) здесь не
    воспроизведена, т.к. не является частью узкого места производительности
    и не нужна для сравнения скорости/корректности с каноном на одинаковых
    условиях.

    Parameters
    ----------
    freeze_basis_at : int or None, default=2
        См. ConjugacyClusterGrowth.
    reg_param : float, default=1e-8
        См. ConjugacyClusterGrowth.
    n_jobs : int, default=1
        Число потоков joblib для параллельного вычисления R-столбцов по
        подклассам на каждой итерации. 1 — последовательно (но всё равно
        векторизовано по кандидатам, без накладных расходов joblib); -1 —
        все ядра.

    Attributes
    ----------
    labels_ : np.ndarray or None
    subspace_bases_ : list[np.ndarray] or None
    n_subclasses_ : int or None
    is_fitted_ : bool
    fit_time_s_ : float or None
        Время выполнения fit() в секундах — для прямого сравнения с
        ConjugacyClusterGrowth на одних и тех же данных.
    """

    def __init__(
        self,
        freeze_basis_at: Optional[int] = 2,
        reg_param: float = 1e-8,
        n_jobs: int = 1,
    ) -> None:
        if freeze_basis_at is not None and freeze_basis_at < 2:
            raise ValueError(f"freeze_basis_at должен быть >= 2, получено {freeze_basis_at}")

        self.freeze_basis_at = freeze_basis_at
        self.reg_param = reg_param
        self.n_jobs = n_jobs

        self.labels_: Optional[np.ndarray] = None
        self.subspace_bases_: Optional[List[np.ndarray]] = None
        self.n_subclasses_: Optional[int] = None
        self.is_fitted_: bool = False
        self.fit_time_s_: Optional[float] = None

    def fit(self, X: np.ndarray, pairs: np.ndarray) -> "ParallelConjugacyClusterGrowth":
        """Распределяет векторы по подклассам — идентичный контракт ConjugacyClusterGrowth.fit().

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов (M, N).
        pairs : np.ndarray
            Начальные пары индексов (S, 2), как в CosineSecondVectorAttacher.

        Returns
        -------
        self : ParallelConjugacyClusterGrowth
        """
        t0 = time.perf_counter()
        X_arr = self._validate_input(X)
        pairs_arr = self._validate_pairs(pairs, X_arr.shape[0])
        n_samples = X_arr.shape[0]
        n_subclasses = len(pairs_arr)

        labels = np.full(n_samples, -1, dtype=int)
        Y_bases: List[np.ndarray] = []
        for s, (idx1, idx2) in enumerate(pairs_arr):
            Y_bases.append(X_arr[[idx1, idx2]].T)
            labels[idx1] = s
            labels[idx2] = s

        remaining = np.where(labels == -1)[0].tolist()
        logger.info(
            "ParallelConjugacyClusterGrowth.fit: старт (n_jobs=%s), %d подклассов, "
            "%d векторов осталось распределить.", self.n_jobs, n_subclasses, len(remaining),
        )

        use_parallel = self.n_jobs != 1
        parallel = Parallel(n_jobs=self.n_jobs, prefer="threads") if use_parallel else None

        while remaining:
            remaining_arr = np.array(remaining)
            X_remaining = X_arr[remaining_arr]

            if use_parallel:
                columns = parallel(
                    delayed(_r_column)(X_remaining, Y, self.reg_param) for Y in Y_bases
                )
            else:
                columns = [_r_column(X_remaining, Y, self.reg_param) for Y in Y_bases]

            R_matrix = np.column_stack(columns)
            flat_idx = np.argmax(R_matrix)
            r_idx, s_idx = np.unravel_index(flat_idx, R_matrix.shape)

            vector_idx = remaining[r_idx]
            labels[vector_idx] = s_idx
            Y_bases[s_idx] = np.column_stack([Y_bases[s_idx], X_arr[vector_idx]])
            remaining.pop(r_idx)

        if self.freeze_basis_at is not None:
            Y_bases = [Y[:, : self.freeze_basis_at] for Y in Y_bases]

        self.labels_ = labels
        self.subspace_bases_ = Y_bases
        self.n_subclasses_ = n_subclasses
        self.is_fitted_ = True
        self.fit_time_s_ = time.perf_counter() - t0
        logger.info(
            "ParallelConjugacyClusterGrowth.fit: готово за %.3fs, размеры подклассов=%s.",
            self.fit_time_s_, np.bincount(labels, minlength=n_subclasses).tolist(),
        )
        return self

    def get_subclass_sizes(self) -> np.ndarray:
        self._check_is_fitted()
        return np.bincount(self.labels_, minlength=self.n_subclasses_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает подкласс для новых векторов через argmax R (векторизовано)."""
        self._check_is_fitted()
        X_arr = self._validate_input(X)
        columns = [_r_column(X_arr, Y, self.reg_param) for Y in self.subspace_bases_]
        R_matrix = np.column_stack(columns)
        return np.argmax(R_matrix, axis=1)

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Ожидалась 2D матрица векторов, получена {X_arr.ndim}D.")
        if X_arr.size == 0:
            raise ValueError("Передана пустая матрица X.")
        return X_arr

    def _validate_pairs(self, pairs: np.ndarray, n_samples: int) -> np.ndarray:
        pairs_arr = np.asarray(pairs, dtype=int)
        if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
            raise ValueError(f"pairs должен иметь форму (n_subclasses, 2), получено {pairs_arr.shape}.")
        if len(pairs_arr) == 0:
            raise ValueError("pairs не может быть пустым.")
        all_indices = pairs_arr.flatten()
        if not np.all((all_indices >= 0) & (all_indices < n_samples)):
            raise ValueError(f"Некоторые индексы в pairs вне диапазона [0, {n_samples}).")
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("pairs содержит дублирующиеся индексы.")
        return pairs_arr

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Модель не обучена. Вызовите fit(X, pairs) перед использованием.")

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"ParallelConjugacyClusterGrowth(n_subclasses={self.n_subclasses_}, "
                f"freeze_basis_at={self.freeze_basis_at}, n_jobs={self.n_jobs}, fitted=True)"
            )
        return (
            f"ParallelConjugacyClusterGrowth(freeze_basis_at={self.freeze_basis_at}, "
            f"n_jobs={self.n_jobs}, fitted=False)"
        )
