"""Параллельная загрузка+векторизация батча изображений (I/O-bound).

`features/vectorization.py::load_and_vectorize_batch` (не изменён) читает
файлы последовательно, одним списком-компрехеншеном. Загрузка с диска —
I/O-bound (`cv2.imread`), а не CPU-bound — поэтому здесь используется
`concurrent.futures.ThreadPoolExecutor` (GIL отпускается на вызовах I/O и
на самом cv2.imread/decode, в отличие от Python-циклов с численными
вычислениями), а не joblib/multiprocessing — процессы добавили бы
накладные расходы на межпроцессную передачу результатов без выигрыша для
такой гранулярности задачи (один файл — одна задача).

Оборачивает ТОЛЬКО существующую `load_and_vectorize` (не изменена) —
результат идентичен последовательной версии, порядок выходных строк X
сохраняется (= порядку image_paths), несмотря на параллельное выполнение.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np

from subspace_conjugacy.features.vectorization import VectorizationType, load_and_vectorize

logger = logging.getLogger(__name__)


def load_and_vectorize_batch_parallel(
    image_paths: List[Union[str, Path]],
    method: VectorizationType = "horizontal",
    backend: Literal["cv2", "pil"] = "cv2",
    max_workers: Optional[int] = None,
) -> np.ndarray:
    """Параллельная версия load_and_vectorize_batch (ThreadPoolExecutor).

    Parameters
    ----------
    image_paths : List[str or Path]
        Пути к изображениям — порядок сохраняется в результирующей X
        независимо от порядка завершения потоков (``executor.map``).
    method : {"horizontal", "vertical"}, default="horizontal"
    backend : {"cv2", "pil"}, default="cv2"
    max_workers : int, optional
        Число потоков. None — ThreadPoolExecutor сам выбирает (обычно
        min(32, os.cpu_count() + 4)).

    Returns
    -------
    X : np.ndarray
        Матрица векторов (N, H*W) — идентична результату
        load_and_vectorize_batch на тех же входных данных.

    Examples
    --------
    >>> from subspace_conjugacy.features.vectorization_parallel import (
    ...     load_and_vectorize_batch_parallel,
    ... )
    >>> X = load_and_vectorize_batch_parallel(paths, max_workers=8)
    """
    t0 = time.perf_counter()
    logger.info(
        "load_and_vectorize_batch_parallel: старт, %d файлов (max_workers=%s).",
        len(image_paths), max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        vectors = list(
            executor.map(
                lambda path: load_and_vectorize(path, method=method, backend=backend),
                image_paths,
            )
        )

    X = np.vstack(vectors)
    logger.info(
        "load_and_vectorize_batch_parallel: готово за %.3fs, X.shape=%s.",
        time.perf_counter() - t0, X.shape,
    )
    return X
