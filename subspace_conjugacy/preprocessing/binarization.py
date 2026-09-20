"""Otsu-бинаризация изображений (статья, раздел "Data Preprocessing").

Korshikov & Fursov, "Pathology Recognition Based on Conjugacy Criteria with
Subspaces of Reference Images" (theory/VI_Korshikov_VA_Fursov_..._Conjugacy.docx):
  "To perform the task of automatic image projection detection, binarization
  is applied to abstract the specific features of the image and extract the
  essential elements, improving the detection accuracy... Since it is not
  possible to set the threshold of image separation in advance due to their
  differences in brightness, the Otsu method is used for binarization,
  which allows to divide the whole set of pixels into two classes: useful
  and background. For their distribution, a binarization threshold is
  calculated such that the inter-class variance is minimized [формула (2):
  σ_ω²(t) = ω1(t)σ1²(t) + ω2(t)σ2²(t)], which is also equivalent to
  maximizing the inter-class variance. In this paper, the useful pixels are
  white and the background pixels are black."

  "For the task of tumor type detection in projection distributed images,
  binarization is NOT performed because it requires knowing the specific
  brightness of the image segments for accurate detection."

Т.е. Otsu-бинаризация в статье применяется ТОЛЬКО на первом этапе пайплайна
(определение проекции axial/sagittal/coronal), а не на втором (определение
типа опухоли внутри уже известной проекции) — refactoring_plan.txt, раздел
10, находка №4. models/sequential_classifier.py::SequentialClassifier
реализует именно эту связку: этап 1 на бинаризованных векторах, этап 2 — на
исходных (не бинаризованных).

Реализация — тонкая обёртка над cv2.threshold(..., cv2.THRESH_OTSU): формула
(2) статьи — это ровно классический метод Отсу (минимизация внутриклассовой
дисперсии эквивалентна максимизации межклассовой), переизобретать поиск
порога вручную не нужно — аналогично resize.py, где cv2.INTER_LANCZOS4
используется вместо ручной реализации LANCZOS-ресемплинга из NB1.
"""

import logging
from pathlib import Path
from typing import List, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _ensure_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Приводит изображение к 2D grayscale uint8 (вход для cv2.threshold)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(
            f"Изображение должно быть 2D (H, W) или 3D (H, W, C), получено {image.ndim}D."
        )

    return gray if gray.dtype == np.uint8 else gray.astype(np.uint8)


def otsu_threshold(image: np.ndarray) -> float:
    """Вычисляет порог бинаризации методом Отсу (формула (2) статьи).

    Parameters
    ----------
    image : np.ndarray
        Изображение (H, W) grayscale или (H, W, C) цветное (будет
        преобразовано в grayscale).

    Returns
    -------
    threshold : float
        Порог яркости, минимизирующий внутриклассовую дисперсию (что
        эквивалентно максимизации межклассовой дисперсии между "полезными"
        и "фоновыми" пикселями).

    Examples
    --------
    >>> import numpy as np
    >>> gray = np.array([[10, 200], [5, 220]], dtype=np.uint8)
    >>> t = otsu_threshold(gray)
    >>> 10 < t < 200
    True
    """
    gray = _ensure_grayscale_uint8(image)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logger.debug("otsu_threshold: %.2f.", threshold)
    return float(threshold)


def otsu_binarize(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Бинаризует изображение методом Отсу (статья, раздел "Data Preprocessing").

    Parameters
    ----------
    image : np.ndarray
        Изображение (H, W) grayscale или (H, W, C) цветное (будет
        преобразовано в grayscale перед бинаризацией — результат всегда 2D).
    invert : bool, default=False
        Если True — инвертирует результат (полезные пиксели чёрные, фон
        белый). По умолчанию — как в статье: полезные пиксели белые (255),
        фон чёрный (0).

    Returns
    -------
    binarized : np.ndarray
        Бинаризованное изображение (H, W), dtype uint8, значения только
        {0, 255}.

    Examples
    --------
    >>> import numpy as np
    >>> gray = np.array([[10, 200], [5, 220]], dtype=np.uint8)
    >>> otsu_binarize(gray)
    array([[  0, 255],
           [  0, 255]], dtype=uint8)
    """
    gray = _ensure_grayscale_uint8(image)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    threshold, binarized = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    logger.debug("otsu_binarize: threshold=%.2f, invert=%s.", threshold, invert)
    return binarized


def binarize_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    invert: bool = False,
    pattern: str = "*.png",
) -> List[Path]:
    """Пакетно бинаризует все изображения директории методом Отсу.

    Симметричный аналог preprocessing.centering.center_directory для стадии
    бинаризации — самостоятельная стадия pipeline ("binarize" в
    pipeline/stages.py), применяемая ТОЛЬКО для подготовки данных этапа
    определения проекции (см. docstring модуля) — обычная стадия
    определения типа опухоли эту стадию не использует вовсе.

    Имена выходных файлов совпадают с именами входных.

    Parameters
    ----------
    input_dir : str or Path
        Директория с исходными (обычно уже resize+center) изображениями.
    output_dir : str or Path
        Директория для сохранения бинаризованных изображений.
    invert : bool, default=False
        См. otsu_binarize.
    pattern : str, default="*.png"
        Glob-паттерн для отбора исходных файлов.

    Returns
    -------
    output_paths : List[Path]
        Пути к сохранённым файлам, в порядке обработки (по сортированному
        имени исходного файла).

    Raises
    ------
    ValueError
        Если во входной директории не найдено файлов по паттерну.
    FileNotFoundError
        Если какой-либо файл не удалось загрузить.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not input_paths:
        logger.error(
            "binarize_directory: нет файлов по паттерну '%s' в '%s'.", pattern, input_dir,
        )
        raise ValueError(
            f"В директории '{input_dir}' не найдено файлов по паттерну '{pattern}'."
        )

    logger.info(
        "binarize_directory: %d файлов из %s -> %s.", len(input_paths), input_dir, output_dir,
    )
    output_paths = []
    for input_path in input_paths:
        image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error("binarize_directory: не удалось загрузить '%s'.", input_path)
            raise FileNotFoundError(f"Не удалось загрузить изображение: {input_path}")

        binarized = otsu_binarize(image, invert=invert)

        output_path = output_dir / input_path.name
        cv2.imwrite(str(output_path), binarized)
        output_paths.append(output_path)

    logger.info(
        "binarize_directory: готово, %d файлов записано в %s.", len(output_paths), output_dir,
    )
    return output_paths
