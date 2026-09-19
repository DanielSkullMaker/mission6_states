"""Центрирование содержимого изображения (NB2: find/image_correction_*).

Ноутбук 2_dataset_preparing.ipynb определяет "центрирование" так: по маске
фона (см. normalization.py) находится, сколько почти-пустых строк/столбцов
есть у верхнего/нижнего и левого/правого края, и изображение сдвигается
(circular shift) так, чтобы контент оказался ближе к центру кадра.

Оригинальная реализация ноутбука — четыре функции с ручными Python-циклами
(find_horizontal_delta, image_correction_top_and_bottom, find_vertical_delta,
image_correction_left_and_right), выполняющие сдвиг через pop()/insert() по
одному пикселю за раз. Здесь — та же идея через numpy (сравнение сумм строк/
столбцов с порогом + np.roll), без циклов по пикселям.

ДВА расхождения с буквальным кодом ноутбука (обе — исправленные баги, не
воспроизводятся сознательно, как и vector_norm(first_reference) в NB4,
задокументированный в refactoring_plan.txt раздел 2.8):

1. Скан "снизу"/"справа" в ноутбуке использует ``image[-line]`` начиная с
   ``line=0``, а ``-0 == 0`` в Python — то есть первая "нижняя" (или
   "правая") строка/столбец на самом деле берётся с противоположного края
   (индекс 0), а не с истинного конца массива. Здесь скан снизу/справа
   делается через явный разворот массива (``[::-1]``), что даёт корректный
   отсчёт от истинного последнего элемента.

2. ``image_correction_left_and_right`` использует условие
   ``elif vertical_delta < 1`` (должно быть симметрично ``< -1``, как в
   горизонтальной версии ``image_correction_top_and_bottom``) — из-за этого
   сдвиг по вертикали асимметрично применяется/не применяется в районе
   delta ∈ {-1, 0, 1} в зависимости от знака. Здесь обе оси используют один
   параметр ``min_shift`` (по умолчанию 2, что эквивалентно исходному
   "|delta| > 1") симметрично.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MIN_SHIFT = 2


def _leading_empty_count(row_or_col_sums: np.ndarray, threshold: float) -> int:
    """Считает число подряд идущих элементов <= threshold с начала массива.

    Аналог ручного цикла ``for line in range(...): ... else: break`` из NB2,
    но через ``np.argmax`` по булевой маске (без явного цикла на Python).
    """
    above_threshold = row_or_col_sums > threshold
    if not above_threshold.any():
        return len(row_or_col_sums)
    return int(np.argmax(above_threshold))


def compute_horizontal_delta(gray: np.ndarray, threshold_multiplier: int = 2) -> int:
    """Вычисляет сдвиг по вертикали (число "пустых" строк сверху минус снизу).

    Соответствует ``find_horizontal_delta`` из NB2, но без бага ``-0`` при
    сканировании снизу (см. docstring модуля).

    Parameters
    ----------
    gray : np.ndarray
        Grayscale-изображение (H, W) с уже подавленным фоном
        (см. normalization.suppress_background).
    threshold_multiplier : int, default=2
        Строка/столбец считается "пустым", если сумма его яркости
        <= threshold_multiplier * ширина (как в NB2: ``image_width * 2``).

    Returns
    -------
    delta : int
        ``int((top_empty - bottom_empty) / 2)`` — положительное значение
        означает, что контент смещён вниз (нужно сдвинуть вверх), и наоборот.
    """
    width = gray.shape[1]
    row_sums = gray.sum(axis=1, dtype=np.int64)
    threshold = threshold_multiplier * width

    top_empty = _leading_empty_count(row_sums, threshold)
    bottom_empty = _leading_empty_count(row_sums[::-1], threshold)

    return int((top_empty - bottom_empty) / 2)


def compute_vertical_delta(gray: np.ndarray, threshold_multiplier: int = 2) -> int:
    """Вычисляет сдвиг по горизонтали (число "пустых" столбцов слева минус справа).

    Соответствует ``find_vertical_delta`` из NB2, но без бага ``-0`` при
    сканировании справа (см. docstring модуля).

    Parameters
    ----------
    gray : np.ndarray
        Grayscale-изображение (H, W) с уже подавленным фоном.
    threshold_multiplier : int, default=2
        Столбец считается "пустым", если сумма его яркости
        <= threshold_multiplier * высота (как в NB2: ``image_length * 2``).

    Returns
    -------
    delta : int
        ``int((left_empty - right_empty) / 2)``.
    """
    length = gray.shape[0]
    col_sums = gray.sum(axis=0, dtype=np.int64)
    threshold = threshold_multiplier * length

    left_empty = _leading_empty_count(col_sums, threshold)
    right_empty = _leading_empty_count(col_sums[::-1], threshold)

    return int((left_empty - right_empty) / 2)


def shift_rows(
    image: np.ndarray, delta: int, min_shift: int = DEFAULT_MIN_SHIFT
) -> np.ndarray:
    """Циклически сдвигает изображение по строкам (аналог image_correction_top_and_bottom).

    Parameters
    ----------
    image : np.ndarray
        Изображение (H, W) или (H, W, C).
    delta : int
        Величина и направление сдвига (см. compute_horizontal_delta).
    min_shift : int, default=2
        Сдвиг применяется только если ``abs(delta) >= min_shift`` — малые
        значения (шум в один пиксель) игнорируются, как ``> 1``/``< -1`` в
        оригинале NB2.

    Returns
    -------
    shifted : np.ndarray
        Изображение того же размера, сдвинутое по оси строк (axis=0).
    """
    if abs(delta) < min_shift:
        return image
    return np.roll(image, -delta, axis=0)


def shift_columns(
    image: np.ndarray, delta: int, min_shift: int = DEFAULT_MIN_SHIFT
) -> np.ndarray:
    """Циклически сдвигает изображение по столбцам (аналог image_correction_left_and_right).

    Parameters
    ----------
    image : np.ndarray
        Изображение (H, W) или (H, W, C).
    delta : int
        Величина и направление сдвига (см. compute_vertical_delta).
    min_shift : int, default=2
        Сдвиг применяется только если ``abs(delta) >= min_shift``.

    Returns
    -------
    shifted : np.ndarray
        Изображение того же размера, сдвинутое по оси столбцов (axis=1).
    """
    if abs(delta) < min_shift:
        return image
    return np.roll(image, -delta, axis=1)


def center_image(
    image: np.ndarray,
    background_threshold: int = 10,
    min_shift: int = DEFAULT_MIN_SHIFT,
) -> np.ndarray:
    """Полный аналог NB2 (cell 12): подавление фона + центрирование по обеим осям.

    Порядок вычислений повторяет ноутбук: оба delta (горизонтальный и
    вертикальный) считаются от ОДНОЙ и той же исходной (ещё не сдвинутой)
    grayscale-маски, затем оба сдвига применяются последовательно к цветному
    изображению — вертикальный сдвиг не пересчитывается после горизонтального.

    Parameters
    ----------
    image : np.ndarray
        Цветное (H, W, 3) или grayscale (H, W) изображение.
    background_threshold : int, default=10
        Порог подавления фона (см. normalization.suppress_background).
    min_shift : int, default=2
        Минимальная величина |delta| для применения сдвига.

    Returns
    -------
    centered : np.ndarray
        Изображение того же размера и типа, что и ``image``.
    """
    from subspace_conjugacy.preprocessing.normalization import suppress_background

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    masked_gray = suppress_background(gray, background_threshold)
    masked_target = suppress_background(image, background_threshold, reference=gray)

    h_delta = compute_horizontal_delta(masked_gray)
    v_delta = compute_vertical_delta(masked_gray)

    result = shift_rows(masked_target, h_delta, min_shift=min_shift)
    result = shift_columns(result, v_delta, min_shift=min_shift)
    logger.debug(
        "center_image: h_delta=%d, v_delta=%d (min_shift=%d).", h_delta, v_delta, min_shift,
    )
    return result


def center_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    background_threshold: int = 10,
    min_shift: int = DEFAULT_MIN_SHIFT,
    pattern: str = "*.png",
) -> List[Path]:
    """Пакетно центрирует все изображения директории (NB2, отдельно от resize).

    Симметричный аналог resize.resize_directory() для стадии центрирования —
    используется как самостоятельная стадия pipeline ("center" в
    pipeline/stages.py), когда resize уже выполнен отдельно (стадия "resize")
    и на входе лежат уже готовые 256x256 файлы.

    Имена выходных файлов совпадают с именами входных (в отличие от
    resize_directory, здесь не требуется перенумерация — предполагается,
    что resize уже пронумеровал файлы как {prefix}{i}.png).

    Parameters
    ----------
    input_dir : str or Path
        Директория с изображениями после resize (обычно config.paths[cls]["resized"]).
    output_dir : str or Path
        Директория для сохранения центрированных изображений.
    background_threshold : int, default=10
        Порог подавления фона (см. normalization.suppress_background).
    min_shift : int, default=2
        Минимальная величина |delta| для применения сдвига.
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
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not input_paths:
        logger.error(
            "center_directory: нет файлов по паттерну '%s' в '%s'.", pattern, input_dir,
        )
        raise ValueError(
            f"В директории '{input_dir}' не найдено файлов по паттерну '{pattern}'."
        )

    logger.info(
        "center_directory: %d файлов из %s -> %s.", len(input_paths), input_dir, output_dir,
    )
    output_paths = []
    for input_path in input_paths:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.error("center_directory: не удалось загрузить '%s'.", input_path)
            raise FileNotFoundError(f"Не удалось загрузить изображение: {input_path}")

        centered = center_image(
            image, background_threshold=background_threshold, min_shift=min_shift
        )

        output_path = output_dir / input_path.name
        cv2.imwrite(str(output_path), centered)
        output_paths.append(output_path)

    logger.info("center_directory: готово, %d файлов записано в %s.", len(output_paths), output_dir)
    return output_paths
