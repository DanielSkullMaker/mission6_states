"""Resize изображений до фиксированного размера (NB1: 1_image_resizer.ipynb).

NB1 использовал PIL с ресемплингом LANCZOS и попиксельными циклами только
для подсчёта файлов — само изменение размера уже было единственной операцией
на изображение (не циклом по пикселям), так что векторизировать здесь нечего;
модуль просто даёт переиспользуемую, тестируемую замену вместо кода в ячейках
ноутбука и добавляет пакетную обработку директорий.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SIZE: Tuple[int, int] = (256, 256)


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int] = DEFAULT_SIZE,
) -> np.ndarray:
    """Изменяет размер изображения (аналог PIL Image.LANCZOS из NB1).

    Parameters
    ----------
    image : np.ndarray
        Входное изображение (H, W) или (H, W, C).
    size : tuple[int, int], default=(256, 256)
        Целевой размер (width, height).

    Returns
    -------
    resized : np.ndarray
        Изображение размера (size[1], size[0][, C]).

    Notes
    -----
    ``cv2.INTER_LANCZOS4`` — ближайший аналог ``PIL.Image.LANCZOS``,
    используемого в NB1.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)


def resize_image_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    size: Tuple[int, int] = DEFAULT_SIZE,
) -> Path:
    """Загружает изображение с диска, изменяет размер и сохраняет как PNG.

    Parameters
    ----------
    input_path : str or Path
        Путь к исходному изображению (любой формат, поддерживаемый OpenCV).
    output_path : str or Path
        Путь для сохранения результата.
    size : tuple[int, int], default=(256, 256)
        Целевой размер (width, height).

    Returns
    -------
    output_path : Path
        Путь к сохранённому файлу.

    Raises
    ------
    FileNotFoundError
        Если исходный файл не существует или не читается OpenCV.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        logger.error("resize_image_file: не удалось загрузить '%s'.", input_path)
        raise FileNotFoundError(f"Не удалось загрузить изображение: {input_path}")

    resized = resize_image(image, size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), resized)
    logger.debug("resize_image_file: %s -> %s (size=%s).", input_path, output_path, size)
    return output_path


def resize_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    size: Tuple[int, int] = DEFAULT_SIZE,
    output_prefix: str = "",
    pattern: str = "*",
) -> List[Path]:
    """Пакетно изменяет размер всех изображений в директории (цикл из NB1).

    Файлы на выходе нумеруются последовательно как
    ``{output_prefix}{i+1}.png`` (i начиная с 0) — формат, совместимый с
    ``DatasetConfig.get_image_paths`` для следующих стадий пайплайна.
    Порядок обработки — по сортированному имени исходного файла.

    Parameters
    ----------
    input_dir : str or Path
        Директория с исходными изображениями.
    output_dir : str or Path
        Директория для сохранения результатов (создаётся при необходимости).
    size : tuple[int, int], default=(256, 256)
        Целевой размер (width, height).
    output_prefix : str, default=""
        Префикс имени выходного файла (например, "glioma" -> "glioma1.png").
    pattern : str, default="*"
        Glob-паттерн для отбора исходных файлов (например, "*.jpg").

    Returns
    -------
    output_paths : List[Path]
        Пути к сохранённым файлам, в порядке обработки.

    Raises
    ------
    ValueError
        Если во входной директории не найдено ни одного файла по паттерну.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not input_paths:
        logger.error(
            "resize_directory: нет файлов по паттерну '%s' в '%s'.", pattern, input_dir,
        )
        raise ValueError(
            f"В директории '{input_dir}' не найдено файлов по паттерну '{pattern}'."
        )

    logger.info(
        "resize_directory: %d файлов из %s -> %s (size=%s).",
        len(input_paths), input_dir, output_dir, size,
    )
    output_paths = []
    for i, input_path in enumerate(input_paths, start=1):
        output_path = output_dir / f"{output_prefix}{i}.png"
        output_paths.append(resize_image_file(input_path, output_path, size))

    logger.info("resize_directory: готово, %d файлов записано в %s.", len(output_paths), output_dir)
    return output_paths
