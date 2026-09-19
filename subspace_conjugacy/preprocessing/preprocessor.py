"""ImagePreprocessor — фасад над resize.py + normalization.py + centering.py.

Объединяет NB1 (resize) и NB2 (нормализация фона + центрирование) в один
объект с in-memory API (массивы) и file/directory API (совместимый с
DatasetConfig), аналогично тому, как FursovClusterer — фасад над
algorithms/global_pair.py + reference_centers.py + subclass_seed.py +
subclass_growth.py.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np

from subspace_conjugacy.preprocessing.centering import DEFAULT_MIN_SHIFT, center_image
from subspace_conjugacy.preprocessing.normalization import DEFAULT_THRESHOLD
from subspace_conjugacy.preprocessing.resize import DEFAULT_SIZE, resize_image

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Препроцессинг МРТ-изображений: resize -> подавление фона -> центрирование.

    Parameters
    ----------
    target_size : tuple[int, int], default=(256, 256)
        Размер после resize (width, height) — NB1.
    background_threshold : int, default=10
        Порог подавления фона — NB2.
    min_shift : int, default=2
        Минимальная величина сдвига для центрирования — NB2
        (см. preprocessing.centering).

    Examples
    --------
    >>> preprocessor = ImagePreprocessor()
    >>> image = cv2.imread("raw/glioma1.jpg", cv2.IMREAD_COLOR)
    >>> processed = preprocessor.process(image)
    >>> processed.shape
    (256, 256, 3)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = DEFAULT_SIZE,
        background_threshold: int = DEFAULT_THRESHOLD,
        min_shift: int = DEFAULT_MIN_SHIFT,
    ) -> None:
        self.target_size = target_size
        self.background_threshold = background_threshold
        self.min_shift = min_shift

    def process(self, image: np.ndarray) -> np.ndarray:
        """Применяет полный пайплайн (resize -> centering) к массиву изображения.

        Parameters
        ----------
        image : np.ndarray
            Изображение (H, W) или (H, W, C) произвольного размера.

        Returns
        -------
        processed : np.ndarray
            Изображение размера (target_size[1], target_size[0][, C]).
        """
        resized = resize_image(image, self.target_size)
        return center_image(
            resized,
            background_threshold=self.background_threshold,
            min_shift=self.min_shift,
        )

    def process_file(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> Path:
        """Загружает изображение с диска, обрабатывает и сохраняет как PNG.

        Parameters
        ----------
        input_path : str or Path
            Путь к исходному изображению.
        output_path : str or Path
            Путь для сохранения результата.

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
            logger.error("ImagePreprocessor.process_file: не удалось загрузить '%s'.", input_path)
            raise FileNotFoundError(f"Не удалось загрузить изображение: {input_path}")

        processed = self.process(image)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), processed)
        logger.debug("ImagePreprocessor.process_file: %s -> %s.", input_path, output_path)
        return output_path

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        output_prefix: str = "",
        pattern: str = "*",
    ) -> List[Path]:
        """Пакетно обрабатывает все изображения директории (NB1+NB2 за один проход).

        В отличие от resize_directory (NB1) + отдельного центрирования (NB2),
        промежуточный resized-файл на диск не пишется — resize и центрирование
        выполняются в памяти на одно изображение за раз. Если нужен именно
        двухстадийный пайплайн с сохранением промежуточного resize (как в
        оригинальных ноутбуках), используйте resize.resize_directory() и
        centering.center_directory() по отдельности, либо
        process_class_via_config() / pipeline.FursovPipeline (стадии
        "resize" и "center").

        Parameters
        ----------
        input_dir : str or Path
            Директория с исходными изображениями.
        output_dir : str or Path
            Директория для сохранения результатов.
        output_prefix : str, default=""
            Префикс имени выходного файла (например, "glioma" -> "glioma1.png").
        pattern : str, default="*"
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
                "ImagePreprocessor.process_directory: нет файлов по паттерну "
                "'%s' в '%s'.", pattern, input_dir,
            )
            raise ValueError(
                f"В директории '{input_dir}' не найдено файлов по паттерну '{pattern}'."
            )

        logger.info(
            "ImagePreprocessor.process_directory: %d файлов из %s -> %s.",
            len(input_paths), input_dir, output_dir,
        )
        output_paths = []
        for i, input_path in enumerate(input_paths, start=1):
            output_path = output_dir / f"{output_prefix}{i}.png"
            output_paths.append(self.process_file(input_path, output_path))

        logger.info(
            "ImagePreprocessor.process_directory: готово, %d файлов записано в %s.",
            len(output_paths), output_dir,
        )
        return output_paths

    def process_class_via_config(
        self, class_name: str, config: "DatasetConfig", pattern: str = "*"
    ) -> List[Path]:
        """Обрабатывает класс через DatasetConfig: raw/ -> resized/ -> centered/.

        Двухстадийный пайплайн, полностью повторяющий структуру исходных
        ноутбуков (NB1 сохраняет промежуточный resize, NB2 читает его и
        сохраняет центрированный результат) — в отличие от process_directory,
        здесь промежуточные resized-файлы реально пишутся на диск.

        Parameters
        ----------
        class_name : str
            Название класса ("glioma", "meningioma", "pituitary", "test").
        config : DatasetConfig
            Конфигурация датасета — определяет пути raw/resized/centered.
        pattern : str, default="*"
            Glob-паттерн для отбора исходных файлов в config.paths[class_name]["raw"].

        Returns
        -------
        centered_paths : List[Path]
            Пути к финальным центрированным PNG.
        """
        from subspace_conjugacy.preprocessing.centering import center_directory
        from subspace_conjugacy.preprocessing.resize import resize_directory

        raw_dir = config.paths[class_name]["raw"]
        resized_dir = config.paths[class_name]["resized"]
        centered_dir = config.paths[class_name]["centered"]
        logger.info(
            "ImagePreprocessor.process_class_via_config: класс='%s', raw=%s.",
            class_name, raw_dir,
        )

        resize_directory(
            raw_dir,
            resized_dir,
            size=self.target_size,
            output_prefix=class_name,
            pattern=pattern,
        )

        return center_directory(
            resized_dir,
            centered_dir,
            background_threshold=self.background_threshold,
            min_shift=self.min_shift,
            pattern=f"{class_name}*.png",
        )
