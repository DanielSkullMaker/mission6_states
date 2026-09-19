"""Векторизация изображений для метода подпространственной сопряжённости.

Преобразует 2D изображения в одномерные векторы признаков через
горизонтальную или вертикальную развёртку пикселей.
"""

import logging
from typing import List, Literal, Union
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


VectorizationType = Literal["horizontal", "vertical"]


def vectorize_image(
    image: np.ndarray,
    method: VectorizationType = "horizontal",
) -> np.ndarray:
    """Векторизует 2D изображение в одномерный вектор признаков.

    Parameters
    ----------
    image : np.ndarray
        Входное изображение размерности (H, W) для grayscale или
        (H, W, C) для цветного (будет преобразовано в grayscale).
    method : {"horizontal", "vertical"}, default="horizontal"
        Метод развёртки:
        - "horizontal": построчное сканирование (row-major order)
        - "vertical": постолбцовое сканирование

    Returns
    -------
    vector : np.ndarray
        Одномерный вектор признаков размерности (H*W,).

    Examples
    --------
    >>> img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    >>> vec = vectorize_image(img, method="horizontal")
    >>> vec.shape
    (65536,)

    Notes
    -----
    Метод эквивалентен numpy flatten/ravel с разным порядком:
    - horizontal → C-order (row-major)
    - vertical → F-order (column-major)

    Преимущество перед циклами из NB3:
    - horizontal: ~1000x быстрее (flatten вместо двойного цикла)
    - vertical: ~500x быстрее (ravel('F') вместо двойного цикла)
    """
    img_gray = _ensure_grayscale(image)

    if method == "horizontal":
        # Row-major order: построчное сканирование
        # Эквивалентно: for row in img: for pixel in row: append(pixel)
        return img_gray.flatten()  # C-order by default
    elif method == "vertical":
        # Column-major order: постолбцовое сканирование
        # Эквивалентно: for col in range(W): for row in img: append(row[col])
        return img_gray.ravel(order="F")
    else:
        logger.error("vectorize_image: неизвестный method '%s'.", method)
        raise ValueError(
            f"Unknown vectorization method: '{method}'. "
            f"Expected 'horizontal' or 'vertical'."
        )


def vectorize_batch(
    images: Union[List[np.ndarray], np.ndarray],
    method: VectorizationType = "horizontal",
) -> np.ndarray:
    """Векторизует батч изображений в матрицу признаков.

    Parameters
    ----------
    images : List[np.ndarray] or np.ndarray
        Список изображений или 3D массив (N, H, W).
    method : {"horizontal", "vertical"}, default="horizontal"
        Метод векторизации.

    Returns
    -------
    X : np.ndarray
        Матрица векторов размерности (N, H*W).

    Examples
    --------
    >>> imgs = [np.random.randint(0, 256, (256, 256)) for _ in range(10)]
    >>> X = vectorize_batch(imgs)
    >>> X.shape
    (10, 65536)
    """
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            # Массив (N, H, W)
            images_list = [images[i] for i in range(images.shape[0])]
        elif images.ndim == 4:
            # Массив (N, H, W, C) — преобразуем в grayscale
            images_list = [images[i] for i in range(images.shape[0])]
        else:
            raise ValueError(
                f"Expected 3D or 4D array, got {images.ndim}D array."
            )
    else:
        images_list = images

    vectors = [vectorize_image(img, method=method) for img in images_list]
    X = np.vstack(vectors)
    logger.debug("vectorize_batch: %d изображений (method=%s) -> X.shape=%s.", len(images_list), method, X.shape)
    return X


def load_and_vectorize(
    image_path: Union[str, Path],
    method: VectorizationType = "horizontal",
    backend: Literal["cv2", "pil"] = "cv2",
) -> np.ndarray:
    """Загружает изображение с диска и векторизует его.

    Parameters
    ----------
    image_path : str or Path
        Путь к изображению.
    method : {"horizontal", "vertical"}, default="horizontal"
        Метод векторизации.
    backend : {"cv2", "pil"}, default="cv2"
        Библиотека для загрузки изображения:
        - "cv2": OpenCV (cv2.imread)
        - "pil": Pillow (PIL.Image.open)

    Returns
    -------
    vector : np.ndarray
        Одномерный вектор признаков размерности (H*W,).

    Raises
    ------
    FileNotFoundError
        Если файл изображения не существует.
    ValueError
        Если backend недоступен или изображение не загружается.
    """
    path = Path(image_path)
    if not path.exists():
        logger.error("load_and_vectorize: файл не найден '%s'.", path)
        raise FileNotFoundError(f"Image file not found: {path}")

    if backend == "cv2":
        if cv2 is None:
            logger.error("load_and_vectorize: OpenCV не установлен.")
            raise ValueError(
                "OpenCV (cv2) not available. Install: pip install opencv-python-headless"
            )
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error("load_and_vectorize: cv2 не смог декодировать '%s'.", path)
            raise ValueError(f"Failed to load image with cv2: {path}")
    elif backend == "pil":
        if Image is None:
            logger.error("load_and_vectorize: Pillow не установлен.")
            raise ValueError(
                "Pillow (PIL) not available. Install: pip install pillow"
            )
        img_pil = Image.open(path).convert("L")
        img = np.array(img_pil)
    else:
        logger.error("load_and_vectorize: неизвестный backend '%s'.", backend)
        raise ValueError(f"Unknown backend: '{backend}'. Expected 'cv2' or 'pil'.")

    logger.debug("load_and_vectorize: %s (backend=%s, method=%s).", path, backend, method)
    return vectorize_image(img, method=method)


def load_and_vectorize_batch(
    image_paths: List[Union[str, Path]],
    method: VectorizationType = "horizontal",
    backend: Literal["cv2", "pil"] = "cv2",
) -> np.ndarray:
    """Загружает и векторизует батч изображений с диска.

    Parameters
    ----------
    image_paths : List[str or Path]
        Список путей к изображениям.
    method : {"horizontal", "vertical"}, default="horizontal"
        Метод векторизации.
    backend : {"cv2", "pil"}, default="cv2"
        Библиотека для загрузки.

    Returns
    -------
    X : np.ndarray
        Матрица векторов размерности (N, H*W).

    Examples
    --------
    >>> from subspace_conjugacy.config import DatasetConfig
    >>> config = DatasetConfig(root="data")
    >>> paths = config.get_image_paths("glioma", "centered", count=10)
    >>> X = load_and_vectorize_batch(paths, method="horizontal")
    >>> X.shape
    (10, 65536)
    """
    logger.info(
        "load_and_vectorize_batch: старт, %d файлов (method=%s, backend=%s).",
        len(image_paths), method, backend,
    )
    vectors = [
        load_and_vectorize(path, method=method, backend=backend)
        for path in image_paths
    ]
    X = np.vstack(vectors)
    logger.info("load_and_vectorize_batch: готово, X.shape=%s.", X.shape)
    return X


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Преобразует изображение в grayscale, если оно цветное.

    Parameters
    ----------
    image : np.ndarray
        Входное изображение (H, W) или (H, W, C).

    Returns
    -------
    img_gray : np.ndarray
        Grayscale изображение размерности (H, W).
    """
    if image.ndim == 2:
        # Уже grayscale
        return image
    elif image.ndim == 3:
        # Цветное изображение — берём первый канал или усредняем
        if image.shape[2] == 1:
            return image[:, :, 0]
        else:
            # Конвертация в grayscale через weighted average (ITU-R BT.601)
            # Эквивалентно cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(image.dtype)
    else:
        raise ValueError(
            f"Image must be 2D (H, W) or 3D (H, W, C), got {image.ndim}D array."
        )
