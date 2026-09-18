"""Векторизация изображений для метода подпространственной сопряжённости.

Преобразует 2D изображения в одномерные векторы признаков через
горизонтальную или вертикальную развёртку пикселей.
"""

from typing import List, Literal, Union
import numpy as np
from pathlib import Path

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
    return np.vstack(vectors)


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
        raise FileNotFoundError(f"Image file not found: {path}")

    if backend == "cv2":
        if cv2 is None:
            raise ValueError(
                "OpenCV (cv2) not available. Install: pip install opencv-python-headless"
            )
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image with cv2: {path}")
    elif backend == "pil":
        if Image is None:
            raise ValueError(
                "Pillow (PIL) not available. Install: pip install pillow"
            )
        img_pil = Image.open(path).convert("L")
        img = np.array(img_pil)
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Expected 'cv2' or 'pil'.")

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
    vectors = [
        load_and_vectorize(path, method=method, backend=backend)
        for path in image_paths
    ]
    return np.vstack(vectors)


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


if __name__ == "__main__":
    print("=== Демонстрация векторизации изображений ===\n")
    np.random.seed(42)

    # 1. Векторизация одного изображения
    print("1. Векторизация синтетического изображения 256×256:")
    img_synthetic = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

    vec_hor = vectorize_image(img_synthetic, method="horizontal")
    vec_ver = vectorize_image(img_synthetic, method="vertical")

    print(f"   Horizontal vector: shape={vec_hor.shape}, dtype={vec_hor.dtype}")
    print(f"   Vertical vector:   shape={vec_ver.shape}, dtype={vec_ver.dtype}")
    print(f"   Первые 5 элементов (hor): {vec_hor[:5]}")
    print(f"   Первые 5 элементов (ver): {vec_ver[:5]}")

    # 2. Проверка эквивалентности с flatten/ravel
    print("\n2. Проверка эквивалентности с numpy flatten/ravel:")
    assert np.array_equal(vec_hor, img_synthetic.flatten()), "Horizontal должен совпадать с flatten()"
    assert np.array_equal(vec_ver, img_synthetic.ravel(order='F')), "Vertical должен совпадать с ravel('F')"
    print("   ✓ Horizontal == flatten()")
    print("   ✓ Vertical == ravel(order='F')")

    # 3. Векторизация батча
    print("\n3. Векторизация батча из 5 изображений:")
    imgs_batch = [np.random.randint(0, 256, (256, 256)) for _ in range(5)]
    X_batch = vectorize_batch(imgs_batch, method="horizontal")
    print(f"   Батч матрица: shape={X_batch.shape}, dtype={X_batch.dtype}")
    print(f"   Первый вектор (первые 5 элементов): {X_batch[0, :5]}")

    # 4. Обработка цветного изображения
    print("\n4. Автоматическая конвертация цветного изображения в grayscale:")
    img_color = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    vec_from_color = vectorize_image(img_color)
    print(f"   Исходное: (256, 256, 3) → Вектор: {vec_from_color.shape}")

    # 5. Бенчмарк: vectorize_image vs NB3 loops
    print("\n5. Сравнение производительности с циклами из NB3:")
    import time

    # Наша реализация
    start = time.perf_counter()
    for _ in range(100):
        _ = vectorize_image(img_synthetic, method="horizontal")
    time_optimized = time.perf_counter() - start

    # Эмуляция NB3 (двойной цикл)
    def nb3_vectorization_horizontal(img):
        vector = []
        for line in img:
            for pixel in line:
                vector.append(pixel)
        return np.array(vector)

    start = time.perf_counter()
    for _ in range(100):
        _ = nb3_vectorization_horizontal(img_synthetic)
    time_nb3 = time.perf_counter() - start

    speedup = time_nb3 / time_optimized
    print(f"   Оптимизированная версия: {time_optimized:.4f}s (100 итераций)")
    print(f"   NB3 двойной цикл:        {time_nb3:.4f}s (100 итераций)")
    print(f"   Ускорение: {speedup:.1f}x")

    print("\n✓ Все демонстрационные проверки завершены!")
