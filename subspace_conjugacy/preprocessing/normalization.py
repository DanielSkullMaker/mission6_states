"""Подавление фона по порогу яркости (NB2: normalization_colour/normalization_grey).

Ноутбук 2_dataset_preparing.ipynb (и его вариант 2_1) определял ДВЕ почти
идентичные функции — ``normalization_colour`` и ``normalization_grey`` —
которые вручную, попиксельно (двойной Python-цикл по H×W) обнуляли пиксели
с яркостью ниже порога, отдельно для цветного и серого представления одного
и того же изображения. Здесь это одна векторизованная функция через
numpy boolean-маску — она работает для обоих случаев (2D grayscale и 3D
цветное изображение), поэтому дублирование не нужно.

Расхождение с ноутбуком: в ``2_dataset_preparing.ipynb`` подавленный пиксель
получает значение 0, а в ``2_1_dataset_preparing.ipynb`` — 1 (единственное
отличие между двумя копиями ноутбука, см. refactoring_plan.txt и историю
анализа scripts/). Здесь используется 0 — стандартное значение "чёрного"
фона, совместимое с cv2/PIL и с реальным датасетом (datasets/*_centered).
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 10


def suppress_background(
    image: np.ndarray,
    threshold: int = DEFAULT_THRESHOLD,
    reference: np.ndarray = None,
) -> np.ndarray:
    """Обнуляет пиксели, где яркость ниже порога (векторизованный аналог NB2).

    Parameters
    ----------
    image : np.ndarray
        Изображение (H, W) для grayscale или (H, W, C) для цветного —
        именно этот массив обнуляется там, где ``reference < threshold``.
    threshold : int, default=10
        Порог яркости. Пиксели строго ниже порога считаются фоном.
    reference : np.ndarray, optional
        Grayscale-изображение (H, W), по которому строится маска фона.
        Если None, используется сам ``image`` (должен быть 2D в этом случае).
        Передавайте grayscale-версию, если ``image`` цветное — так функция
        подавляет и цветной, и серый вариант одной и той же маской, как в
        NB2 (маска строится по grayscale, применяется к обоим представлениям).

    Returns
    -------
    result : np.ndarray
        Копия ``image`` с обнулёнными фоновыми пикселями.

    Raises
    ------
    ValueError
        Если ``reference`` не 2D, либо его пространственные размеры (H, W)
        не совпадают с ``image``.

    Examples
    --------
    >>> gray = np.array([[5, 50], [3, 80]], dtype=np.uint8)
    >>> suppress_background(gray)
    array([[ 0, 50],
           [ 0, 80]], dtype=uint8)
    """
    if reference is None:
        reference = image

    if reference.ndim != 2:
        logger.error(
            "suppress_background: reference.ndim=%d, ожидалось 2D.", reference.ndim,
        )
        raise ValueError(
            f"reference должен быть 2D (grayscale), получено {reference.ndim}D."
        )
    if reference.shape[:2] != image.shape[:2]:
        logger.error(
            "suppress_background: несовпадение размеров reference=%s, image=%s.",
            reference.shape[:2], image.shape[:2],
        )
        raise ValueError(
            f"Пространственные размеры reference {reference.shape[:2]} не "
            f"совпадают с image {image.shape[:2]}."
        )

    mask = reference < threshold
    result = image.copy()
    result[mask] = 0
    logger.debug(
        "suppress_background: threshold=%d, подавлено %d/%d пикселей.",
        threshold, int(mask.sum()), mask.size,
    )
    return result
