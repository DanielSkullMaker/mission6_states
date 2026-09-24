"""Векторизация видеопоследовательностей для метода подпространственной
сопряжённости (статья 4, "Метод подпространственной сопряжённости для
классификации аудио- и видеоданных: перенос за пределы статических
изображений", theory/article_plans/04_audio_video.txt).

Видео — это последовательность кадров ПЕРЕМЕННОЙ длины (разные записи длятся
разное время / состоят из разного числа кадров). Как и для аудио
(features/audio.py), единственное, чего не хватало в библиотеке для проверки
метода на видео, — способа привести запись переменной длины к вектору
ФИКСИРОВАННОЙ размерности. План статьи (раздел 5, шаг 2) явно называет два
способа:

  - "один опорный кадр" (vectorize_frames_keyframe) — берётся ОДИН
    представительный кадр записи (по умолчанию средний по времени),
    векторизуется как обычное изображение (features.vectorization) — самый
    дешёвый по итоговой размерности вариант, но игнорирует всю динамику
    записи, кроме единственного момента.
  - "конкатенация признаков кадров" (vectorize_frames_concat) — из записи
    равномерно отбирается ФИКСИРОВАННОЕ число кадров (n_frames), каждый
    векторизуется отдельно, все векторы склеиваются в один вектор длины
    n_frames x H x W.

План (раздел "Риски и ограничения") прямо предупреждает: рост размерности
входа при конкатенации кадров может быть непрактичным при большом n_frames
или высоком разрешении кадра — оба параметра поэтому по умолчанию скромные
(64x64, 8 кадров = 32768 признаков, того же порядка, что и MRI-датасет
проекта, 65536 признаков после развёртки 256x256).

Датасет Kvasir-Capsule (план, раздел 0) распространяется как отдельные
кадры видео капсульной эндоскопии, сгруппированные по записи, а не только
как цельные видеофайлы — поэтому здесь ДВА источника кадров:
  - extract_frames — из видеофайла (.mp4/.avi/...) через cv2.VideoCapture;
  - extract_frames_from_paths — из УЖЕ ГОТОВОЙ упорядоченной последовательности
    файлов-кадров (например, папка с кадрами одной записи Kvasir-Capsule).
Оба возвращают один и тот же формат (n_frames, H, W) и совместимы с
одинаковыми vectorize_frames_*/vectorize_video ниже.

Использует cv2 (opencv-python-headless) — уже входит в ОСНОВНЫЕ (не extras)
зависимости subspace_conjugacy (pyproject.toml; тот же опциональный импорт,
что и в features/vectorization.py) — новых зависимостей не требует.
"""

import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from subspace_conjugacy.features.vectorization import VectorizationType, vectorize_image

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

VideoMode = Literal["concat", "keyframe"]


def _select_frame_indices(total_frames: int, n_frames: int) -> np.ndarray:
    """Индексы n_frames кадров, равномерно распределённых по [0, total_frames)."""
    if total_frames <= 0:
        raise ValueError(f"Число доступных кадров должно быть > 0, получено {total_frames}.")
    return np.linspace(0, total_frames - 1, n_frames).round().astype(int)


def _finalize_frames(frames: List[np.ndarray], n_frames: int, source: str) -> np.ndarray:
    """Дополняет список кадров до n_frames повтором последнего (короткая запись)
    и проверяет, что хотя бы один кадр был прочитан."""
    if not frames:
        logger.error("_finalize_frames: не удалось прочитать ни одного кадра из %s.", source)
        raise ValueError(f"Не удалось прочитать ни одного кадра: {source}")
    if len(frames) < n_frames:
        logger.warning(
            "_finalize_frames: %s содержит меньше кадров (%d), чем n_frames=%d — "
            "последний кадр продублирован.", source, len(frames), n_frames,
        )
        frames = list(frames) + [frames[-1]] * (n_frames - len(frames))
    return np.stack(frames[:n_frames])


def extract_frames(
    video_path: Union[str, Path],
    n_frames: int = 8,
    resize: Optional[Tuple[int, int]] = (64, 64),
) -> np.ndarray:
    """Считывает видеофайл и равномерно отбирает n_frames кадров по всей записи.

    Parameters
    ----------
    video_path : str or Path
        Путь к видеофайлу (любой формат, поддерживаемый cv2.VideoCapture —
        .mp4, .avi и т.д.).
    n_frames : int, default=8
        Число кадров, отбираемых равномерно по времени записи (см. план,
        "Риски и ограничения" — держать небольшим, чтобы избежать
        непрактичного роста размерности при конкатенации, см.
        vectorize_frames_concat).
    resize : Tuple[int, int] or None, default=(64, 64)
        Размер (ширина, высота), до которого приводится каждый кадр перед
        переводом в grayscale. None — оставить исходный размер кадра
        (осторожно: может дать очень большую размерность вектора).

    Returns
    -------
    frames : np.ndarray
        Массив (n_frames, H, W), uint8, grayscale.

    Raises
    ------
    FileNotFoundError
        Если файл не существует.
    ValueError
        Если cv2 не установлен, файл не читается, или в нём 0 кадров.

    Notes
    -----
    Если реальных кадров в записи меньше n_frames, последний прочитанный
    кадр дублируется до нужного количества (с предупреждением в лог) —
    гарантирует, что ВСЕ записи датасета дают массив одинаковой формы
    независимо от фактической длины, что необходимо для дальнейшей
    векторизации в (M, N)-матрицу.
    """
    if cv2 is None:
        logger.error("extract_frames: OpenCV (cv2) не установлен.")
        raise ValueError(
            "OpenCV (cv2) not available. Install: pip install opencv-python-headless"
        )
    path = Path(video_path)
    if not path.exists():
        logger.error("extract_frames: файл не найден '%s'.", path)
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        logger.error("extract_frames: не удалось определить число кадров '%s'.", path)
        raise ValueError(f"Не удалось прочитать видео (0 кадров): {path}")

    wanted = _select_frame_indices(total, n_frames)
    wanted_set = set(wanted.tolist())
    frames: List[np.ndarray] = []
    frame_idx = 0
    while cap.isOpened() and len(frames) < len(wanted):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in wanted_set:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if resize is not None:
                gray = cv2.resize(gray, resize, interpolation=cv2.INTER_AREA)
            frames.append(gray)
            wanted_set.discard(frame_idx)
        frame_idx += 1
    cap.release()

    logger.debug(
        "extract_frames: %s, total_frames=%d -> %d отобранных кадров.",
        path, total, len(frames),
    )
    return _finalize_frames(frames, n_frames, str(path))


def extract_frames_from_paths(
    frame_paths: List[Union[str, Path]],
    n_frames: int = 8,
    resize: Optional[Tuple[int, int]] = (64, 64),
    backend: Literal["cv2", "pil"] = "cv2",
) -> np.ndarray:
    """Строит массив кадров из УЖЕ ГОТОВОЙ упорядоченной последовательности
    изображений (одна запись = один файл на кадр) — альтернатива extract_frames
    для датасетов, распространяемых как папки кадров (например, Kvasir-Capsule,
    план раздел 0), а не как цельные видеофайлы.

    Parameters
    ----------
    frame_paths : List[str or Path]
        Пути к файлам-кадрам ОДНОЙ записи, в порядке следования по времени
        (например, отсортированные по имени файла).
    n_frames : int, default=8
        Целевое число кадров — если ``frame_paths`` длиннее, кадры
        отбираются равномерно (как в extract_frames); если короче,
        последний кадр дублируется.
    resize : Tuple[int, int] or None, default=(64, 64)
        См. extract_frames.
    backend : {"cv2", "pil"}, default="cv2"
        Библиотека для загрузки отдельных файлов-кадров (см.
        features.vectorization.load_and_vectorize).

    Returns
    -------
    frames : np.ndarray
        Массив (n_frames, H, W), grayscale — тот же формат, что и у
        extract_frames, совместим с vectorize_video/vectorize_frames_*.
    """
    if not frame_paths:
        logger.error("extract_frames_from_paths: пустой список frame_paths.")
        raise ValueError("frame_paths не может быть пустым списком.")

    indices = _select_frame_indices(len(frame_paths), min(n_frames, len(frame_paths)))
    frames: List[np.ndarray] = [
        _load_grayscale_frame(frame_paths[int(idx)], resize=resize, backend=backend)
        for idx in indices
    ]
    return _finalize_frames(frames, n_frames, f"{len(frame_paths)} файлов-кадров")


def _load_grayscale_frame(
    path: Union[str, Path],
    resize: Optional[Tuple[int, int]],
    backend: Literal["cv2", "pil"],
) -> np.ndarray:
    """Загружает ОДИН файл-кадр как 2D grayscale-массив (без векторизации в
    вектор — используется extract_frames_from_paths, где нужна именно
    2D-форма, единообразная с extract_frames)."""
    file_path = Path(path)
    if not file_path.exists():
        logger.error("_load_grayscale_frame: файл не найден '%s'.", file_path)
        raise FileNotFoundError(f"Frame file not found: {file_path}")

    if backend == "cv2":
        if cv2 is None:
            logger.error("_load_grayscale_frame: OpenCV (cv2) не установлен.")
            raise ValueError(
                "OpenCV (cv2) not available. Install: pip install opencv-python-headless"
            )
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error("_load_grayscale_frame: cv2 не смог декодировать '%s'.", file_path)
            raise ValueError(f"Failed to load image with cv2: {file_path}")
        if resize is not None:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    elif backend == "pil":
        from PIL import Image

        img_pil = Image.open(file_path).convert("L")
        if resize is not None:
            img_pil = img_pil.resize(resize, Image.LANCZOS)
        img = np.array(img_pil)
    else:
        logger.error("_load_grayscale_frame: неизвестный backend '%s'.", backend)
        raise ValueError(f"Unknown backend: '{backend}'. Expected 'cv2' or 'pil'.")

    return img


def vectorize_frames_concat(
    frames: np.ndarray, method: VectorizationType = "horizontal",
) -> np.ndarray:
    """Конкатенация векторизованных кадров в один вектор длины n_frames*H*W.

    Parameters
    ----------
    frames : np.ndarray
        Массив (n_frames, H, W) — например, из extract_frames.
    method : {"horizontal", "vertical"}, default="horizontal"
        Направление развёртки КАЖДОГО кадра (см. features.vectorization);
        порядок самих кадров во времени сохраняется всегда (кадр 0 всегда
        идёт первым в итоговом векторе, независимо от method).

    Returns
    -------
    vector : np.ndarray
        Одномерный вектор длины n_frames * H * W.
    """
    vectors = [vectorize_image(frame, method=method) for frame in frames]
    return np.concatenate(vectors)


def vectorize_frames_keyframe(
    frames: np.ndarray,
    index: Union[int, Literal["middle"]] = "middle",
    method: VectorizationType = "horizontal",
) -> np.ndarray:
    """Векторизует ОДИН опорный кадр записи (по умолчанию — средний по времени).

    Parameters
    ----------
    frames : np.ndarray
        Массив (n_frames, H, W).
    index : int or "middle", default="middle"
        Индекс опорного кадра. "middle" — n_frames // 2 (кадр из середины
        записи — типично наименее подвержен эффектам начала/конца съёмки).
    method : {"horizontal", "vertical"}, default="horizontal"
        См. features.vectorization.

    Returns
    -------
    vector : np.ndarray
        Одномерный вектор длины H * W (той же длины, что и обычное
        изображение того же размера).
    """
    frame_idx = frames.shape[0] // 2 if index == "middle" else int(index)
    return vectorize_image(frames[frame_idx], method=method)


def vectorize_video(
    frames: np.ndarray,
    mode: VideoMode = "concat",
    method: VectorizationType = "horizontal",
    keyframe_index: Union[int, Literal["middle"]] = "middle",
) -> np.ndarray:
    """Векторизует запись целиком, выбирая один из двух способов плана
    (раздел 5, шаг 2): "concat" (vectorize_frames_concat) или "keyframe"
    (vectorize_frames_keyframe).
    """
    if mode == "concat":
        return vectorize_frames_concat(frames, method=method)
    elif mode == "keyframe":
        return vectorize_frames_keyframe(frames, index=keyframe_index, method=method)
    else:
        logger.error("vectorize_video: неизвестный mode '%s'.", mode)
        raise ValueError(f"Unknown mode: '{mode}'. Expected 'concat' or 'keyframe'.")


def load_and_vectorize_video(
    path: Union[str, Path],
    n_frames: int = 8,
    resize: Optional[Tuple[int, int]] = (64, 64),
    mode: VideoMode = "concat",
    method: VectorizationType = "horizontal",
    keyframe_index: Union[int, Literal["middle"]] = "middle",
) -> np.ndarray:
    """Загружает видеофайл с диска и полностью векторизует (extract_frames + vectorize_video)."""
    frames = extract_frames(path, n_frames=n_frames, resize=resize)
    return vectorize_video(frames, mode=mode, method=method, keyframe_index=keyframe_index)


def load_and_vectorize_video_batch(
    paths: List[Union[str, Path]],
    n_frames: int = 8,
    resize: Optional[Tuple[int, int]] = (64, 64),
    mode: VideoMode = "concat",
    method: VectorizationType = "horizontal",
    keyframe_index: Union[int, Literal["middle"]] = "middle",
) -> np.ndarray:
    """Загружает и векторизует батч видеофайлов — общий n_frames/resize/mode
    гарантирует одинаковую длину вектора для всех записей.

    Returns
    -------
    X : np.ndarray
        Матрица векторов размерности (len(paths), N), готовая для
        SubspaceConjugacyClassifier.fit()/predict().
    """
    logger.info(
        "load_and_vectorize_video_batch: старт, %d файлов (mode=%s, n_frames=%d, resize=%s).",
        len(paths), mode, n_frames, resize,
    )
    vectors = [
        load_and_vectorize_video(
            path, n_frames=n_frames, resize=resize, mode=mode,
            method=method, keyframe_index=keyframe_index,
        )
        for path in paths
    ]
    X = np.vstack(vectors)
    logger.info("load_and_vectorize_video_batch: готово, X.shape=%s.", X.shape)
    return X
