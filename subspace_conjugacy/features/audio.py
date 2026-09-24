"""Векторизация аудиосигналов для метода подпространственной сопряжённости
(статья 4, "Метод подпространственной сопряжённости для классификации
аудио- и видеоданных: перенос за пределы статических изображений",
theory/article_plans/04_audio_video.txt).

Показатель сопряжённости R(x, Y) (core/metrics.py) математически не завязан
на "изображенческую" природу входа — определён для ЛЮБОГО числового вектора
фиксированной длины (план, раздел 1). Единственное, чего не хватало в
библиотеке для проверки этого тезиса на аудио, — способа превратить
аудиозапись ПЕРЕМЕННОЙ длины в вектор признаков ФИКСИРОВАННОЙ длины
(features/vectorization.py решает похожую задачу для изображений, которые и
так уже имеют фиксированный размер после препроцессинга — здесь этот шаг
нужно сделать заново для звука). Модуль закрывает именно этот пробел, тремя
шагами (план, раздел 5, шаг 1):

  1. Одноканальный сигнал (переменной длины) -> спектрограмма (STFT) ИЛИ
     MFCC (мел-частотные кепстральные коэффициенты) — в обоих случаях
     двумерная матрица "частота/коэффициент x время", тоже переменной
     ширины по времени (зависит от длительности записи).
  2. Матрица приводится к ФИКСИРОВАННОМУ числу временных кадров (обрезка
     длинных записей, паддинг нулями коротких, pad_or_truncate_frames) —
     без этого шага SubspaceConjugacyClassifier.fit() не примет записи
     разной длины как строки одной (M, N)-матрицы.
  3. Двумерная матрица разворачивается в вектор УЖЕ СУЩЕСТВУЮЩЕЙ
     features.vectorization.vectorize_image — спектрограмма/MFCC такая же
     обычная 2D-матрица чисел, как и изображение, поэтому переиспользуется
     тот же код, а не пишется отдельная функция развёртки. Статья 3
     (её отчёт, раздел 2.3) доказала, что направление развёртки (параметр
     method) не может повлиять на решение ЧИСТОГО метода сопряжённости —
     здесь этот параметр оставлен настраиваемым только ради единообразия
     API и честного использования вместе с доменно-специфичными базовыми
     методами (план, раздел 0.1), для которых, в отличие от метода
     сопряжённости, направление развёртки не является безразличным выбором.

Использует только зависимости, уже входящие в ОСНОВНЫЕ (не extras)
зависимости subspace_conjugacy — numpy и scipy (pyproject.toml) — новых
зависимостей не требует.
"""

import logging
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np

from subspace_conjugacy.features.vectorization import VectorizationType, vectorize_image

logger = logging.getLogger(__name__)

try:
    from scipy.io import wavfile
    from scipy.signal import stft
    from scipy.fft import dct
except ImportError:  # pragma: no cover - scipy входит в основные зависимости
    wavfile = None
    stft = None
    dct = None

AudioRepresentation = Literal["stft", "mfcc"]


def load_waveform(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """Загружает одноканальный сигнал из .wav файла.

    Parameters
    ----------
    path : str or Path
        Путь к .wav файлу (например, запись ICBHI 2017 Respiratory Sound
        Database — план, раздел 0).

    Returns
    -------
    waveform : np.ndarray
        Одномерный сигнал, float64, нормализован в диапазон [-1, 1].
    sample_rate : int
        Частота дискретизации записи (Гц).

    Raises
    ------
    FileNotFoundError
        Если файл не существует.
    ValueError
        Если scipy не установлен.

    Notes
    -----
    Многоканальные записи усредняются в моно (среднее по каналам) — метод
    сопряжённости работает с одним числовым вектором на объект, отдельная
    обработка стереоканалов выходит за рамки этой статьи.
    """
    if wavfile is None:
        logger.error("load_waveform: scipy.io.wavfile недоступен.")
        raise ValueError("scipy не установлен. Установите: pip install scipy")

    file_path = Path(path)
    if not file_path.exists():
        logger.error("load_waveform: файл не найден '%s'.", file_path)
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    sample_rate, data = wavfile.read(str(file_path))
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)

    max_abs = np.max(np.abs(data))
    if max_abs > 0:
        data = data / max_abs

    logger.debug(
        "load_waveform: %s, sample_rate=%d, длительность=%.2fс.",
        file_path, sample_rate, len(data) / sample_rate,
    )
    return data, int(sample_rate)


def compute_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 256,
    log_scale: bool = True,
) -> np.ndarray:
    """Вычисляет амплитудную спектрограмму одноканального сигнала (STFT).

    Parameters
    ----------
    waveform : np.ndarray
        Одномерный сигнал (например, из load_waveform).
    sample_rate : int
        Частота дискретизации (Гц).
    n_fft : int, default=512
        Длина окна БПФ (число отсчётов на кадр).
    hop_length : int, default=256
        Шаг между соседними кадрами (число отсчётов) — n_fft - hop_length
        отсчётов перекрытия между кадрами.
    log_scale : bool, default=True
        Логарифмировать ли амплитуду (log1p) — человеческое восприятие
        громкости и типичный динамический диапазон записи ближе к
        логарифмической, чем к линейной шкале; стандартная практика перед
        подачей спектрограммы в любой классификатор.

    Returns
    -------
    spectrogram : np.ndarray
        Матрица (n_freq_bins, n_time_frames), n_freq_bins = n_fft // 2 + 1.
        Число временных кадров n_time_frames зависит от длительности
        ``waveform`` — приведение к фиксированному размеру делает
        pad_or_truncate_frames.
    """
    if stft is None:
        logger.error("compute_spectrogram: scipy.signal.stft недоступен.")
        raise ValueError("scipy не установлен. Установите: pip install scipy")

    _, _, Zxx = stft(
        waveform, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length,
    )
    magnitude = np.abs(Zxx)
    if log_scale:
        magnitude = np.log1p(magnitude)

    logger.debug(
        "compute_spectrogram: waveform.shape=%s -> spectrogram.shape=%s.",
        waveform.shape, magnitude.shape,
    )
    return magnitude


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_filters: int, n_fft: int, sample_rate: int, fmin: float = 0.0, fmax: float = None,
) -> np.ndarray:
    """Строит треугольный мел-фильтр-банк (стандартная конструкция MFCC).

    Parameters
    ----------
    n_filters : int
        Число треугольных фильтров (типично 20-40).
    n_fft : int
        Длина окна БПФ, использованная при вычислении спектрограммы —
        определяет число частотных бинов n_fft // 2 + 1.
    sample_rate : int
        Частота дискретизации (Гц).
    fmin : float, default=0.0
        Нижняя граница диапазона частот (Гц).
    fmax : float, optional
        Верхняя граница диапазона частот (Гц). По умолчанию — частота
        Найквиста (sample_rate / 2).

    Returns
    -------
    filterbank : np.ndarray
        Матрица (n_filters, n_fft // 2 + 1) — строка m ненулевая только в
        треугольной окрестности m-й мел-полосы.

    Notes
    -----
    Мел-шкала сжимает высокие частоты сильнее низких (ближе к человеческому
    восприятию высоты тона) — формула перевода Гц в мел:
    mel = 2595 * log10(1 + hz / 700). Фильтры равномерно распределены ПО
    МЕЛ-ШКАЛЕ, а не по герцам, поэтому на низких частотах они уже (выше
    разрешение), а на высоких — шире.
    """
    fmax = sample_rate / 2.0 if fmax is None else fmax
    mel_min, mel_max = _hz_to_mel(np.array(fmin)), _hz_to_mel(np.array(fmax))
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    n_freq_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_filters, n_freq_bins), dtype=np.float64)
    for m in range(1, n_filters + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_left, min(f_center, n_freq_bins)):
            if f_center > f_left:
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, min(f_right, n_freq_bins)):
            if f_right > f_center:
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    return filterbank


def compute_mfcc(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 40,
    n_mfcc: int = 13,
) -> np.ndarray:
    """Вычисляет мел-частотные кепстральные коэффициенты (MFCC).

    Parameters
    ----------
    waveform : np.ndarray
        Одномерный сигнал.
    sample_rate : int
        Частота дискретизации (Гц).
    n_fft, hop_length : int
        См. compute_spectrogram.
    n_mels : int, default=40
        Число мел-фильтров (см. mel_filterbank).
    n_mfcc : int, default=13
        Число возвращаемых кепстральных коэффициентов (после дискретного
        косинусного преобразования оставляются только первые n_mfcc —
        они несут основную форму спектральной огибающей, старшие
        коэффициенты в основном шум).

    Returns
    -------
    mfcc : np.ndarray
        Матрица (n_mfcc, n_time_frames).

    Notes
    -----
    Классический конвейер MFCC: мощность спектра -> проекция на
    мел-фильтр-банк -> логарифм -> дискретное косинусное преобразование
    (DCT-II) по частотной оси. DCT дополнительно декоррелирует признаки
    (соседние мел-полосы сильно коррелируют между собой) — стандартная
    причина, по которой MFCC часто предпочитают сырой мел-спектрограмме для
    классических (не свёрточных) моделей.
    """
    if dct is None:
        logger.error("compute_mfcc: scipy.fft.dct недоступен.")
        raise ValueError("scipy не установлен. Установите: pip install scipy")

    power_spec = compute_spectrogram(
        waveform, sample_rate, n_fft=n_fft, hop_length=hop_length, log_scale=False,
    ) ** 2
    filterbank = mel_filterbank(n_mels, n_fft, sample_rate)
    mel_spec = filterbank @ power_spec
    log_mel = np.log(mel_spec + 1e-10)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]

    logger.debug(
        "compute_mfcc: waveform.shape=%s -> mfcc.shape=%s.", waveform.shape, mfcc.shape,
    )
    return mfcc


def pad_or_truncate_frames(matrix: np.ndarray, n_frames: int) -> np.ndarray:
    """Приводит матрицу "признак x время" к фиксированному числу кадров.

    Parameters
    ----------
    matrix : np.ndarray
        Матрица (F, T) — F признаков (частотных бинов или MFCC-коэффициентов)
        на T временных кадров.
    n_frames : int
        Целевое число временных кадров.

    Returns
    -------
    fixed : np.ndarray
        Матрица (F, n_frames). Если T > n_frames — обрезка справа (более
        поздняя часть записи отбрасывается); если T < n_frames — паддинг
        нулями справа (тишина); если T == n_frames — возвращается копия без
        изменений.
    """
    n_features, n_time = matrix.shape
    if n_time == n_frames:
        return matrix.copy()
    if n_time > n_frames:
        return matrix[:, :n_frames].copy()

    padded = np.zeros((n_features, n_frames), dtype=matrix.dtype)
    padded[:, :n_time] = matrix
    return padded


def vectorize_audio(
    waveform: np.ndarray,
    sample_rate: int,
    representation: AudioRepresentation = "stft",
    n_fft: int = 512,
    hop_length: int = 256,
    n_frames: int = 128,
    n_mels: int = 40,
    n_mfcc: int = 13,
    method: VectorizationType = "horizontal",
) -> np.ndarray:
    """Полный конвейер: сигнал -> спектрограмма/MFCC -> фиксированный размер -> вектор.

    Parameters
    ----------
    waveform : np.ndarray
        Одномерный сигнал (например, из load_waveform).
    sample_rate : int
        Частота дискретизации (Гц).
    representation : {"stft", "mfcc"}, default="stft"
        "stft" — амплитудная спектрограмма (compute_spectrogram); "mfcc" —
        мел-частотные кепстральные коэффициенты (compute_mfcc, заметно
        компактнее по числу "частотных" признаков: n_mfcc обычно 13-40
        против n_fft // 2 + 1 у STFT).
    n_fft, hop_length : int
        См. compute_spectrogram.
    n_frames : int, default=128
        Целевое число временных кадров после pad_or_truncate_frames — все
        записи датасета получают вектор ОДНОЙ и той же длины независимо от
        исходной длительности.
    n_mels, n_mfcc : int
        Действуют только при representation="mfcc" (см. compute_mfcc).
    method : {"horizontal", "vertical"}, default="horizontal"
        Направление развёртки двумерной матрицы в вектор (см. docstring
        модуля — для чистого метода сопряжённости результат от этого
        параметра не зависит, статья 3).

    Returns
    -------
    vector : np.ndarray
        Одномерный вектор признаков. Длина: (n_fft // 2 + 1) * n_frames для
        "stft", n_mfcc * n_frames для "mfcc".
    """
    if representation == "stft":
        matrix = compute_spectrogram(waveform, sample_rate, n_fft=n_fft, hop_length=hop_length)
    elif representation == "mfcc":
        matrix = compute_mfcc(
            waveform, sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, n_mfcc=n_mfcc,
        )
    else:
        logger.error("vectorize_audio: неизвестный representation '%s'.", representation)
        raise ValueError(
            f"Unknown representation: '{representation}'. Expected 'stft' or 'mfcc'."
        )

    matrix = pad_or_truncate_frames(matrix, n_frames)
    return vectorize_image(matrix, method=method)


def load_and_vectorize_audio(
    path: Union[str, Path],
    representation: AudioRepresentation = "stft",
    n_fft: int = 512,
    hop_length: int = 256,
    n_frames: int = 128,
    n_mels: int = 40,
    n_mfcc: int = 13,
    method: VectorizationType = "horizontal",
) -> np.ndarray:
    """Загружает .wav файл с диска и полностью векторизует (см. vectorize_audio)."""
    waveform, sample_rate = load_waveform(path)
    return vectorize_audio(
        waveform, sample_rate, representation=representation, n_fft=n_fft,
        hop_length=hop_length, n_frames=n_frames, n_mels=n_mels, n_mfcc=n_mfcc,
        method=method,
    )


def load_and_vectorize_audio_batch(
    paths: List[Union[str, Path]],
    representation: AudioRepresentation = "stft",
    n_fft: int = 512,
    hop_length: int = 256,
    n_frames: int = 128,
    n_mels: int = 40,
    n_mfcc: int = 13,
    method: VectorizationType = "horizontal",
) -> np.ndarray:
    """Загружает и векторизует батч .wav файлов — общий n_frames гарантирует
    одинаковую длину вектора для всех записей независимо от их длительности.

    Returns
    -------
    X : np.ndarray
        Матрица векторов размерности (len(paths), N), готовая для
        SubspaceConjugacyClassifier.fit()/predict().
    """
    logger.info(
        "load_and_vectorize_audio_batch: старт, %d файлов (representation=%s, n_frames=%d).",
        len(paths), representation, n_frames,
    )
    vectors = [
        load_and_vectorize_audio(
            path, representation=representation, n_fft=n_fft, hop_length=hop_length,
            n_frames=n_frames, n_mels=n_mels, n_mfcc=n_mfcc, method=method,
        )
        for path in paths
    ]
    X = np.vstack(vectors)
    logger.info("load_and_vectorize_audio_batch: готово, X.shape=%s.", X.shape)
    return X
