"""Модуль сериализации и сохранения моделей и базисных матриц подпространств.

Предоставляет функции для сохранения и загрузки обученных классификаторов
и матриц базисов Y_s в форматах Pickle, NumPy Archive (.npz) и JSON.
"""

import json
import logging
import os
import pickle
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Union
import numpy as np

logger = logging.getLogger(__name__)


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """Сохраняет полный экземпляр обученной модели в файл формата Pickle.

    Parameters
    ----------
    model : Any
        Обученный экземпляр классификатора или модели.
    filepath : Union[str, Path]
        Путь для сохранения файла (например, 'model.pkl').

    Raises
    ------
    RuntimeError
        Если модель не обучена (отсутствует атрибут is_fitted_).
    IOError
        При ошибках записи на диск.
    """
    if getattr(model, "is_fitted_", False) is not True:
        logger.error("save_model: попытка сохранить необученную модель %r.", model)
        raise RuntimeError(
            "Попытка сохранить необученную модель. "
            "Вызовите метод 'fit' перед сохранением."
        )

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as err:
        logger.error("save_model: ошибка записи '%s': %s", path, err)
        raise IOError(f"Не удалось сохранить модель в файл '{path}': {err}") from err

    logger.info("save_model: модель %s сохранена в %s.", type(model).__name__, path)


def load_model(filepath: Union[str, Path]) -> Any:
    """Загружает сохраненный экземпляр модели из файла Pickle.

    Parameters
    ----------
    filepath : Union[str, Path]
        Путь к сохраненному файлу модели.

    Returns
    -------
    model : Any
        Восстановленный экземпляр модели.

    Raises
    ------
    FileNotFoundError
        Если файл по указанному пути не существует.
    IOError
        При ошибках чтения или повреждении файла.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error("load_model: файл не найден '%s'.", path)
        raise FileNotFoundError(f"Файл модели не найден по пути: '{path}'.")

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except Exception as err:
        logger.error("load_model: ошибка чтения '%s': %s", path, err)
        raise IOError(f"Ошибка при загрузке модели из файла '{path}': {err}") from err

    logger.info("load_model: модель %s загружена из %s.", type(model).__name__, path)
    return model


def export_subspaces_npz(model: Any, filepath: Union[str, Path]) -> None:
    """Экспортирует базисные матрицы Y_{c, s} в сжатый архив NumPy (.npz).

    Каждая базисная матрица сохраняется с ключом вида 'subspace_{class}_{subclass_idx}'.

    Parameters
    ----------
    model : Any
        Обученная модель, содержащая словарь subspaces_.
    filepath : Union[str, Path]
        Путь к сохраняемому архиву (например, 'subspaces.npz').
    """
    if not hasattr(model, "subspaces_") or not model.subspaces_:
        logger.error("export_subspaces_npz: модель %r не содержит subspaces_.", model)
        raise ValueError("Модель не содержит обученных подпространств 'subspaces_'.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_dict: Dict[str, np.ndarray] = {}
    for cls, bases_list in model.subspaces_.items():
        for idx, Y_s in enumerate(bases_list):
            key = f"subspace_{cls}_{idx}"
            export_dict[key] = Y_s

    np.savez_compressed(path, **export_dict)
    logger.info(
        "export_subspaces_npz: %d подпространств (%d классов) сохранено в %s.",
        len(export_dict), len(model.subspaces_), path,
    )


def import_subspaces_npz(filepath: Union[str, Path]) -> Dict[str, List[np.ndarray]]:
    """Импортирует базисные матрицы подпространств из файла формата .npz.

    Parameters
    ----------
    filepath : Union[str, Path]
        Путь к архиву .npz.

    Returns
    -------
    subspaces : Dict[str, List[np.ndarray]]
        Словарь, где ключ — имя класса, а значение — список матриц Y_s.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error("import_subspaces_npz: файл не найден '%s'.", path)
        raise FileNotFoundError(f"Архив подпространств не найден: '{path}'.")

    data = np.load(path)
    subspaces: Dict[str, List[np.ndarray]] = {}

    for key in sorted(data.files):
        # Ключи формируются как 'subspace_{class}_{subclass_idx}'
        parts = key.split("_")
        if len(parts) >= 3:
            cls_name = "_".join(parts[1:-1])
            Y_s = data[key]

            if cls_name not in subspaces:
                subspaces[cls_name] = []
            subspaces[cls_name].append(Y_s)

    logger.info(
        "import_subspaces_npz: загружено %d классов из %s.", len(subspaces), path,
    )
    return subspaces


def export_subspaces_json(
    model: Any, filepath: Union[str, Path], indent: int = 4
) -> None:
    """Экспортирует подпространства и метаданные модели в формат JSON.

    Parameters
    ----------
    model : Any
        Обученная модель подпространств.
    filepath : Union[str, Path]
        Путь для сохранения текстового JSON файла.
    indent : int, default=4
        Количество отступов для форматирования JSON.
    """
    if not hasattr(model, "subspaces_") or not model.subspaces_:
        logger.error("export_subspaces_json: модель %r не содержит subspaces_.", model)
        raise ValueError("Модель не содержит подпространств для экспорта.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_payload: Dict[str, Any] = {
        "n_subclasses": getattr(model, "n_subclasses", None),
        "n_features_in": getattr(model, "n_features_in_", None),
        "classes": (
            model.classes_.tolist()
            if hasattr(model, "classes_") and isinstance(model.classes_, np.ndarray)
            else list(model.subspaces_.keys())
        ),
        "subspaces": {},
    }

    for cls, bases_list in model.subspaces_.items():
        str_cls = str(cls)
        json_payload["subspaces"][str_cls] = [
            Y_s.tolist() for Y_s in bases_list
        ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=indent)
    logger.info(
        "export_subspaces_json: %d классов сохранено в %s.",
        len(json_payload["subspaces"]), path,
    )


def import_subspaces_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Загружает базисы и метаданные подпространств из файла JSON.

    Parameters
    ----------
    filepath : Union[str, Path]
        Путь к файлу JSON.

    Returns
    -------
    payload : Dict[str, Any]
        Словарь с восстановленными матрицами NumPy и метаданными.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error("import_subspaces_json: файл не найден '%s'.", path)
        raise FileNotFoundError(f"JSON файл не найден: '{path}'.")

    with open(path, "r", encoding="utf-8") as f:
        raw_payload = json.load(f)

    subspaces_converted: Dict[str, List[np.ndarray]] = {}
    for cls_str, bases_list in raw_payload.get("subspaces", {}).items():
        subspaces_converted[cls_str] = [
            np.array(Y_list, dtype=np.float64) for Y_list in bases_list
        ]

    raw_payload["subspaces"] = subspaces_converted
    logger.info(
        "import_subspaces_json: загружено %d классов из %s.",
        len(subspaces_converted), path,
    )
    return raw_payload
