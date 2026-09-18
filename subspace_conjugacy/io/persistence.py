"""Модуль сериализации и сохранения моделей и базисных матриц подпространств.

Предоставляет функции для сохранения и загрузки обученных классификаторов
и матриц базисов Y_s в форматах Pickle, NumPy Archive (.npz) и JSON.
"""

import json
import os
import pickle
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Union
import numpy as np


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
        raise IOError(f"Не удалось сохранить модель в файл '{path}': {err}") from err


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
        raise FileNotFoundError(f"Файл модели не найден по пути: '{path}'.")

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as err:
        raise IOError(f"Ошибка при загрузке модели из файла '{path}': {err}") from err


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
        raise ValueError("Модель не содержит обученных подпространств 'subspaces_'.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_dict: Dict[str, np.ndarray] = {}
    for cls, bases_list in model.subspaces_.items():
        for idx, Y_s in enumerate(bases_list):
            key = f"subspace_{cls}_{idx}"
            export_dict[key] = Y_s

    np.savez_compressed(path, **export_dict)


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
        raise FileNotFoundError(f"JSON файл не найден: '{path}'.")

    with open(path, "r", encoding="utf-8") as f:
        raw_payload = json.load(f)

    subspaces_converted: Dict[str, List[np.ndarray]] = {}
    for cls_str, bases_list in raw_payload.get("subspaces", {}).items():
        subspaces_converted[cls_str] = [
            np.array(Y_list, dtype=np.float64) for Y_list in bases_list
        ]

    raw_payload["subspaces"] = subspaces_converted
    return raw_payload


if __name__ == "__main__":
    print("=== Запуск тестов и демонстрации модуля persistence.py ===\n")
    np.random.seed(42)

    # Заглушка модели для тестирования сохранения
    class MockSubspaceClassifier:
        def __init__(self):
            self.n_subclasses = 2
            self.n_features_in_ = 16
            self.is_fitted_ = True
            self.classes_ = np.array(["Glioma", "Meningioma"])
            # Две матрицы признаков 16x3 для каждого класса
            self.subspaces_ = {
                "Glioma": [np.random.randn(16, 3), np.random.randn(16, 3)],
                "Meningioma": [np.random.randn(16, 3), np.random.randn(16, 3)],
            }

    mock_model = MockSubspaceClassifier()

    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        # 1. Тестирование Pickle (save_model / load_model)
        print("1. Тестирование сохранения и загрузки Pickle:")
        pkl_path = dir_path / "model.pkl"
        save_model(mock_model, pkl_path)
        assert pkl_path.exists(), "Файл Pickle не был создан"

        loaded_model = load_model(pkl_path)
        assert loaded_model.is_fitted_ is True
        assert loaded_model.n_features_in_ == 16
        print("   Успешно восстановлена модель из Pickle.")

        # 1.1 Перехват ошибки при сохранении необученной модели
        mock_model.is_fitted_ = False
        try:
            save_model(mock_model, pkl_path)
        except RuntimeError as err:
            print(f"   [Перехвачена ожидаемая ошибка]: {err}")
        mock_model.is_fitted_ = True

        # 2. Тестирование экспорта и импорта .npz
        print("\n2. Тестирование архивации подпространств в .npz:")
        npz_path = dir_path / "subspaces.npz"
        export_subspaces_npz(mock_model, npz_path)
        assert npz_path.exists(), "Файл .npz не был создан"

        loaded_subspaces_npz = import_subspaces_npz(npz_path)
        assert "Glioma" in loaded_subspaces_npz
        assert len(loaded_subspaces_npz["Glioma"]) == 2
        assert loaded_subspaces_npz["Glioma"][0].shape == (16, 3)
        print("   Успешно импортированы матрицы Y_s из формата .npz.")

        # 3. Тестирование экспорта и импорта JSON
        print("\n3. Тестирование сохранения подпространств в JSON:")
        json_path = dir_path / "subspaces.json"
        export_subspaces_json(mock_model, json_path)
        assert json_path.exists(), "Файл JSON не был создан"

        json_data = import_subspaces_json(json_path)
        assert json_data["n_features_in"] == 16
        assert "Meningioma" in json_data["subspaces"]
        assert isinstance(json_data["subspaces"]["Meningioma"][0], np.ndarray)
        print("   Успешно прочитан и распарсен JSON файл с метаданными.")

    print("\n Все тесты сохранения и загрузки моделей пройдены!")