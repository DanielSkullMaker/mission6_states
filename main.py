"""Главный скрипт для запуска эксперимента классификации МРТ-изображений.

Загружает векторные датасеты патологий головного мозга (glioma_centered,
meningioma_centered, pituitary_centered), выполняет разбиение данных,
обучение подпространственного классификатора и сохранение результатов.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from subspace_conjugacy.features.vectorization import load_and_vectorize_batch
from subspace_conjugacy.io.persistence import (
    export_subspaces_json,
    save_model,
)
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier
from subspace_conjugacy.utils.validation import check_X_y

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_class_dataset(class_dir: Path) -> np.ndarray:
    """Загружает векторы признаков класса из файла или каталога.

    Директория с изображениями (.png/.jpg/.jpeg, как datasets/*_centered/
    — выход NB1-2) векторизуется через features.vectorization (NB3,
    horizontal method). Директория/файл с уже готовыми векторами
    (.npy/.csv) читается напрямую.

    Parameters
    ----------
    class_dir : Path
        Путь к директории класса или отдельному файлу (.npy/.csv).

    Returns
    -------
    vectors : np.ndarray
        Матрица векторов размерности (M, N).
    """
    if not class_dir.exists():
        raise FileNotFoundError(f"Каталог класса не найден: '{class_dir}'")

    if class_dir.is_file():
        if class_dir.suffix == ".npy":
            return np.load(class_dir)
        if class_dir.suffix == ".csv":
            return np.loadtxt(class_dir, delimiter=",")
        raise ValueError(f"Неподдерживаемый формат: '{class_dir.suffix}'")

    image_paths = sorted(
        p for p in class_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if image_paths:
        return load_and_vectorize_batch(image_paths, method="horizontal")

    vectors_list: List[np.ndarray] = []
    for file_path in sorted(class_dir.glob("*")):
        if file_path.suffix == ".npy":
            arr = np.load(file_path)
            vectors_list.append(arr.ravel())
        elif file_path.suffix == ".csv":
            arr = np.loadtxt(file_path, delimiter=",")
            vectors_list.append(arr.ravel())

    if not vectors_list:
        raise ValueError(
            f"В директории '{class_dir}' не найдены файлы формата "
            f".png/.jpg/.jpeg, .npy или .csv."
        )

    return np.vstack(vectors_list)


def load_all_datasets(
    dataset_paths: Dict[str, Path]
) -> Tuple[np.ndarray, np.ndarray]:
    """Загружает и объединяет векторизованные данные всех указанных классов.

    Parameters
    ----------
    dataset_paths : Dict[str, Path]
        Словарь соответствия 'Имя класса' -> 'Путь к директории'.

    Returns
    -------
    X : np.ndarray
        Общая матрица признаков (M, N).
    y : np.ndarray
        Вектор меток классов (M,).
    """
    x_list: List[np.ndarray] = []
    y_list: List[str] = []

    for class_name, path in dataset_paths.items():
        print(f"Загрузка данных для класса '{class_name}' из {path}...")
        vectors = load_class_dataset(path)
        x_list.append(vectors)
        y_list.extend([class_name] * vectors.shape[0])
        print(
            f" -> Загружено {vectors.shape[0]} векторов, "
            f"размерность: {vectors.shape[1]}"
        )

    X = np.vstack(x_list)
    y = np.array(y_list)

    return X, y


def main() -> None:
    """Запускает полный цикл эксперимента классификации."""
    print("=== Эксперимент классификации подпространственной сопряженности ===\n")

    # 1. Задание путей к указанным датасетам (метка класса -> директория)
    base_dir = Path("datasets")
    dataset_paths = {
        "glioma": base_dir / "glioma_centered",
        "meningioma": base_dir / "meningioma_centered",
        "pituitary": base_dir / "pituitary_centered",
    }

    # 2. Загрузка и валидация данных
    X_raw, y_raw = load_all_datasets(dataset_paths)
    X, y = check_X_y(X_raw, y_raw)

    print(f"\nИтоговый датасет: {X.shape[0]} объектов, {X.shape[1]} признаков.")
    classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(classes, counts):
        print(f" - Класс '{cls}': {count} объектов")

    # 3. Разделение на обучающую (70%) и тестовую (30%) выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(
        f"\nРазбиение завершено: Train = {X_train.shape[0]} об., "
        f"Test = {X_test.shape[0]} об."
    )

    # 4. Обучение классификатора
    n_subclasses = 8
    print(f"\nОбучение SubspaceConjugacyClassifier (подклассов: {n_subclasses})...")
    clf = SubspaceConjugacyClassifier(
        n_subclasses=n_subclasses,
        reg_param=1e-8,
    )
    clf.fit(X_train, y_train)
    print("Обучение успешно завершено.")

    # 5. Оценка качества на тестовой выборке
    print("\nКлассификация тестовых векторов...")
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print("\n" + "=" * 50)
    print(f"  Точность модели (Accuracy): {accuracy * 100:.2f}%")
    print("=" * 50 + "\n")

    print("Подробный отчет по классам (Classification Report):")
    print(classification_report(y_test, y_pred, digits=4))

    print("Матрица ошибок (Confusion Matrix):")
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    print(cm)

    # 6. Сохранение обученной модели и экспорт артефактов
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "subspace_model.pkl"
    json_path = artifacts_dir / "subspaces.json"

    save_model(clf, model_path)
    export_subspaces_json(clf, json_path)

    print(f"\nАртефакты сохранены в папку '{artifacts_dir}':")
    print(f" - Модель (Pickle): {model_path}")
    print(f" - Базисы (JSON):   {json_path}")
    print("\n=== Эксперимент успешно завершен ===")


if __name__ == "__main__":
    main()