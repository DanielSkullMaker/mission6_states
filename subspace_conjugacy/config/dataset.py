"""Configuration module for dataset paths and hyperparameters.

Заменяет hardcoded macOS пути из ноутбуков на конфигурируемую структуру,
совместимую с Windows/Linux.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Конфигурация путей к датасету и гиперпараметров модели.

    Parameters
    ----------
    root : Path or str
        Корневая директория датасета (заменяет hardcoded
        /Users/vladkorsikov/research/dataset_main/).
    classes : List[str], default=["glioma", "meningioma", "pituitary"]
        Список классов опухолей мозга.
    n_subclasses : int, default=8
        Количество подклассов (кластеров) на каждый класс.
    subclass_factor : int, default=2
        Количество базисных векторов в каждом подклассе для классификации.
    image_size : tuple, default=(256, 256)
        Размер изображений после resize (ширина, высота) в пикселях.
    test_samples_per_class : int, default=25
        Количество тестовых изображений на класс (всего 75 для 3 классов).

    Attributes
    ----------
    paths : Dict[str, Dict[str, Path]]
        Словарь путей для каждого этапа pipeline:
        - raw: исходные изображения
        - resized: после resize до 256×256
        - centered: после центрирования
        - binarized: после Otsu-бинаризации (только для этапа определения
          проекции axial/sagittal/coronal, статья "Data Preprocessing",
          refactoring_plan.txt раздел 10 находка №4 — НЕ используется для
          обычной классификации типа опухоли)
        - vectors: CSV с векторами признаков
        - subclasses: базисы подклассов

    Examples
    --------
    >>> config = DatasetConfig(root=r"C:\\mission_6states\\data")
    >>> print(config.paths["glioma"]["raw"])
    WindowsPath('C:/mission_6states/data/glioma_raw')

    >>> # Linux/macOS
    >>> config = DatasetConfig(root="/home/user/datasets/brain_tumors")
    >>> print(config.paths["glioma"]["vectors"])
    PosixPath('/home/user/datasets/brain_tumors/5_all_vectors/glioma')
    """

    root: Path = field(default_factory=lambda: Path("data"))
    classes: List[str] = field(
        default_factory=lambda: ["glioma", "meningioma", "pituitary"]
    )
    n_subclasses: int = 8
    subclass_factor: int = 2
    image_size: tuple = (256, 256)
    test_samples_per_class: int = 25

    def __post_init__(self):
        """Преобразует root в Path и создаёт структуру путей."""
        self.root = Path(self.root).resolve()
        self._build_paths()
        logger.debug(
            "DatasetConfig: root=%s, classes=%s, n_subclasses=%d.",
            self.root, self.classes, self.n_subclasses,
        )

    def _build_paths(self) -> None:
        """Создаёт словарь путей для всех классов и этапов обработки."""
        self.paths: Dict[str, Dict[str, Path]] = {}

        for cls in self.classes:
            self.paths[cls] = {
                "raw": self.root / f"{cls}_raw",
                "resized": self.root / f"2_{cls}_resize",
                "centered": self.root / f"3_{cls}_centered",
                # "binarized" — только для этапа определения проекции (Otsu,
                # статья "Data Preprocessing", refactoring_plan.txt, раздел
                # 10, находка №4); обычная классификация типа опухоли эту
                # стадию не использует.
                "binarized": self.root / f"4_{cls}_binarized",
                "vectors": self.root / "5_all_vectors" / cls,
                "subclasses": self.root / "5_all_vectors" / cls,
            }

        # Специальные пути для test выборки
        self.paths["test"] = {
            "raw": self.root / "test_raw",
            "resized": self.root / "2_test_resize",
            "centered": self.root / "3_test_centered",
            "binarized": self.root / "4_test_binarized",
            "vectors": self.root / "5_all_vectors" / "test",
        }

    def get_vector_csv_path(
        self, class_name: str, vector_type: str = "horizontal"
    ) -> Path:
        """Возвращает путь к CSV файлу с векторами признаков.

        Parameters
        ----------
        class_name : str
            Название класса ("glioma", "meningioma", "pituitary", "test").
        vector_type : str, default="horizontal"
            Тип векторизации: "horizontal" или "vertical".

        Returns
        -------
        Path
            Путь к CSV файлу (напр., glioma_horizontal_vector.csv).
        """
        filename = f"{class_name}_{vector_type}_vector.csv"
        return self.paths[class_name]["vectors"] / filename

    def get_initial_pair_path(self, class_name: str) -> Path:
        """Возвращает путь к CSV с первой парой опорных векторов (Legacy NB4-5).

        Parameters
        ----------
        class_name : str
            Название класса ("glioma", "meningioma", "pituitary").

        Returns
        -------
        Path
            Путь к файлу (напр., glioma_class_first_and_second_core_vectors.csv).
        """
        filename = f"{class_name}_class_first_and_second_core_vectors.csv"
        return self.paths[class_name]["vectors"] / filename

    def get_center_vectors_path(self, class_name: str) -> Path:
        """Возвращает путь к CSV с индексами центров подклассов (A.2-A.3, NB5).

        Parameters
        ----------
        class_name : str
            Название класса ("glioma", "meningioma", "pituitary").

        Returns
        -------
        Path
            Путь к файлу (напр., glioma_class_core_vectors.csv).
        """
        filename = f"{class_name}_class_core_vectors.csv"
        return self.paths[class_name]["vectors"] / filename

    def get_subclass_pairs_path(self, class_name: str) -> Path:
        """Возвращает путь к CSV с парами (центр, второй вектор) подклассов (B.1, NB6).

        Parameters
        ----------
        class_name : str
            Название класса ("glioma", "meningioma", "pituitary").

        Returns
        -------
        Path
            Путь к файлу (напр., glioma_new_classes.csv).
        """
        filename = f"{class_name}_new_classes.csv"
        return self.paths[class_name]["vectors"] / filename

    def get_pipeline_metadata_path(self) -> Path:
        """Возвращает путь к JSON с метаданными обученного пайплайна.

        Используется save_pipeline_artifact/load_pretrained_classifier
        (io/vectors.py) для хранения гиперпараметров классификатора рядом
        с CSV базисов подклассов.

        Returns
        -------
        Path
            Путь к файлу (напр., root/pipeline_metadata.json).
        """
        return self.root / "pipeline_metadata.json"

    def get_subclass_bases_path(self, class_name: str) -> Path:
        """Возвращает путь к CSV с базисными векторами подклассов.

        Parameters
        ----------
        class_name : str
            Название класса ("glioma", "meningioma", "pituitary").

        Returns
        -------
        Path
            Путь к файлу (напр., 8_glioma_subclasses_vectors.csv).
        """
        filename = f"{self.n_subclasses}_{class_name}_subclasses_vectors.csv"
        return self.paths[class_name]["subclasses"] / filename

    def get_image_paths(
        self, class_name: str, stage: str, count: Optional[int] = None
    ) -> List[Path]:
        """Возвращает список путей к изображениям на заданном этапе.

        Parameters
        ----------
        class_name : str
            Название класса или "test".
        stage : str
            Этап обработки: "raw", "resized", "centered".
        count : int, optional
            Количество изображений. Если None, определяется автоматически
            (100 для классов, 75 для test).

        Returns
        -------
        List[Path]
            Список путей к изображениям.
        """
        if count is None:
            count = (
                self.test_samples_per_class * len(self.classes)
                if class_name == "test"
                else 100
            )

        base_dir = self.paths[class_name][stage]
        extension = "jpg" if stage == "raw" and class_name != "test" else "png"

        return [
            base_dir / f"{class_name}{i+1}.{extension}" for i in range(count)
        ]

    def create_directories(self, stages: Optional[List[str]] = None) -> None:
        """Создаёт необходимые директории для всех классов.

        Parameters
        ----------
        stages : List[str], optional
            Список этапов для создания директорий. Если None, создаются все.
            Возможные значения: "raw", "resized", "centered", "vectors".
        """
        if stages is None:
            stages = ["raw", "resized", "centered", "vectors", "subclasses"]

        logger.info("DatasetConfig.create_directories: stages=%s, root=%s.", stages, self.root)
        for cls in self.classes + ["test"]:
            for stage in stages:
                if stage in self.paths[cls]:
                    self.paths[cls][stage].mkdir(parents=True, exist_ok=True)

    @property
    def n_features(self) -> int:
        """Возвращает размерность вектора признаков (256×256 = 65536)."""
        return self.image_size[0] * self.image_size[1]

    @property
    def total_subclasses(self) -> int:
        """Общее количество подклассов для классификации (3 класса × 8 = 24)."""
        return len(self.classes) * self.n_subclasses

    def __repr__(self) -> str:
        return (
            f"DatasetConfig(\n"
            f"  root={self.root},\n"
            f"  classes={self.classes},\n"
            f"  n_subclasses={self.n_subclasses},\n"
            f"  image_size={self.image_size},\n"
            f"  n_features={self.n_features}\n"
            f")"
        )
