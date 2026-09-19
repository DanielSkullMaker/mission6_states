"""Отдельные стадии end-to-end пайплайна (refactoring_plan.txt, раздел «ФАЗА 6»).

Каждая функция — тонкая обёртка над уже существующим модулем библиотеки;
``stages.py`` не содержит новой логики, только сборку в единый реестр
``STAGE_REGISTRY``, которым пользуется ``FursovPipeline.run_stage()``.

| stage             | Теория  | Модуль                    | Ноутбук |
|-------------------|---------|---------------------------|---------|
| resize            | —       | preprocessing.resize      | NB1     |
| center            | —       | preprocessing.centering   | NB2     |
| vectorize         | вход X  | features                  | NB3     |
| global_pair       | A.1     | algorithms.global_pair    | —       |
| reference_centers | A.2-A.3 | algorithms.reference_centers | NB5  |
| subclass_seed     | B.1     | algorithms.subclass_seed  | NB6     |
| subclass_growth   | B.2     | algorithms.subclass_growth| NB7     |
| cluster           | A+B     | algorithms.fursov_clusterer | —     |
| export_subspaces  | export  | algorithms.subclass_export| NB6-7   |
| classify          | C       | models.classifier         | NB8     |
| legacy_notebook   | —       | algorithms.legacy.notebook_pipeline | NB4-7 |

Канон — единственный путь в "cluster"/"classify" (refactoring_plan.txt,
раздел 6, п.1); "legacy_notebook" существует только для parity, не
используется в run_class()/run_all_classes().
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def stage_resize(
    config: "DatasetConfig",
    class_name: str,
    pattern: str = "*.jpg",
) -> List[Path]:
    """NB1: config.paths[class_name]["raw"] -> ["resized"]."""
    from subspace_conjugacy.preprocessing.resize import resize_directory

    logger.info("stage_resize: класс='%s'.", class_name)
    return resize_directory(
        config.paths[class_name]["raw"],
        config.paths[class_name]["resized"],
        size=config.image_size,
        output_prefix=class_name,
        pattern=pattern,
    )


def stage_center(
    config: "DatasetConfig",
    class_name: str,
    background_threshold: int = 10,
    min_shift: int = 2,
) -> List[Path]:
    """NB2: config.paths[class_name]["resized"] -> ["centered"]."""
    from subspace_conjugacy.preprocessing.centering import center_directory

    logger.info("stage_center: класс='%s'.", class_name)
    return center_directory(
        config.paths[class_name]["resized"],
        config.paths[class_name]["centered"],
        background_threshold=background_threshold,
        min_shift=min_shift,
        pattern=f"{class_name}*.png",
    )


def stage_vectorize(
    config: "DatasetConfig",
    class_name: str,
    stage: str = "centered",
    method: str = "horizontal",
    save: bool = True,
    count: Optional[int] = None,
) -> np.ndarray:
    """NB3: изображения на заданной стадии -> матрица векторов (M, N).

    При ``save=True`` дополнительно сохраняет CSV в формате NB3
    (io.vectors.save_class_vectors), чтобы результат был переиспользуем
    другими стадиями/ноутбуками без повторной векторизации.

    Parameters
    ----------
    count : int, optional
        Количество изображений. Если None, ``DatasetConfig.get_image_paths``
        использует фиксированное значение по умолчанию (100 для классов, или
        test_samples_per_class * len(classes) для "test") — это соответствует
        оригинальному датасету ноутбуков, но не произвольному по размеру
        датасету. Здесь count автоматически определяется подсчётом реальных
        файлов ``{class_name}*.png`` на стадии ``stage``, если явно не задан.
    """
    from subspace_conjugacy.features.extraction import extract_class_vectors
    from subspace_conjugacy.io.vectors import save_class_vectors

    if count is None:
        stage_dir = config.paths[class_name][stage]
        count = len(list(stage_dir.glob(f"{class_name}*.png")))
        if count == 0:
            logger.error(
                "stage_vectorize: нет файлов '%s*.png' в %s.", class_name, stage_dir,
            )
            raise ValueError(
                f"Не найдено файлов '{class_name}*.png' в {stage_dir}. "
                "Убедитесь, что стадия 'resize'/'center' уже выполнена, "
                "либо передайте count явно."
            )

    logger.info(
        "stage_vectorize: класс='%s', stage='%s', count=%d, save=%s.",
        class_name, stage, count, save,
    )
    X = extract_class_vectors(config, class_name, stage=stage, method=method, count=count)
    if save:
        save_class_vectors(X, class_name, config, method)
    return X


def stage_global_pair(X: np.ndarray) -> "GlobalMinCosinePairFinder":
    """Теория A.1: глобальная пара с минимальным косинусным сходством."""
    from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder

    logger.info("stage_global_pair: X.shape=%s.", X.shape)
    finder = GlobalMinCosinePairFinder()
    finder.fit(X)
    return finder


def stage_reference_centers(
    X: np.ndarray,
    initial_pair,
    n_subclasses: int = 8,
    reg_param: float = 1e-8,
) -> "ReferenceCenterBuilder":
    """Теория A.2-A.3: последовательные центры через минимум R."""
    from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder

    logger.info("stage_reference_centers: n_subclasses=%d.", n_subclasses)
    builder = ReferenceCenterBuilder(n_subclasses=n_subclasses, reg_param=reg_param)
    builder.fit(X, initial_pair)
    return builder


def stage_subclass_seed(X: np.ndarray, center_indices) -> "CosineSecondVectorAttacher":
    """Теория B.1: второй вектор для каждого центра через минимум cos."""
    from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher

    logger.info("stage_subclass_seed: %d центров.", len(center_indices))
    attacher = CosineSecondVectorAttacher()
    attacher.fit(X, center_indices)
    return attacher


def stage_subclass_growth(
    X: np.ndarray,
    pairs,
    freeze_basis_at: Optional[int] = 2,
    strategy: str = "default",
    reg_param: float = 1e-8,
) -> "ConjugacyClusterGrowth":
    """Теория B.2: последовательное наполнение кластеров через максимум R."""
    from subspace_conjugacy.algorithms.subclass_growth import ConjugacyClusterGrowth

    logger.info("stage_subclass_growth: strategy=%s.", strategy)
    growth = ConjugacyClusterGrowth(
        freeze_basis_at=freeze_basis_at, strategy=strategy, reg_param=reg_param
    )
    growth.fit(X, pairs)
    return growth


def stage_cluster(
    X: np.ndarray,
    n_subclasses: int = 8,
    freeze_basis_at: Optional[int] = 2,
    growth_strategy: str = "default",
    reg_param: float = 1e-8,
) -> "FursovClusterer":
    """Фасад A.1->A.3->B.1->B.2 (канон) — один вызов вместо 4 стадий выше."""
    from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer

    logger.info("stage_cluster: X.shape=%s, n_subclasses=%d.", X.shape, n_subclasses)
    clusterer = FursovClusterer(
        n_subclasses=n_subclasses,
        freeze_basis_at=freeze_basis_at,
        growth_strategy=growth_strategy,
        reg_param=reg_param,
    )
    clusterer.fit(X)
    return clusterer


def stage_export_subspaces(
    clusterer: "FursovClusterer",
    class_name: str,
    config: "DatasetConfig",
) -> Path:
    """Экспорт базисов Y_s обученного clusterer в CSV (формат NB6-7)."""
    from subspace_conjugacy.algorithms.subclass_export import export_clusterer_bases

    logger.info("stage_export_subspaces: класс='%s'.", class_name)
    return export_clusterer_bases(
        clusterer, config.get_subclass_bases_path(class_name)
    )


def stage_classify(
    classifier: "SubspaceConjugacyClassifier",
    X_test: np.ndarray,
    y_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Теория C / NB8: предсказание + (опционально) отчёт accuracy."""
    from subspace_conjugacy.evaluation.metrics import evaluate_classifier

    logger.info("stage_classify: %d тестовых объектов.", X_test.shape[0])
    predictions = classifier.predict(X_test)
    result: Dict[str, Any] = {"predictions": predictions}
    if y_test is not None:
        result["report"] = evaluate_classifier(y_test, predictions)
    return result


def stage_legacy_notebook(
    X: np.ndarray,
    n_subclasses: int = 8,
    use_canonical_pair: bool = True,
    manual_pair_index: Optional[int] = None,
    strategy: str = "default",
    reg_param: float = 1e-8,
) -> Dict[str, Any]:
    """Staged NB4-7 (legacy) — ТОЛЬКО для parity-тестов, не для production.

    См. algorithms/legacy/notebook_pipeline.py. Не используется в
    FursovPipeline.run_class()/run_all_classes() — канон формируется через
    стадию "cluster".
    """
    from subspace_conjugacy.algorithms.legacy.notebook_pipeline import (
        NotebookStagedPipeline,
    )

    logger.info("stage_legacy_notebook [LEGACY]: use_canonical_pair=%s.", use_canonical_pair)
    pipeline = NotebookStagedPipeline(
        n_subclasses=n_subclasses,
        use_canonical_pair=use_canonical_pair,
        manual_pair_index=manual_pair_index,
        reg_param=reg_param,
    )
    return pipeline.run_full_pipeline(X, strategy=strategy)


STAGE_REGISTRY: Dict[str, Callable[..., Any]] = {
    "resize": stage_resize,
    "center": stage_center,
    "vectorize": stage_vectorize,
    "global_pair": stage_global_pair,
    "reference_centers": stage_reference_centers,
    "subclass_seed": stage_subclass_seed,
    "subclass_growth": stage_subclass_growth,
    "cluster": stage_cluster,
    "export_subspaces": stage_export_subspaces,
    "classify": stage_classify,
    "legacy_notebook": stage_legacy_notebook,
}
