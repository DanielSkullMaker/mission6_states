"""Subspace Conjugacy Method for MRI Brain Tumor Classification.

Библиотека реализует метод Fursov (подпространственная сопряжённость) для
классификации МРТ-изображений опухолей мозга.

Основные компоненты:
--------------------
- `conjugate_criterion`: вычисление показателя сопряжённости R(x, Y)
- `cosine_similarity_matrix`: косинусное сходство векторов
- `SubspaceClusterer`: кластеризация векторов по подпространствам
- `SubspaceConjugacyClassifier`: классификатор на основе подпространств

Примеры использования:
----------------------
>>> from subspace_conjugacy import SubspaceConjugacyClassifier
>>> clf = SubspaceConjugacyClassifier(n_subclasses=8)
>>> clf.fit(X_train, y_train)
>>> predictions = clf.predict(X_test)
"""

__version__ = "0.1.0"

import logging as _logging

# Библиотека не настраивает handlers сама (стандартная практика для
# библиотек) — NullHandler подавляет предупреждение "No handlers could be
# found" для потребителей, которые не настроили логирование. Чтобы увидеть
# подробный трейс работы алгоритмов, вызовите configure_logging().
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from subspace_conjugacy.utils.logging_config import configure_logging

# Core metrics
from subspace_conjugacy.core.metrics import (
    conjugate_criterion,
    cosine_similarity_matrix,
    compute_gram_inverse,
)

# Models
from subspace_conjugacy.models.base import BaseSubspaceEstimator
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier
from subspace_conjugacy.models.sequential_classifier import SequentialClassifier
from subspace_conjugacy.models.ensemble import (
    PrefitVotingClassifier,
    PrefitStackingClassifier,
    SwitchingEnsembleClassifier,
)
from subspace_conjugacy.models.multi_representation import (
    MultiRepresentationConjugacyClassifier,
)

# Canonical algorithms (фасад A+B)
from subspace_conjugacy.algorithms.fursov_clusterer import (
    FursovClusterer,
    SubspaceClusterer,  # Alias для обратной совместимости
)
from subspace_conjugacy.algorithms.reference_filter import LinearDependencyFilter
from subspace_conjugacy.algorithms.informativeness_filter import (
    DEFAULT_BRIGHTNESS_THRESHOLD,
    DEFAULT_MIN_FRACTION_OF_MEAN,
    LowInformativenessFilter,
)
from subspace_conjugacy.algorithms.correlated_pair_splitter import CorrelatedPairSplitter
from subspace_conjugacy.algorithms.subclass_export import equalize_subspace_bases

# Pipeline orchestrator (Фаза 6)
from subspace_conjugacy.pipeline import FursovPipeline

# Preprocessing (NB1-NB2)
from subspace_conjugacy.preprocessing import (
    ImagePreprocessor,
    center_image,
    resize_image,
    suppress_background,
)

# Otsu-бинаризация (статья, этап определения проекции, находка №4)
from subspace_conjugacy.preprocessing.binarization import otsu_binarize, otsu_threshold

# Feature extraction
from subspace_conjugacy.features.vectorization import (
    vectorize_image,
    vectorize_batch,
    vectorize_batch_multi,
    load_and_vectorize,
    load_and_vectorize_batch,
    load_and_vectorize_batch_multi,
)
from subspace_conjugacy.features.extraction import (
    extract_class_vectors,
    extract_all_classes,
    extract_training_data,
)

# Audio/video feature extraction (статья 4)
from subspace_conjugacy.features.audio import (
    load_waveform,
    compute_spectrogram,
    compute_mfcc,
    mel_filterbank,
    pad_or_truncate_frames,
    vectorize_audio,
    load_and_vectorize_audio,
    load_and_vectorize_audio_batch,
)
from subspace_conjugacy.features.video import (
    extract_frames,
    extract_frames_from_paths,
    vectorize_frames_concat,
    vectorize_frames_keyframe,
    vectorize_video,
    load_and_vectorize_video,
    load_and_vectorize_video_batch,
)

# Evaluation (Phase C, NB8)
from subspace_conjugacy.evaluation.metrics import (
    confidence_summary,
    evaluate_classifier,
    per_class_accuracy,
)
from subspace_conjugacy.evaluation.ensemble_diagnostics import (
    compute_ensemble_diagnostics,
)

# IO and persistence
from subspace_conjugacy.io.persistence import (
    save_model,
    load_model,
    export_subspaces_npz,
    import_subspaces_npz,
    export_subspaces_json,
    import_subspaces_json,
)
from subspace_conjugacy.io.vectors import (
    save_vectors_csv,
    load_vectors_csv,
    save_class_vectors,
    load_class_vectors,
    save_subclass_bases,
    load_subclass_bases,
    load_subclass_bases_as_list,
    save_initial_pair_indices,
    load_initial_pair_indices,
    save_center_indices,
    load_center_indices,
    save_subclass_pairs,
    load_subclass_pairs,
    save_pipeline_artifact,
    load_pretrained_classifier,
)

# Validation utilities
from subspace_conjugacy.utils.validation import (
    check_array_X,
    check_X_y,
    check_basis_matrix,
    check_hyperparameters,
    check_is_fitted,
)

# Hyperparameter search (grid search / random search)
from subspace_conjugacy.model_selection import (
    DEFAULT_PARAM_DISTRIBUTIONS,
    DEFAULT_PARAM_GRID,
    grid_search_classifier,
    random_search_classifier,
    search_hyperparameters,
    summarize_search_results,
)

__all__ = [
    # Version
    "__version__",
    # Logging
    "configure_logging",
    # Core metrics
    "conjugate_criterion",
    "cosine_similarity_matrix",
    "compute_gram_inverse",
    # Models
    "BaseSubspaceEstimator",
    "SubspaceConjugacyClassifier",
    "SequentialClassifier",
    "PrefitVotingClassifier",
    "PrefitStackingClassifier",
    "SwitchingEnsembleClassifier",
    "MultiRepresentationConjugacyClassifier",
    # Canonical algorithms
    "FursovClusterer",
    "SubspaceClusterer",  # Alias
    "LinearDependencyFilter",
    "LowInformativenessFilter",
    "DEFAULT_BRIGHTNESS_THRESHOLD",
    "DEFAULT_MIN_FRACTION_OF_MEAN",
    "CorrelatedPairSplitter",
    "equalize_subspace_bases",
    # Pipeline orchestrator
    "FursovPipeline",
    # Preprocessing
    "ImagePreprocessor",
    "resize_image",
    "suppress_background",
    "center_image",
    "otsu_threshold",
    "otsu_binarize",
    # Feature extraction
    "vectorize_image",
    "vectorize_batch",
    "vectorize_batch_multi",
    "load_and_vectorize",
    "load_and_vectorize_batch",
    "load_and_vectorize_batch_multi",
    "extract_class_vectors",
    "extract_all_classes",
    "extract_training_data",
    # Audio (статья 4)
    "load_waveform",
    "compute_spectrogram",
    "compute_mfcc",
    "mel_filterbank",
    "pad_or_truncate_frames",
    "vectorize_audio",
    "load_and_vectorize_audio",
    "load_and_vectorize_audio_batch",
    # Video (статья 4)
    "extract_frames",
    "extract_frames_from_paths",
    "vectorize_frames_concat",
    "vectorize_frames_keyframe",
    "vectorize_video",
    "load_and_vectorize_video",
    "load_and_vectorize_video_batch",
    # Evaluation
    "confidence_summary",
    "evaluate_classifier",
    "per_class_accuracy",
    "compute_ensemble_diagnostics",
    # IO - persistence
    "save_model",
    "load_model",
    "export_subspaces_npz",
    "import_subspaces_npz",
    "export_subspaces_json",
    "import_subspaces_json",
    # IO - vectors
    "save_vectors_csv",
    "load_vectors_csv",
    "save_class_vectors",
    "load_class_vectors",
    "save_subclass_bases",
    "load_subclass_bases",
    "load_subclass_bases_as_list",
    "save_initial_pair_indices",
    "load_initial_pair_indices",
    "save_center_indices",
    "load_center_indices",
    "save_subclass_pairs",
    "load_subclass_pairs",
    "save_pipeline_artifact",
    "load_pretrained_classifier",
    # Validation
    "check_array_X",
    "check_X_y",
    "check_basis_matrix",
    "check_hyperparameters",
    "check_is_fitted",
    # Hyperparameter search
    "DEFAULT_PARAM_GRID",
    "DEFAULT_PARAM_DISTRIBUTIONS",
    "grid_search_classifier",
    "random_search_classifier",
    "search_hyperparameters",
    "summarize_search_results",
]
