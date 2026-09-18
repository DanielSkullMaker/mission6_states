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

# Core metrics
from subspace_conjugacy.core.metrics import (
    conjugate_criterion,
    cosine_similarity_matrix,
    compute_gram_inverse,
)

# Models
from subspace_conjugacy.models.base import BaseSubspaceEstimator
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier

# Canonical algorithms (фасад A+B)
from subspace_conjugacy.algorithms.fursov_clusterer import (
    FursovClusterer,
    SubspaceClusterer,  # Alias для обратной совместимости
)

# Feature extraction
from subspace_conjugacy.features.vectorization import (
    vectorize_image,
    vectorize_batch,
    load_and_vectorize,
    load_and_vectorize_batch,
)
from subspace_conjugacy.features.extraction import (
    extract_class_vectors,
    extract_all_classes,
    extract_training_data,
)

# Evaluation (Phase C, NB8)
from subspace_conjugacy.evaluation.metrics import (
    confidence_summary,
    evaluate_classifier,
    per_class_accuracy,
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

__all__ = [
    # Version
    "__version__",
    # Core metrics
    "conjugate_criterion",
    "cosine_similarity_matrix",
    "compute_gram_inverse",
    # Models
    "BaseSubspaceEstimator",
    "SubspaceConjugacyClassifier",
    # Canonical algorithms
    "FursovClusterer",
    "SubspaceClusterer",  # Alias
    # Feature extraction
    "vectorize_image",
    "vectorize_batch",
    "load_and_vectorize",
    "load_and_vectorize_batch",
    "extract_class_vectors",
    "extract_all_classes",
    "extract_training_data",
    # Evaluation
    "confidence_summary",
    "evaluate_classifier",
    "per_class_accuracy",
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
]
