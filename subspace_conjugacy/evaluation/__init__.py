"""Evaluation utilities for the subspace conjugacy classifier (Phase C, NB8)."""

from subspace_conjugacy.evaluation.metrics import (
    build_confusion_matrix,
    confidence_summary,
    confusion_matrix_to_table,
    evaluate_classifier,
    per_class_accuracy,
    plot_confusion_matrix,
)
from subspace_conjugacy.evaluation.paper_methodology import (
    DEFAULT_N_SUBCLASSES_GRID,
    compare_informativeness_filter,
    find_best_n_subclasses,
    paired_correlation_split,
    paper_train_test_split,
)

__all__ = [
    "confidence_summary",
    "evaluate_classifier",
    "per_class_accuracy",
    "build_confusion_matrix",
    "confusion_matrix_to_table",
    "plot_confusion_matrix",
    "paper_train_test_split",
    "find_best_n_subclasses",
    "compare_informativeness_filter",
    "paired_correlation_split",
    "DEFAULT_N_SUBCLASSES_GRID",
]
