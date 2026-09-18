"""Evaluation utilities for the subspace conjugacy classifier (Phase C, NB8)."""

from subspace_conjugacy.evaluation.metrics import (
    confidence_summary,
    evaluate_classifier,
    per_class_accuracy,
)

__all__ = [
    "confidence_summary",
    "evaluate_classifier",
    "per_class_accuracy",
]
