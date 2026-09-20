"""Models package: base estimator, clusterer, and classifier."""

from subspace_conjugacy.models.base import BaseSubspaceEstimator
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier
from subspace_conjugacy.models.sequential_classifier import SequentialClassifier

# Импорт канонического кластеризатора из algorithms
# (для обратной совместимости models.clusterer.SubspaceClusterer)
from subspace_conjugacy.algorithms.fursov_clusterer import (
    FursovClusterer,
    SubspaceClusterer,
)

__all__ = [
    "BaseSubspaceEstimator",
    "SubspaceConjugacyClassifier",
    "SequentialClassifier",
    "FursovClusterer",
    "SubspaceClusterer",
]
