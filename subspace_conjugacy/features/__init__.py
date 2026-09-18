"""Feature extraction package for image vectorization."""

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

__all__ = [
    # Vectorization
    "vectorize_image",
    "vectorize_batch",
    "load_and_vectorize",
    "load_and_vectorize_batch",
    # Extraction
    "extract_class_vectors",
    "extract_all_classes",
    "extract_training_data",
]
