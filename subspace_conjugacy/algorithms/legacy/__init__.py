"""Legacy algorithms package for notebook parity tests only."""

from subspace_conjugacy.algorithms.legacy.per_vector_pairs import (
    compute_per_vector_pairs_nb4,
    extract_pair_from_trio_list,
    find_global_min_from_trio_list,
)
from subspace_conjugacy.algorithms.legacy.notebook_pipeline import NotebookStagedPipeline

__all__ = [
    "compute_per_vector_pairs_nb4",
    "extract_pair_from_trio_list",
    "find_global_min_from_trio_list",
    "NotebookStagedPipeline",
]
