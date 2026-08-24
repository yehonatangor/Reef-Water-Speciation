"""Machine-learning layer: synthetic data generation, scaling and the CNN."""

from __future__ import annotations

from .dataset import Dataset, generate_dataset, split_dataset
from .noise import NoiseModel, NoisyCurve, apply_noise
from .normalization import Normalizer
from .sampling import Composition, SamplingRanges, sample_composition

__all__ = [
    "Composition",
    "Dataset",
    "NoiseModel",
    "NoisyCurve",
    "Normalizer",
    "SamplingRanges",
    "apply_noise",
    "generate_dataset",
    "sample_composition",
    "split_dataset",
]
