"""Top-level package for the fraudforge synthetic data generator."""

from __future__ import annotations

from .config import (
    BucketOptions,
    DataQualityConfig,
    DataQualityIssue,
    GeneratorConfig,
    OutputOptions,
    ReferenceFitConfig,
)
from .generator import TransactionGenerator

__all__ = [
    "__version__",
    "BucketOptions",
    "DataQualityConfig",
    "DataQualityIssue",
    "GeneratorConfig",
    "OutputOptions",
    "ReferenceFitConfig",
    "TransactionGenerator",
]

__version__ = "0.1.0"
