"""Ports and adapter interfaces for fraudforge."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd

from .config import GeneratorConfig, ReferenceFitConfig

__all__ = [
    "Scenario",
    "Writer",
    "DataQualityInjector",
    "ReferenceProfilerPort",
    "Synthesizer",
]


class Scenario(Protocol):
    """Scenario port for generating domain-specific transactions."""

    name: ClassVar[str]

    def generate(self, n: int, rng: np.random.Generator, base_cfg: GeneratorConfig) -> pd.DataFrame:
        """Generate a batch of transactions.

        Args:
            n: Number of rows to generate.
            rng: Deterministic RNG seeded per scenario.
            base_cfg: Validated generator configuration.

        Returns:
            pd.DataFrame: Generated transactions.

        Complexity:
            Time: O(n); Memory: O(n).
        """


class Writer(Protocol):
    """Streaming writer interface."""

    def write(self, df: pd.DataFrame) -> None:
        """Write a dataframe chunk.

        Args:
            df: Chunk dataframe.

        Complexity:
            Time: O(n); Memory: O(1) beyond buffers.
        """

    def finalize(self, metadata: dict[str, Any]) -> None:
        """Finalize output and emit metadata."""


class DataQualityInjector(Protocol):
    """Interface for injecting data quality issues."""

    def apply(
        self, df: pd.DataFrame, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Apply dirty data transformations.

        Args:
            df: Input chunk.
            rng: RNG for stochastic selection.

        Returns:
            Tuple containing mutated dataframe and issue counters.

        Complexity:
            Time: O(n); Memory: O(n) due to copy semantics.
        """


class ReferenceProfile(Protocol):
    """Lightweight profile object produced by ReferenceProfiler."""

    age_dist: dict[str, float]
    channel_dist: dict[str, float]
    region_dist: dict[str, float]
    merchant_category_dist: dict[str, float]
    fraud_rate: float
    fraud_type_dist: dict[str, float]
    amount_log_mean: float
    amount_log_sigma: float
    hour_hist: list[float]


class ReferenceProfilerPort(Protocol):
    """Protocol for profile fitting from reference datasets."""

    def fit(
        self,
        df: pd.DataFrame,
        cfg: ReferenceFitConfig,
        rng: np.random.Generator | None = None,
    ) -> ReferenceProfile:
        """Profile input dataframe and return normalized distributions."""


class Synthesizer(Protocol):
    """Protocol for optional synthesizer integrations."""

    def fit(self, df: pd.DataFrame, *, metadata: dict[str, Any] | None = None) -> None:
        """Fit synthesizer on dataframe."""

    def sample(self, n: int, *, conditions: dict[str, Any] | None = None) -> pd.DataFrame:
        """Sample new rows from synthesizer."""

    def calibrate_columns(
        self,
        df: pd.DataFrame,
        cols: list[str],
        key_cols: list[str],
    ) -> pd.DataFrame:
        """Calibrate specific columns while preserving key columns."""
