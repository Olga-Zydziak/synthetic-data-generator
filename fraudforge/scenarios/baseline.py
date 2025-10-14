"""Baseline fraud scenario implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import GeneratorConfig
from .base import BaseScenario, ScenarioTargets


class BaselineFraudScenario(BaseScenario):
    """Generates background transactional behavior with configurable fraud rate."""

    name = "baseline"

    def __init__(self, targets: ScenarioTargets) -> None:
        super().__init__(targets)

    def generate(self, n: int, rng: np.random.Generator, base_cfg: GeneratorConfig) -> pd.DataFrame:
        """Generate baseline transactions.

        Args:
            n: Number of records.
            rng: Scenario specific RNG.
            base_cfg: Generator configuration.

        Returns:
            Dataframe with baseline transactions.

        Complexity:
            Time: O(n); Memory: O(n).
        """

        df = self._generate_base_frame(n, rng, base_cfg)
        fraud_flags = self._draw_exact_flags(n, rng, kind="fraud")
        df.loc[:, "is_fraud"] = fraud_flags
        df.loc[:, "is_causal_fraud"] = self._draw_exact_flags(n, rng, kind="causal")
        self._assign_fraud_types(df, fraud_flags, rng, base_cfg)
        df.loc[:, "scenario"] = self.name
        df.loc[:, "is_casual_fraud"] = df["is_causal_fraud"]
        return df
