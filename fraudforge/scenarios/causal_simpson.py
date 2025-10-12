"""Simpson's paradox causal fraud scenario."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import GeneratorConfig
from .base import BaseScenario, ScenarioTargets


class CausalSimpsonScenario(BaseScenario):
    """Injects fraud rows exhibiting Simpson's paradox across region and channel."""

    name = "causal_simpson"

    DESCRIPTION = (
        "Higher transaction amounts appear safer within each region "
        "yet become riskier when regions are aggregated, mimicking a manual review bias."
    )

    def __init__(self, targets: ScenarioTargets) -> None:
        super().__init__(targets)

    def generate(self, n: int, rng: np.random.Generator, base_cfg: GeneratorConfig) -> pd.DataFrame:
        df = self._generate_base_frame(n, rng, base_cfg)
        df.loc[:, "scenario"] = self.name

        # Amplify regional amount differences to create paradox structure.
        regional_multiplier = {
            "NORTH": 1.6,
            "SOUTH": 0.9,
            "EAST": 1.4,
            "WEST": 1.0,
        }
        amounts = df["amount"].to_numpy()
        regions = df["region"].to_numpy()
        multipliers = np.vectorize(regional_multiplier.get)(regions)
        df.loc[:, "amount"] = np.round(amounts * multipliers, 2)

        desired = min(self._remaining_fraud, n)
        fraud_flags = np.zeros(n, dtype=bool)
        if desired > 0:
            indices = self._select_low_amount_indices(df, desired)
            fraud_flags[indices] = True
            self._remaining_fraud -= desired
            self._remaining_causal -= desired
        df.loc[:, "is_fraud"] = fraud_flags
        df.loc[:, "is_causal_fraud"] = fraud_flags
        self._assign_fraud_types(df, fraud_flags, rng, base_cfg)
        df.loc[:, "is_casual_fraud"] = df["is_causal_fraud"]
        return df

    @staticmethod
    def _select_low_amount_indices(df: pd.DataFrame, count: int) -> np.ndarray:
        indices: list[int] = []
        grouped = df.groupby("region")
        quota = max(1, count // max(1, grouped.ngroups))
        for _region, group in grouped:
            group_sorted = group.sort_values("amount")
            take = min(quota, group_sorted.shape[0])
            indices.extend(group_sorted.index[:take].tolist())
        if len(indices) < count:
            leftovers = df.sort_values("amount").index.tolist()
            for idx in leftovers:
                if idx not in indices:
                    indices.append(int(idx))
                    if len(indices) >= count:
                        break
        return np.array(indices[:count], dtype=int)
