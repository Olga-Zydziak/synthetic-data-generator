"""Collider bias causal scenario."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import GeneratorConfig
from .base import BaseScenario, ScenarioTargets


class CausalColliderScenario(BaseScenario):
    """Scenario that flips fraud correlation after conditioning on review decisions."""

    name = "causal_collider"

    DESCRIPTION = (
        "Fraudulent accounts trigger manual reviews, "
        "making higher amounts appear safer within the reviewed subset "
        "and creating a collider bias between amount and fraud labels."
    )

    def __init__(self, targets: ScenarioTargets) -> None:
        super().__init__(targets)

    def generate(self, n: int, rng: np.random.Generator, base_cfg: GeneratorConfig) -> pd.DataFrame:
        df = self._generate_base_frame(n, rng, base_cfg)
        df.loc[:, "scenario"] = self.name

        latent_risk = (
            0.4 * df["txns_last_24h"].to_numpy()
            + 0.6 * df["chargeback_count_90d"].to_numpy()
            + rng.normal(0.0, 0.5, size=n)
        )
        thresholds = np.quantile(latent_risk, 0.7)
        reviewed = latent_risk > thresholds
        df.loc[reviewed, "amount"] = np.round(df.loc[reviewed, "amount"] * 0.7, 2)
        df.loc[~reviewed, "amount"] = np.round(df.loc[~reviewed, "amount"] * 1.2, 2)

        desired = min(self._remaining_fraud, n)
        fraud_flags = np.zeros(n, dtype=bool)
        if desired > 0:
            order = np.argsort(latent_risk)[::-1]
            fraud_indices = order[:desired]
            fraud_flags[fraud_indices] = True
            self._remaining_fraud -= desired
            self._remaining_causal -= desired
        df.loc[:, "is_fraud"] = fraud_flags
        df.loc[:, "is_causal_fraud"] = fraud_flags
        self._assign_fraud_types(df, fraud_flags, rng, base_cfg)
        df.loc[:, "is_casual_fraud"] = df["is_causal_fraud"]
        return df
