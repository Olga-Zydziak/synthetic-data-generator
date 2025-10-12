"""Reference dataset profiling and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import GeneratorConfig, ReferenceFitConfig
from ..exceptions import ConfigurationError


@dataclass(slots=True)
class ReferenceProfile:
    """Summary profile extracted from a reference dataset."""

    age_dist: dict[str, float]
    channel_dist: dict[str, float]
    region_dist: dict[str, float]
    merchant_category_dist: dict[str, float]
    fraud_rate: float
    fraud_type_dist: dict[str, float]
    amount_log_mean: float
    amount_log_sigma: float
    hour_hist: list[float]


class ReferenceProfiler:
    """Profiles reference transactions to calibrate generator settings."""

    def fit(
        self,
        df: pd.DataFrame,
        cfg: ReferenceFitConfig,
        rng: np.random.Generator | None = None,
    ) -> ReferenceProfile:
        if df.empty:
            raise ConfigurationError("Reference dataframe must not be empty")
        sampler = rng or np.random.default_rng()
        max_cats = cfg.fit_max_categories

        def _top_k(series: pd.Series) -> dict[str, float]:
            counts = series.value_counts().head(max_cats)
            if cfg.dp_epsilon is not None:
                noise = sampler.laplace(0.0, 1.0 / cfg.dp_epsilon, size=len(counts))
                counts = counts + noise
                counts = counts.clip(lower=0.0)
            total = counts.sum()
            if total <= 0:
                return {}
            normalized = (counts / total).to_dict()
            other_share = 1.0 - sum(normalized.values())
            if other_share > 0:
                normalized["__OTHER__"] = other_share
            return {str(k): float(v) for k, v in normalized.items()}

        age_dist = _top_k(df["age_band"].astype(str))
        channel_dist = _top_k(df["channel"].astype(str))
        region_dist = _top_k(df["region"].astype(str))
        merchant_dist = _top_k(df["merchant_category"].astype(str))

        is_fraud = df["is_fraud"].astype(bool)
        fraud_rate = float(is_fraud.mean())
        fraud_type_dist = (
            _top_k(df.loc[is_fraud, "fraud_type"].dropna().astype(str)) if fraud_rate > 0 else {}
        )

        log_amount = np.log(df["amount"].clip(lower=0.01))
        amount_log_mean = float(log_amount.mean())
        amount_log_sigma = float(log_amount.std(ddof=0))

        event_time = pd.to_datetime(df[cfg.time_col])
        hour_counts = event_time.dt.hour.value_counts().sort_index()
        if cfg.dp_epsilon is not None:
            noise = sampler.laplace(0.0, 1.0 / cfg.dp_epsilon, size=hour_counts.shape[0])
            hour_counts = hour_counts + noise
            hour_counts = hour_counts.clip(lower=0.0)
        hist = hour_counts.reindex(range(24), fill_value=0.0).to_numpy(dtype=float)
        total_hist = hist.sum() or 1.0
        hour_hist = (hist / total_hist).tolist()

        return ReferenceProfile(
            age_dist=age_dist,
            channel_dist=channel_dist,
            region_dist=region_dist,
            merchant_category_dist=merchant_dist,
            fraud_rate=fraud_rate,
            fraud_type_dist=fraud_type_dist,
            amount_log_mean=amount_log_mean,
            amount_log_sigma=amount_log_sigma,
            hour_hist=hour_hist,
        )


class ConfigCalibrator:
    """Applies reference profile insights to generator configuration."""

    def calibrate(self, profile: ReferenceProfile, base_cfg: GeneratorConfig) -> GeneratorConfig:
        updates: dict[str, Any] = {}
        if base_cfg.channel_dist is None:
            updates["channel_dist"] = profile.channel_dist
        if base_cfg.region_dist is None:
            updates["region_dist"] = profile.region_dist
        if base_cfg.merchant_category_dist is None:
            updates["merchant_category_dist"] = profile.merchant_category_dist
        if base_cfg.amount_model is None:
            updates["amount_model"] = {
                "log_mean": profile.amount_log_mean,
                "log_sigma": profile.amount_log_sigma,
            }
        if base_cfg.time_model is None:
            updates["time_model"] = {"hour_hist": profile.hour_hist}
        if not updates:
            return base_cfg
        return base_cfg.model_copy(update=updates)
