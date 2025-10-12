"""Base scenario utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from ..config import GeneratorConfig

DEFAULT_CHANNEL_DIST = {
    "APP": 0.35,
    "WEB": 0.35,
    "ATM": 0.1,
    "POS": 0.15,
    "WIRE": 0.05,
}
DEFAULT_REGION_DIST = {"NORTH": 0.25, "SOUTH": 0.25, "EAST": 0.25, "WEST": 0.25}
DEFAULT_MERCHANT_CAT_DIST = {
    "grocery": 0.2,
    "electronics": 0.15,
    "travel": 0.1,
    "restaurant": 0.2,
    "online_services": 0.15,
    "fashion": 0.2,
}

SAFE_MISSING_COLS = [
    "merchant_category",
    "merchant_country",
    "os",
    "app_version",
    "ip",
    "device_id",
]


@dataclass(slots=True)
class ScenarioTargets:
    """Scenario target counts for controlled fraud composition."""

    total_rows: int
    fraud_rows: int
    causal_rows: int = 0


class BaseScenario:
    """Base scenario helper providing common synthetic data generation."""

    name: ClassVar[str] = "base"

    def __init__(self, targets: ScenarioTargets) -> None:
        self._targets = targets
        self._remaining_fraud = targets.fraud_rows
        self._remaining_causal = targets.causal_rows

    def generate(
        self, n: int, rng: np.random.Generator, base_cfg: GeneratorConfig
    ) -> pd.DataFrame:
        """Generate transactions for the scenario.

        Args:
            n: Number of rows to create.
            rng: Scenario-specific RNG.
            base_cfg: Generator configuration.

        Returns:
            pd.DataFrame: Scenario-specific rows.

        Raises:
            NotImplementedError: Always; subclasses must override.

        Complexity:
            Time: O(n); Memory: O(n).
        """

        raise NotImplementedError("Subclasses must implement generate()")

    def _draw_exact_flags(self, n: int, rng: np.random.Generator, *, kind: str) -> np.ndarray:
        if kind == "fraud":
            remaining = self._remaining_fraud
        else:
            remaining = self._remaining_causal
        flags = np.zeros(n, dtype=bool)
        if remaining <= 0:
            return flags
        take = min(n, remaining)
        indices = rng.choice(n, size=take, replace=False)
        flags[indices] = True
        if kind == "fraud":
            self._remaining_fraud -= int(take)
        else:
            self._remaining_causal -= int(take)
        return flags

    def _select_distribution(
        self,
        cfg_dist: Mapping[str, float] | None,
        default: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        dist = dict(cfg_dist) if cfg_dist is not None else dict(default)
        keys = np.array(list(dist.keys()))
        probs = np.array(list(dist.values()), dtype=float)
        probs = probs / probs.sum()
        return keys, probs

    def _generate_base_frame(
        self,
        n: int,
        rng: np.random.Generator,
        cfg: GeneratorConfig,
    ) -> pd.DataFrame:
        age_keys, age_probs = self._select_distribution(cfg.age_dist, cfg.age_dist)
        channel_keys, channel_probs = self._select_distribution(
            cfg.channel_dist,
            DEFAULT_CHANNEL_DIST,
        )
        region_keys, region_probs = self._select_distribution(
            cfg.region_dist,
            DEFAULT_REGION_DIST,
        )
        merchant_keys, merchant_probs = self._select_distribution(
            cfg.merchant_category_dist, DEFAULT_MERCHANT_CAT_DIST
        )

        ages = rng.choice(age_keys, size=n, p=age_probs)
        channels = rng.choice(channel_keys, size=n, p=channel_probs)
        regions = rng.choice(region_keys, size=n, p=region_probs)
        merchants = rng.choice(merchant_keys, size=n, p=merchant_probs)

        device_type_map = {
            "APP": "mobile",
            "WEB": "desktop",
            "ATM": "atm",
            "POS": "pos",
            "WIRE": "desktop",
        }
        device_types = np.array([device_type_map.get(ch, "desktop") for ch in channels])

        amount_model = cfg.amount_model or {"log_mean": 3.5, "log_sigma": 0.8}
        amounts = rng.lognormal(
            mean=float(amount_model["log_mean"]),
            sigma=float(amount_model["log_sigma"]),
            size=n,
        )
        amounts = np.round(amounts, 2)

        hour_hist = cfg.time_model["hour_hist"] if cfg.time_model else np.ones(24) / 24
        hours = rng.choice(np.arange(24), size=n, p=hour_hist)
        minutes = rng.integers(0, 60, size=n)
        seconds = rng.integers(0, 60, size=n)
        day_offsets = rng.integers(0, cfg.days, size=n)
        base_date = pd.Timestamp(cfg.start_date)
        event_times = base_date + pd.to_timedelta(day_offsets, unit="D")
        event_times = event_times + pd.to_timedelta(hours, unit="h")
        event_times = event_times + pd.to_timedelta(minutes, unit="m")
        event_times = event_times + pd.to_timedelta(seconds, unit="s")

        def _format(prefix: str, values: np.ndarray) -> list[str]:
            return [f"{prefix}-{int(v):010d}" for v in values]

        customer_ids = _format("CUST", rng.integers(1, 9_999_999_999, size=n))
        account_ids = _format("ACCT", rng.integers(1, 9_999_999_999, size=n))
        device_ids = _format("DEV", rng.integers(1, 9_999_999_999, size=n))
        merchant_ids = _format("MCH", rng.integers(1, 9_999_999_999, size=n))

        os_options = np.array(["iOS", "Android", "Windows", "macOS", "Linux"])
        os_values = rng.choice(os_options, size=n)
        app_versions = [
            f"{rng.integers(1, 6)}.{rng.integers(0, 10)}.{rng.integers(0, 10)}"
            for _ in range(n)
        ]
        ip_blocks = rng.integers(1, 255, size=(n, 4))
        ips = [".".join(str(int(part)) for part in block) for block in ip_blocks]

        txns_last_24h = rng.poisson(2.0, size=n).astype(int)
        avg_amount_7d = np.round(np.clip(amounts * rng.uniform(0.6, 1.4, size=n), 1.0, None), 2)
        chargebacks = rng.poisson(0.2, size=n).astype(int)
        tenure = rng.integers(30, 3650, size=n)

        merchant_country_map = {
            "NORTH": "US",
            "SOUTH": "US",
            "EAST": "US",
            "WEST": "US",
        }
        merchant_country = np.array([merchant_country_map.get(region, "US") for region in regions])

        df = pd.DataFrame(
            {
                "transaction_id": [self._uuid_str(rng) for _ in range(n)],
                "event_time": event_times,
                "customer_id": customer_ids,
                "account_id": account_ids,
                "age_band": ages,
                "region": regions,
                "account_tenure_days": tenure,
                "channel": channels,
                "device_id": device_ids,
                "device_type": device_types,
                "os": os_values,
                "app_version": app_versions,
                "ip": ips,
                "merchant_id": merchant_ids,
                "merchant_category": merchants,
                "merchant_country": merchant_country,
                "amount": amounts,
                "currency": "USD",
                "txns_last_24h": txns_last_24h,
                "avg_amount_7d": avg_amount_7d,
                "chargeback_count_90d": chargebacks,
                "is_dirty": False,
                "dirty_issues": [[] for _ in range(n)],
            }
        )
        return df

    def _assign_fraud_types(
        self,
        df: pd.DataFrame,
        fraud_flags: np.ndarray,
        rng: np.random.Generator,
        cfg: GeneratorConfig,
    ) -> None:
        fraud_types = list(cfg.fraud_type_dist.keys())
        fraud_probs = np.array(list(cfg.fraud_type_dist.values()), dtype=float)
        fraud_probs = fraud_probs / fraud_probs.sum()
        count = int(fraud_flags.sum())
        if count == 0:
            df.loc[:, "fraud_type"] = None
            return
        sampled = rng.choice(fraud_types, size=count, p=fraud_probs)
        df.loc[:, "fraud_type"] = None
        fraud_indices = np.flatnonzero(fraud_flags)
        df.loc[df.index[fraud_indices], "fraud_type"] = sampled

    @staticmethod
    def _uuid_str(rng: np.random.Generator) -> str:
        ints = rng.integers(0, 2**32, size=4, dtype=np.uint32)
        return "-".join(f"{int(x):08x}" for x in ints)
