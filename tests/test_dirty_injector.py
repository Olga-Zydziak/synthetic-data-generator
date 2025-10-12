from __future__ import annotations

import numpy as np
import pandas as pd

from fraudforge.config import DataQualityConfig, DataQualityIssue
from fraudforge.dq.injectors import DefaultDirtyInjector


def base_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"tx-{i}" for i in range(10)],
            "event_time": pd.date_range("2024-01-01", periods=10, freq="T"),
            "amount": np.linspace(10, 100, 10),
            "avg_amount_7d": np.linspace(5, 50, 10),
            "txns_last_24h": np.zeros(10),
            "chargeback_count_90d": np.zeros(10),
            "is_fraud": [False] * 10,
            "fraud_type": [None] * 10,
            "is_causal_fraud": [False] * 10,
            "is_casual_fraud": [False] * 10,
            "scenario": ["baseline"] * 10,
            "is_dirty": [False] * 10,
            "dirty_issues": [[] for _ in range(10)],
        }
    )


def test_duplicate_rows_preserve_row_count() -> None:
    cfg = DataQualityConfig(
        enabled=True,
        row_dirty_rate=1.0,
        issue_dist={
            DataQualityIssue.DUPLICATE_ROWS: 0.5,
            DataQualityIssue.OUTLIER_AMOUNT: 0.5,
        },
    )
    injector = DefaultDirtyInjector(cfg)
    df = base_dataframe()
    mutated, issues = injector.apply(df, np.random.default_rng(0))
    assert mutated.shape[0] == df.shape[0]
    assert issues["__rows__"] == df.shape[0]
    assert mutated["transaction_id"].is_unique
