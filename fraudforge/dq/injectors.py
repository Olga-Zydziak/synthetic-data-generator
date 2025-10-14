"""Data quality injectors for fraudforge."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from ..config import DataQualityConfig, DataQualityIssue

SAFE_STRING_COLS = [
    "merchant_id",
    "device_id",
    "ip",
    "os",
    "app_version",
    "merchant_category",
    "merchant_country",
]

SWAP_PAIRS = [("customer_id", "account_id"), ("merchant_id", "device_id")]


class DefaultDirtyInjector:
    """Injects realistic dirty data patterns into transaction batches."""

    def __init__(self, cfg: DataQualityConfig) -> None:
        self._cfg = cfg
        self._issues = list(cfg.issue_dist.keys()) if cfg.enabled else []
        self._issue_probs = (
            np.array([cfg.issue_dist[issue] for issue in self._issues], dtype=float)
            if cfg.enabled and self._issues
            else np.array([])
        )

    def apply(
        self, df: pd.DataFrame, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, Counter[str]]:
        """Apply dirty transformations to dataframe."""

        if not self._cfg.enabled or df.empty:
            return df, Counter()

        mutated = df.copy(deep=True)
        mutated["dirty_issues"] = mutated["dirty_issues"].apply(lambda issues: list(issues))
        issue_counter: Counter[str] = Counter()
        mask = rng.random(mutated.shape[0]) < self._cfg.row_dirty_rate
        indices = np.flatnonzero(mask)
        for idx in indices:
            issues = self._sample_issues(rng)
            for issue in issues:
                self._apply_issue(mutated, idx, issue, rng)
                issue_counter[issue.value] += 1
        if indices.size > 0:
            mutated.loc[indices, "is_dirty"] = True
        issue_counter["__rows__"] = int(indices.size)
        mutated.loc[:, "is_casual_fraud"] = mutated["is_causal_fraud"]
        return mutated, issue_counter

    def _sample_issues(self, rng: np.random.Generator) -> list[DataQualityIssue]:
        if not self._issues:
            return []
        count = int(rng.integers(1, self._cfg.max_issues_per_row + 1))
        if count <= 0 or self._issue_probs.size == 0:
            return []
        indices = rng.choice(len(self._issues), size=count, replace=False, p=self._issue_probs)
        return [self._issues[int(i)] for i in np.atleast_1d(indices)]

    def _apply_issue(
        self,
        df: pd.DataFrame,
        idx: int,
        issue: DataQualityIssue,
        rng: np.random.Generator,
    ) -> None:
        if issue == DataQualityIssue.MISSING_VALUES:
            self._missing_values(df, idx, rng)
        elif issue == DataQualityIssue.TYPOS_NOISE:
            self._typos_noise(df, idx, rng)
        elif issue == DataQualityIssue.OUTLIER_AMOUNT:
            self._outlier_amount(df, idx, rng)
        elif issue == DataQualityIssue.DUPLICATE_ROWS:
            self._duplicate_rows(df, idx, rng)
        elif issue == DataQualityIssue.SWAP_FIELDS:
            self._swap_fields(df, idx)
        elif issue == DataQualityIssue.DATE_JITTER:
            self._date_jitter(df, idx, rng)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported issue {issue}")
        df.at[idx, "dirty_issues"].append(issue.value)

    def _missing_values(self, df: pd.DataFrame, idx: int, rng: np.random.Generator) -> None:
        cols = self._cfg.missing_cols_whitelist or SAFE_STRING_COLS
        col = rng.choice(cols)
        if col == "transaction_id":
            return
        df.at[idx, col] = pd.NA

    def _typos_noise(self, df: pd.DataFrame, idx: int, rng: np.random.Generator) -> None:
        cols = self._cfg.typos_cols_whitelist or SAFE_STRING_COLS
        col = rng.choice(cols)
        value = str(df.at[idx, col])
        if not value:
            return
        insert_pos = int(rng.integers(0, len(value) + 1))
        noisy_char = chr(rng.integers(65, 91))
        mutated = value[:insert_pos] + noisy_char + value[insert_pos:]
        df.at[idx, col] = mutated

    def _outlier_amount(self, df: pd.DataFrame, idx: int, rng: np.random.Generator) -> None:
        factor = rng.lognormal(mean=2.0, sigma=0.5)
        new_amount = max(0.01, float(df.at[idx, "amount"]) * factor)
        df.at[idx, "amount"] = round(new_amount, 2)
        df.at[idx, "avg_amount_7d"] = round(new_amount * 0.8, 2)

    def _duplicate_rows(self, df: pd.DataFrame, idx: int, rng: np.random.Generator) -> None:
        source_idx = int(rng.integers(0, df.shape[0]))
        if source_idx == idx:
            source_idx = (source_idx + 1) % df.shape[0]
        for col in df.columns:
            if col == "transaction_id":
                continue
            df.at[idx, col] = df.at[source_idx, col]
        df.at[idx, "transaction_id"] = self._random_transaction_id(rng)
        jitter_seconds = int(rng.integers(-300, 301))
        df.at[idx, "event_time"] = df.at[idx, "event_time"] + pd.to_timedelta(
            jitter_seconds, unit="s"
        )
        df.at[idx, "amount"] = round(float(df.at[idx, "amount"]) * rng.uniform(0.95, 1.05), 2)

    def _swap_fields(self, df: pd.DataFrame, idx: int) -> None:
        for left, right in SWAP_PAIRS:
            left_val = df.at[idx, left]
            df.at[idx, left] = df.at[idx, right]
            df.at[idx, right] = left_val

    def _date_jitter(self, df: pd.DataFrame, idx: int, rng: np.random.Generator) -> None:
        jitter = int(rng.integers(-600, 601))
        df.at[idx, "event_time"] = df.at[idx, "event_time"] + pd.to_timedelta(jitter, unit="s")

    @staticmethod
    def _random_transaction_id(rng: np.random.Generator) -> str:
        ints = rng.integers(0, 2**32, size=4, dtype=np.uint32)
        return "-".join(f"{int(x):08x}" for x in ints)
