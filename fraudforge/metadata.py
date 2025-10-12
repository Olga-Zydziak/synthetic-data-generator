"""Metadata aggregation utilities."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .config import GeneratorConfig

__all__ = ["MetadataCollector"]


class MetadataCollector:
    """Incrementally aggregates generation metadata."""

    def __init__(self, cfg: GeneratorConfig) -> None:
        self._cfg = cfg
        self._counts: Counter[str] = Counter()
        self._fraud_counts: dict[str, Counter[str]] = {
            "fraud_type": Counter(),
            "age_band": Counter(),
            "region": Counter(),
            "channel": Counter(),
            "merchant_category": Counter(),
        }
        self._causal_counts: Counter[str] = Counter()
        self._causal_descriptions: dict[str, str] = {}
        self._dirty_counts: Counter[str] = Counter()
        self._dirty_rows = 0
        self._fit_profile: dict[str, Any] | None = None
        self._synth_info: dict[str, Any] | None = None

    def register_causal_description(self, scenario: str, description: str) -> None:
        """Register human readable description for a causal scenario."""

        self._causal_descriptions[scenario] = description

    def update(
        self, df: pd.DataFrame, *, dirty_issues: Counter[str] | None = None
    ) -> None:
        """Update metadata with a newly generated chunk."""

        self._counts["total_records"] += int(df.shape[0])
        fraud_mask = df["is_fraud"].astype(bool)
        fraud_df = df.loc[fraud_mask]
        non_fraud = int(df.shape[0] - fraud_mask.sum())
        self._counts["fraud_total"] += int(fraud_mask.sum())
        self._counts["non_fraud"] += non_fraud

        for name, counter in self._fraud_counts.items():
            if not fraud_df.empty:
                counts = fraud_df[name].value_counts()
                for key, value in counts.items():
                    counter[str(key)] += int(value)

        causal_mask = df["is_causal_fraud"].astype(bool)
        causal_counts = df.loc[causal_mask, "scenario"].value_counts()
        for scenario, count in causal_counts.items():
            self._causal_counts[str(scenario)] += int(count)

        if dirty_issues is not None:
            issue_counter = Counter(dirty_issues)
            rows = issue_counter.pop("__rows__", 0)
            self._dirty_rows += rows
            for issue, value in issue_counter.items():
                self._dirty_counts[issue] += value
        else:
            dirty_mask = df["is_dirty"].astype(bool)
            self._dirty_rows += int(dirty_mask.sum())
            dirty_series = df.loc[dirty_mask, "dirty_issues"].explode().dropna()
            issue_counts = dirty_series.value_counts()
            for issue, value in issue_counts.items():
                self._dirty_counts[str(issue)] += int(value)

    def set_fit_profile(self, profile: dict[str, Any]) -> None:
        """Attach reference fit profile details."""

        self._fit_profile = profile

    def set_synth_info(self, info: dict[str, Any]) -> None:
        """Attach synthesizer metadata."""

        self._synth_info = info

    def finalize(self) -> dict[str, Any]:
        """Render aggregated metadata structure."""

        total = self._counts["total_records"] or 1
        causal_total = sum(self._causal_counts.values())
        dirty_rows = self._dirty_rows
        metadata: dict[str, Any] = {
            "counts": {
                "total_records": total,
                "non_fraud": self._counts["non_fraud"],
                "fraud_total": self._counts["fraud_total"],
                "fraud_by": {
                    key: dict(counter) for key, counter in self._fraud_counts.items()
                },
            },
            "causal": {
                "causal_fraud_count": causal_total,
                "causal_fraud_share": causal_total / total,
                "scenarios": {
                    name: {
                        "count": self._causal_counts.get(name, 0),
                        "description": self._causal_descriptions.get(name, ""),
                    }
                    for name in self._causal_descriptions
                },
            },
            "data_quality": {
                "dirty_rows": dirty_rows,
                "dirty_share": dirty_rows / total,
                "issues_by_type": dict(self._dirty_counts),
            },
            "lineage": {
                "seed": self._cfg.seed,
                "generator_version": "0.1.0",
                "timestamp": datetime.now(UTC).isoformat(),
                "config": self._cfg.model_dump(mode="json"),
            },
        }

        if self._fit_profile is not None:
            metadata["fit_profile"] = self._fit_profile
        if self._synth_info is not None:
            metadata["synth"] = self._synth_info

        return metadata
