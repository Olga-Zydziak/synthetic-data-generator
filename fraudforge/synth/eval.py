"""Optional quality evaluation helpers."""

from __future__ import annotations

import pandas as pd

from ..exceptions import MissingExtraError


def evaluate_quality(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    """Compute SDMetrics quality score when available."""

    try:
        from sdmetrics.reports.single_table import QualityReport
    except ImportError as exc:  # pragma: no cover - import guard
        raise MissingExtraError("sdv") from exc
    report = QualityReport()
    report.generate(real_df, synth_df)
    return float(report.get_score())
