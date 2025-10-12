"""ydata-synthetic adapter placeholder."""

from __future__ import annotations

import pandas as pd

from ..exceptions import MissingExtraError


class YDataSynthesizer:
    """Adapter for ydata-synthetic (requires optional dependency)."""

    def __init__(self) -> None:  # pragma: no cover - import guard
        try:
            import ydata_synthetic  # noqa: F401
        except ImportError as exc:
            raise MissingExtraError("ydata") from exc

    def fit(self, df: pd.DataFrame, *, metadata: dict[str, object] | None = None) -> None:
        raise MissingExtraError("ydata")

    def sample(self, n: int, *, conditions: dict[str, object] | None = None) -> pd.DataFrame:
        raise MissingExtraError("ydata")

    def calibrate_columns(
        self,
        df: pd.DataFrame,
        cols: list[str],
        key_cols: list[str],
    ) -> pd.DataFrame:
        raise MissingExtraError("ydata")
