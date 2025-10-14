"""SDV adapter placeholder."""

from __future__ import annotations

import pandas as pd

from ..exceptions import MissingExtraError


class SDVSynthesizer:
    """Adapter for SDV synthesizers (requires optional dependency)."""

    def __init__(self) -> None:  # pragma: no cover - import guard
        try:
            import sdv  # noqa: F401
        except ImportError as exc:
            raise MissingExtraError("sdv") from exc

    def fit(self, df: pd.DataFrame, *, metadata: dict[str, object] | None = None) -> None:
        raise MissingExtraError("sdv")

    def sample(self, n: int, *, conditions: dict[str, object] | None = None) -> pd.DataFrame:
        raise MissingExtraError("sdv")

    def calibrate_columns(
        self,
        df: pd.DataFrame,
        cols: list[str],
        key_cols: list[str],
    ) -> pd.DataFrame:
        raise MissingExtraError("sdv")
