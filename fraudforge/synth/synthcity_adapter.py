"""synthcity adapter placeholder."""

from __future__ import annotations

import pandas as pd

from ..exceptions import MissingExtraError


class SynthCitySynthesizer:
    """Adapter for synthcity models (requires optional dependency)."""

    def __init__(self) -> None:  # pragma: no cover - import guard
        try:
            import synthcity  # noqa: F401
        except ImportError as exc:
            raise MissingExtraError("synthcity") from exc

    def fit(self, df: pd.DataFrame, *, metadata: dict[str, object] | None = None) -> None:
        raise MissingExtraError("synthcity")

    def sample(self, n: int, *, conditions: dict[str, object] | None = None) -> pd.DataFrame:
        raise MissingExtraError("synthcity")

    def calibrate_columns(
        self,
        df: pd.DataFrame,
        cols: list[str],
        key_cols: list[str],
    ) -> pd.DataFrame:
        raise MissingExtraError("synthcity")
