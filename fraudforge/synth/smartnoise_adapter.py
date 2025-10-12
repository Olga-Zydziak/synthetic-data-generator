"""SmartNoise DP synthesizer adapter placeholder."""

from __future__ import annotations

import pandas as pd

from ..exceptions import MissingExtraError


class SmartNoiseSynthesizer:
    """Adapter for SmartNoise differential privacy synthesizers."""

    def __init__(self) -> None:  # pragma: no cover - import guard
        try:
            import smartnoise_synth  # noqa: F401
        except ImportError as exc:
            raise MissingExtraError("dp") from exc

    def fit(self, df: pd.DataFrame, *, metadata: dict[str, object] | None = None) -> None:
        raise MissingExtraError("dp")

    def sample(self, n: int, *, conditions: dict[str, object] | None = None) -> pd.DataFrame:
        raise MissingExtraError("dp")

    def calibrate_columns(
        self,
        df: pd.DataFrame,
        cols: list[str],
        key_cols: list[str],
    ) -> pd.DataFrame:
        raise MissingExtraError("dp")
