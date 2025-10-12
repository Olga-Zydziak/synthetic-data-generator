"""Factory for optional synthesizer integrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..exceptions import MissingExtraError
from ..ports import Synthesizer


@dataclass(slots=True)
class SynthInfo:
    """Metadata about synthesizer usage for lineage tracking."""

    backend: str
    calibrate_cols: list[str]
    condition_cols: list[str]
    dp_epsilon: float | None = None
    quality_score: float | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "calibrate_cols": self.calibrate_cols,
            "condition_cols": self.condition_cols,
            "dp_epsilon": self.dp_epsilon,
            "quality_score": self.quality_score,
        }


class NoneSynthesizer:
    """No-op synthesizer used when no backend is requested."""

    def fit(self, df: pd.DataFrame, *, metadata: dict[str, Any] | None = None) -> None:
        return None

    def sample(self, n: int, *, conditions: dict[str, Any] | None = None) -> pd.DataFrame:
        raise MissingExtraError("none")

    def calibrate_columns(
        self,
        df: pd.DataFrame,
        cols: list[str],
        key_cols: list[str],
    ) -> pd.DataFrame:
        return df


def _load_faker() -> Synthesizer:
    from .faker_provider import FakerSynthesizer

    return FakerSynthesizer()


def _load_sdv() -> Synthesizer:
    from .sdv_adapter import SDVSynthesizer

    return SDVSynthesizer()


def _load_ydata() -> Synthesizer:
    from .ydata_adapter import YDataSynthesizer

    return YDataSynthesizer()


def _load_synthcity() -> Synthesizer:
    from .synthcity_adapter import SynthCitySynthesizer

    return SynthCitySynthesizer()


def _load_smartnoise() -> Synthesizer:
    from .smartnoise_adapter import SmartNoiseSynthesizer

    return SmartNoiseSynthesizer()


FACTORY: dict[str, Callable[[], Synthesizer]] = {
    "none": lambda: NoneSynthesizer(),
    "faker": _load_faker,
    "sdv": _load_sdv,
    "ydata": _load_ydata,
    "synthcity": _load_synthcity,
    "smartnoise": _load_smartnoise,
}


def create_synthesizer(
    backend: str,
    *,
    calibrate_cols: list[str],
    condition_cols: list[str],
    dp_epsilon: float | None = None,
) -> tuple[Synthesizer, SynthInfo]:
    """Instantiate synthesizer backend by name."""

    backend_key = backend.lower()
    if backend_key not in FACTORY:
        raise MissingExtraError(backend_key)
    synthesizer = FACTORY[backend_key]()
    info = SynthInfo(
        backend=backend_key,
        calibrate_cols=calibrate_cols,
        condition_cols=condition_cols,
        dp_epsilon=dp_epsilon,
    )
    return synthesizer, info
