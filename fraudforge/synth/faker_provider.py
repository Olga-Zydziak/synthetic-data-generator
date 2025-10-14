"""Faker-based synthesizer for enriching categorical columns."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from ..exceptions import MissingExtraError

ALLOWED_COLUMNS = {"merchant_id", "device_id", "ip", "os", "app_version"}


class FakerSynthesizer:
    """Lightweight synthesizer using Faker and Mimesis when installed."""

    def __init__(self) -> None:
        try:
            from faker import Faker
        except ImportError as exc:  # pragma: no cover - import guard
            raise MissingExtraError("faker") from exc
        self._faker = Faker()
        self._faker.seed_instance(42)

    def fit(self, df: pd.DataFrame, *, metadata: dict[str, Any] | None = None) -> None:
        return None

    def sample(self, n: int, *, conditions: dict[str, Any] | None = None) -> pd.DataFrame:
        raise MissingExtraError("faker")

    def calibrate_columns(
        self,
        df: pd.DataFrame,
        cols: list[str],
        key_cols: list[str],
    ) -> pd.DataFrame:
        mutated = df.copy()
        for col in cols:
            if col not in ALLOWED_COLUMNS:
                continue
            mutated.loc[:, col] = [
                self._generate_value(col) for _ in range(mutated.shape[0])
            ]
        for key in key_cols:
            mutated.loc[:, key] = df[key]
        mutated.loc[:, "is_casual_fraud"] = mutated["is_causal_fraud"]
        return mutated

    def _generate_value(self, column: str) -> str:
        if column == "merchant_id":
            return f"MCH-{self._faker.unique.random_number(digits=8):08d}"
        if column == "device_id":
            return f"DEV-{self._faker.unique.random_number(digits=10):010d}"
        if column == "ip":
            return cast(str, self._faker.ipv4())
        if column == "os":
            return cast(str, self._faker.random_element([
                "iOS",
                "Android",
                "Windows",
                "macOS",
                "Linux",
            ]))
        if column == "app_version":
            return cast(str, self._faker.numerify(text="#.##.##"))
        return cast(str, self._faker.pystr())
