"""Configuration models for the fraudforge generator."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .exceptions import ConfigurationError
from .models import AgeBand, Channel, FraudType, Region

__all__ = [
    "OutputOptions",
    "DataQualityIssue",
    "DataQualityConfig",
    "ReferenceFitConfig",
    "GeneratorConfig",
    "parse_generator_config",
]


class OutputOptions(BaseModel):
    """Output configuration for streaming writers."""

    model_config = ConfigDict(frozen=True)

    format: str = Field(pattern="^(csv|json|parquet)$")
    outdir: Path
    chunk_size: int = Field(ge=1, default=50_000)

    @model_validator(mode="after")
    def _resolve_outdir(self) -> OutputOptions:
        resolved = self.outdir.expanduser().resolve()
        object.__setattr__(self, "outdir", resolved)
        return self


class DataQualityIssue(str, Enum):
    """Enumeration of supported data quality issues."""

    MISSING_VALUES = "MISSING_VALUES"
    TYPOS_NOISE = "TYPOS_NOISE"
    OUTLIER_AMOUNT = "OUTLIER_AMOUNT"
    DUPLICATE_ROWS = "DUPLICATE_ROWS"
    SWAP_FIELDS = "SWAP_FIELDS"
    DATE_JITTER = "DATE_JITTER"



_DEFAULT_ISSUE_DISTRIBUTION = {
    DataQualityIssue.MISSING_VALUES: 1.0,
    DataQualityIssue.TYPOS_NOISE: 1.0,
    DataQualityIssue.OUTLIER_AMOUNT: 1.0,
    DataQualityIssue.DUPLICATE_ROWS: 1.0,
    DataQualityIssue.SWAP_FIELDS: 1.0,
    DataQualityIssue.DATE_JITTER: 1.0,
}


class DataQualityConfig(BaseModel):
    """Configuration for data quality injection.

    When ``enabled`` is ``True`` and no explicit ``issue_dist`` is supplied the
    configuration defaults to a uniform distribution across all supported issue
    types.
    """

class DataQualityConfig(BaseModel):
    """Configuration for data quality injection."""


    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    row_dirty_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    issue_dist: Mapping[DataQualityIssue, float] = Field(default_factory=dict)
    max_issues_per_row: int = Field(ge=1, default=2)
    missing_cols_whitelist: list[str] | None = None
    typos_cols_whitelist: list[str] | None = None

    @model_validator(mode="after")
    def _validate_dist(self) -> DataQualityConfig:

        source: Mapping[DataQualityIssue, float] | None = None
        if self.enabled:
            source = self.issue_dist or _DEFAULT_ISSUE_DISTRIBUTION
        elif self.issue_dist:
            source = self.issue_dist

        if source is not None:
            normalized = _normalize_dist({k.value: v for k, v in source.items()})

        if self.enabled:
            if not self.issue_dist:
                raise ConfigurationError("issue_dist must be provided when dirty data is enabled")
            normalized = _normalize_dist({k.value: v for k, v in self.issue_dist.items()})

            object.__setattr__(
                self,
                "issue_dist",
                {DataQualityIssue(k): v for k, v in normalized.items()},
            )
        return self


class ReferenceFitConfig(BaseModel):
    """Configuration for reference dataset profiling."""

    model_config = ConfigDict(frozen=True)

    dp_epsilon: float | None = Field(default=None, gt=0.0)
    fit_max_categories: int = Field(ge=5, default=10)
    fit_from_path: Path | None = None
    time_col: str = "event_time"

    @model_validator(mode="after")
    def _resolve_path(self) -> ReferenceFitConfig:
        if self.fit_from_path is not None:
            object.__setattr__(self, "fit_from_path", self.fit_from_path.expanduser().resolve())
        return self


class GeneratorConfig(BaseModel):
    """Top-level configuration for dataset generation."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    records: int = Field(ge=1)
    seed: int | None = Field(default=None, ge=0)
    start_date: date = Field(default_factory=date.today)
    days: int = Field(ge=1, default=7)
    age_dist: Mapping[str, float]
    channel_dist: Mapping[str, float] | None = None
    region_dist: Mapping[str, float] | None = None
    merchant_category_dist: Mapping[str, float] | None = None
    fraud_rate: float = Field(ge=0.0, le=1.0, default=0.02)
    fraud_type_dist: Mapping[str, float] = Field(

        default_factory=lambda: {
            FraudType.CARD_NOT_PRESENT.value: 0.22,
            FraudType.ACCOUNT_TAKEOVER.value: 0.18,
            FraudType.AUTHORIZED_PUSH_PAYMENT.value: 0.12,
            FraudType.CARD_PRESENT_CLONED.value: 0.1,
            FraudType.SKIMMING.value: 0.1,
            FraudType.SYNTHETIC_IDENTITY.value: 0.1,
            FraudType.MONEY_MULE.value: 0.08,
            FraudType.FRIENDLY_FRAUD.value: 0.05,
            FraudType.SOCIAL_ENGINEERING.value: 0.05,
        }

        default_factory=lambda: {FraudType.CARD_NOT_PRESENT.value: 1.0}

    )
    causal_fraud: bool = Field(alias="casual_fraud", default=False)
    causal_fraud_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    output: OutputOptions
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    amount_model: dict[str, float] | None = None
    time_model: dict[str, list[float]] | None = None
    reference_fit: ReferenceFitConfig | None = None
    synth_backend: str = Field(default="none")
    synth_calibrate_cols: list[str] = Field(default_factory=list)
    synth_condition_cols: list[str] = Field(default_factory=list)
    synth_fit_from: Path | None = None
    synth_max_rows: int | None = Field(default=None, ge=1)
    eval_synth: bool = False

    @field_validator("age_dist", mode="before")
    @classmethod
    def _normalize_age(
        cls, value: Mapping[AgeBand | str, float]
    ) -> Mapping[str, float]:
        return _normalize_enum_mapping(value, {band.value for band in AgeBand})

    @field_validator("channel_dist", mode="before")
    @classmethod
    def _normalize_channel(
        cls, value: Mapping[Channel | str, float] | None
    ) -> Mapping[str, float] | None:
        if value is None:
            return None
        return _normalize_enum_mapping(value, {channel.value for channel in Channel})

    @field_validator("region_dist", mode="before")
    @classmethod
    def _normalize_region(
        cls, value: Mapping[Region | str, float] | None
    ) -> Mapping[str, float] | None:
        if value is None:
            return None
        return _normalize_enum_mapping(value, {region.value for region in Region})

    @field_validator("merchant_category_dist", mode="before")
    @classmethod
    def _normalize_merchant(
        cls, value: Mapping[str, float] | None
    ) -> Mapping[str, float] | None:
        if value is None:
            return None
        return _normalize_dist(dict(value))

    @field_validator("fraud_type_dist", mode="before")
    @classmethod
    def _normalize_fraud(
        cls, value: Mapping[FraudType | str, float]
    ) -> Mapping[str, float]:
        return _normalize_enum_mapping(value, {fraud.value for fraud in FraudType})

    @model_validator(mode="after")
    def _validate_causal(self) -> GeneratorConfig:
        if self.causal_fraud_rate > self.fraud_rate + 1e-9:
            raise ConfigurationError("causal_fraud_rate must be <= fraud_rate")
        if self.amount_model is not None:
            if set(self.amount_model) != {"log_mean", "log_sigma"}:
                raise ConfigurationError("amount_model must define log_mean and log_sigma")
        if self.time_model is not None:
            hist = self.time_model.get("hour_hist")
            if hist is None or len(hist) != 24:
                raise ConfigurationError("time_model.hour_hist must contain 24 values")
            total = sum(hist)
            if total <= 0:
                raise ConfigurationError("time_model.hour_hist must sum to positive value")
            normalized = [float(x) / total for x in hist]
            object.__setattr__(self, "time_model", {"hour_hist": normalized})
        if self.synth_fit_from is not None:
            object.__setattr__(self, "synth_fit_from", self.synth_fit_from.expanduser().resolve())
        return self


def _normalize_enum_mapping(
    mapping: Mapping[AgeBand | Channel | Region | FraudType | str, float],
    allowed: set[str],
) -> Mapping[str, float]:
    coerced: dict[str, float] = {}
    for key, value in mapping.items():
        str_key = key.value if isinstance(key, Enum) else str(key)
        if str_key not in allowed:
            raise ConfigurationError(f"Invalid key '{str_key}' for distribution")
        coerced[str_key] = float(value)
    return _normalize_dist(coerced)


def _normalize_dist(dist: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(dist.values()))
    if total <= 0:
        raise ConfigurationError("Distribution total must be positive")
    return {k: float(v) / total for k, v in dist.items()}


def parse_generator_config(data: Mapping[str, Any]) -> GeneratorConfig:
    """Parse raw mapping into :class:`GeneratorConfig` with rich errors."""

    try:
        return GeneratorConfig.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - forwarded to caller
        raise ConfigurationError(str(exc)) from exc
