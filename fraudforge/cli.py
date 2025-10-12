"""Command line interface for fraudforge."""

from __future__ import annotations

import json
from pathlib import Path

import orjson
import pandas as pd
import typer
from rich.console import Console

from .adapters.writer_base import _json_default
from .config import GeneratorConfig, ReferenceFitConfig
from .exceptions import GenerationError
from .fit import ReferenceProfiler
from .generator import TransactionGenerator

app = typer.Typer(help="Synthetic fraud transaction data generator.")
console = Console()


def _parse_mapping(argument: str | None) -> dict[str, float] | None:
    if not argument:
        return None
    mapping: dict[str, float] = {}
    for chunk in argument.split(","):
        if not chunk:
            continue
        key, value = chunk.split(":", 1)
        mapping[key.strip()] = float(value)
    return mapping if mapping else None


@app.command()
def generate(
    records: int = typer.Option(..., help="Number of records to generate."),
    seed: int | None = typer.Option(None, help="Random seed."),
    age_dist: str = typer.Option(
        ..., help="Age band distribution e.g. A18_25:0.2,..."
    ),
    channel_dist: str | None = typer.Option(None, help="Channel distribution."),
    region_dist: str | None = typer.Option(None, help="Region distribution."),
    merchant_category_dist: str | None = typer.Option(
        None, help="Merchant category distribution."
    ),
    fraud_rate: float = typer.Option(0.05, help="Overall fraud rate."),
    fraud_type_dist: str = typer.Option(
        "CARD_NOT_PRESENT:1.0", help="Fraud type distribution."
    ),
    causal_fraud: bool = typer.Option(
        False,
        "--causal-fraud/--no-causal-fraud",
        help="Enable causal fraud.",
    ),
    casual_fraud: bool = typer.Option(
        False,
        "--casual-fraud/--no-casual-fraud",
        help="Alias for causal.",
    ),
    causal_fraud_rate: float = typer.Option(0.01, help="Share of causal fraud."),
    output_format: str = typer.Option(
        "csv", help="Output format (csv,json,parquet)."
    ),
    outdir: Path = typer.Option(..., help="Output directory."),
    chunk_size: int = typer.Option(1000, help="Chunk size for streaming."),
    dirty: bool = typer.Option(
        False,
        "--dirty/--no-dirty",
        help="Enable dirty data injection.",
    ),
    dirty_rate: float = typer.Option(0.0, help="Row dirty rate."),
    dirty_issue_dist: str | None = typer.Option(
        None, help="Issue distribution ISSUE:PROB,..."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help="Optional config file."
    ),
    synth_backend: str = typer.Option("none", help="Synthesizer backend."),
    synth_calibrate_cols: str | None = typer.Option(
        None, help="Columns to calibrate via synth."
    ),
    synth_condition_cols: str | None = typer.Option(
        None, help="Condition columns for synth."
    ),
) -> None:
    """Generate synthetic transactions and metadata.

    Complexity:
        Time: O(n); Memory: O(1) beyond chunk buffers.
    """

    data: dict[str, object] = {}
    if config_path:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

    age_mapping = _parse_mapping(age_dist)
    fraud_mapping = _parse_mapping(fraud_type_dist)
    if age_mapping is None or fraud_mapping is None:
        raise typer.BadParameter("Distributions must contain at least one entry.")

    data.update(
        {
            "records": records,
            "seed": seed,
            "age_dist": age_mapping,
            "fraud_rate": fraud_rate,
            "fraud_type_dist": fraud_mapping,
            "causal_fraud": causal_fraud or casual_fraud,
            "causal_fraud_rate": causal_fraud_rate,
            "output": {
                "format": output_format,
                "outdir": str(outdir),
                "chunk_size": chunk_size,
            },
            "synth_backend": synth_backend,
            "synth_calibrate_cols": (
                synth_calibrate_cols.split(",") if synth_calibrate_cols else []
            ),
            "synth_condition_cols": (
                synth_condition_cols.split(",") if synth_condition_cols else []
            ),
        }
    )

    for field_name, mapping in (
        ("channel_dist", _parse_mapping(channel_dist)),
        ("region_dist", _parse_mapping(region_dist)),
        ("merchant_category_dist", _parse_mapping(merchant_category_dist)),
    ):
        if mapping is not None:
            data[field_name] = mapping

    issue_mapping = _parse_mapping(dirty_issue_dist) or {}
    data["data_quality"] = {
        "enabled": dirty,
        "row_dirty_rate": dirty_rate,
        "issue_dist": issue_mapping,
    }

    cfg = GeneratorConfig.model_validate(data)
    generator = TransactionGenerator(cfg)
    try:
        metadata = generator.run()
    except GenerationError as exc:  # pragma: no cover - runtime safety
        console.print(f"[red]Generation error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    payload = orjson.dumps(metadata, default=_json_default).decode("utf-8")
    console.print_json(payload)


@app.command("fit-profile")
def fit_profile(
    fit_from: Path = typer.Option(..., help="Reference dataset path."),
    dp_epsilon: float | None = typer.Option(
        None, help="Differential privacy epsilon."
    ),
    fit_max_categories: int = typer.Option(
        10, help="Maximum categories for profiling."
    ),
    time_col: str = typer.Option("event_time", help="Event timestamp column."),
) -> None:
    """Profile existing dataset to derive configuration priors.

    Complexity:
        Time: O(n); Memory: O(n) due to dataframe load.
    """

    if fit_from.suffix == ".parquet":
        df = pd.read_parquet(fit_from)
    else:
        df = pd.read_csv(fit_from)
    cfg = ReferenceFitConfig(
        dp_epsilon=dp_epsilon,
        fit_max_categories=fit_max_categories,
        fit_from_path=fit_from,
        time_col=time_col,
    )
    profiler = ReferenceProfiler()
    profile = profiler.fit(df, cfg)
    payload = orjson.dumps(
        {
            "age_dist": profile.age_dist,
            "channel_dist": profile.channel_dist,
            "region_dist": profile.region_dist,
            "merchant_category_dist": profile.merchant_category_dist,
            "fraud_rate": profile.fraud_rate,
            "fraud_type_dist": profile.fraud_type_dist,
            "amount_log_mean": profile.amount_log_mean,
            "amount_log_sigma": profile.amount_log_sigma,
            "hour_hist": profile.hour_hist,
        },
        default=_json_default,
    ).decode("utf-8")
    console.print_json(payload)


__all__ = ["app"]
