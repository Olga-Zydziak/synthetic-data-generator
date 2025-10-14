"""Core transaction generation orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .adapters import CSVWriter, JSONWriter, ParquetWriter
from .config import GeneratorConfig
from .dq import DefaultDirtyInjector
from .exceptions import GenerationError
from .fit import ConfigCalibrator, ReferenceProfiler
from .metadata import MetadataCollector
from .ports import Synthesizer, Writer
from .scenarios import BaselineFraudScenario, CausalColliderScenario, CausalSimpsonScenario
from .scenarios.base import BaseScenario, ScenarioTargets
from .synth import create_synthesizer


@dataclass(slots=True)
class _ScenarioState:
    """Internal tracking structure for scenario progress."""

    name: str
    scenario: BaseScenario
    remaining: int
    rng: np.random.Generator


class TransactionGenerator:
    """Generates synthetic transactions according to configuration."""

    def __init__(self, cfg: GeneratorConfig) -> None:
        self._cfg = cfg
        self._fit_metadata: dict[str, Any] | None = None

    def run(self) -> dict[str, Any]:
        """Generate the dataset and return metadata.

        Returns:
            dict[str, object]: Finalized metadata payload.

        Complexity:
            Time: O(n); Memory: O(1) beyond chunk buffers.
        """

        cfg = self._maybe_calibrate(self._cfg)
        writer = self._build_writer(cfg)
        synthesizer, synth_info = create_synthesizer(
            cfg.synth_backend,
            calibrate_cols=cfg.synth_calibrate_cols,
            condition_cols=cfg.synth_condition_cols,
            dp_epsilon=cfg.reference_fit.dp_epsilon if cfg.reference_fit else None,
        )

        metadata = MetadataCollector(cfg)
        if self._fit_metadata is not None:
            metadata.set_fit_profile(self._fit_metadata)
        if cfg.causal_fraud:
            metadata.register_causal_description(
                CausalSimpsonScenario.name,
                CausalSimpsonScenario.DESCRIPTION,
            )
            metadata.register_causal_description(
                CausalColliderScenario.name,
                CausalColliderScenario.DESCRIPTION,
            )

        total_records = cfg.records
        fraud_total = int(round(cfg.records * cfg.fraud_rate))
        causal_total = int(round(cfg.records * cfg.causal_fraud_rate)) if cfg.causal_fraud else 0
        baseline_rows = max(total_records - causal_total, 0)
        baseline_fraud = max(fraud_total - causal_total, 0)
        simpson_rows = causal_total // 2
        collider_rows = causal_total - simpson_rows

        seed_value = cfg.seed if cfg.seed is not None else 0
        seed_sequence = np.random.SeedSequence(seed_value)
        scenario_count = 1 + (2 if cfg.causal_fraud else 0)
        subsequences = seed_sequence.spawn(scenario_count + 1)

        scenario_entries: list[_ScenarioState] = []
        baseline_scenario = BaselineFraudScenario(
            ScenarioTargets(total_rows=baseline_rows, fraud_rows=baseline_fraud)
        )
        scenario_entries.append(
            _ScenarioState(
                name=BaselineFraudScenario.name,
                scenario=baseline_scenario,
                remaining=baseline_rows,
                rng=np.random.Generator(np.random.PCG64(subsequences[0])),
            )
        )

        if cfg.causal_fraud and causal_total > 0:
            simpson_scenario = CausalSimpsonScenario(
                ScenarioTargets(
                    total_rows=simpson_rows,
                    fraud_rows=simpson_rows,
                    causal_rows=simpson_rows,
                )
            )
            collider_scenario = CausalColliderScenario(
                ScenarioTargets(
                    total_rows=collider_rows,
                    fraud_rows=collider_rows,
                    causal_rows=collider_rows,
                )
            )
            scenario_entries.append(
                _ScenarioState(
                    name=CausalSimpsonScenario.name,
                    scenario=simpson_scenario,
                    remaining=simpson_rows,
                    rng=np.random.Generator(np.random.PCG64(subsequences[1])),
                )
            )
            scenario_entries.append(
                _ScenarioState(
                    name=CausalColliderScenario.name,
                    scenario=collider_scenario,
                    remaining=collider_rows,
                    rng=np.random.Generator(np.random.PCG64(subsequences[2])),
                )
            )

        injector = DefaultDirtyInjector(cfg.data_quality)
        injector_rng = np.random.Generator(np.random.PCG64(subsequences[-1]))

        generated = 0
        while generated < total_records:
            chunk_size = min(cfg.output.chunk_size, total_records - generated)
            chunk_frames: list[pd.DataFrame] = []
            produced = 0
            for entry in scenario_entries:
                if produced >= chunk_size:
                    break
                if entry.remaining <= 0:
                    continue
                take = min(entry.remaining, chunk_size - produced)
                df_chunk = entry.scenario.generate(take, entry.rng, cfg)
                entry.remaining -= take
                produced += df_chunk.shape[0]
                chunk_frames.append(df_chunk)
            if not chunk_frames:
                break
            chunk_df = pd.concat(chunk_frames, ignore_index=True)
            if cfg.synth_calibrate_cols:
                chunk_df = self._apply_synth_calibration(
                    synthesizer,
                    chunk_df,
                    cfg.synth_calibrate_cols,
                    cfg.synth_condition_cols,
                )
            issues: Counter[str]
            if cfg.data_quality.enabled:
                chunk_df, issues = injector.apply(chunk_df, injector_rng)
            else:
                issues = Counter[str]()
            metadata.update(chunk_df, dirty_issues=issues)
            writer.write(chunk_df)
            generated += chunk_df.shape[0]

        metadata.set_synth_info(synth_info.to_metadata())
        output_metadata = metadata.finalize()
        writer.finalize(output_metadata)
        return output_metadata

    def _apply_synth_calibration(
        self,
        synthesizer: Synthesizer,
        df: pd.DataFrame,
        calibrate_cols: list[str],
        condition_cols: list[str],
    ) -> pd.DataFrame:
        """Apply synthesizer calibration to selected columns.

        Complexity:
            Time: O(n); Memory: O(n) due to grouping copies.
        """

        if not condition_cols:
            return synthesizer.calibrate_columns(
                df,
                calibrate_cols,
                [
                    "transaction_id",
                    "is_fraud",
                    "fraud_type",
                    "is_causal_fraud",
                    "scenario",
                    "is_casual_fraud",
                ],
            )
        grouped_frames: list[pd.DataFrame] = []
        for _, group in df.groupby(condition_cols, dropna=False):
            grouped_frames.append(
                synthesizer.calibrate_columns(
                    group,
                    calibrate_cols,
                    [
                        "transaction_id",
                        "is_fraud",
                        "fraud_type",
                        "is_causal_fraud",
                        "scenario",
                        "is_casual_fraud",
                    ],
                )
            )
        return pd.concat(grouped_frames, ignore_index=True)

    def _build_writer(self, cfg: GeneratorConfig) -> Writer:
        """Instantiate the appropriate streaming writer.

        Complexity:
            Time: O(1); Memory: O(1).
        """

        outdir = cfg.output.outdir
        bucket = cfg.output.bucket.exporter() if cfg.output.bucket is not None else None
        if cfg.output.format == "csv":
            return CSVWriter(outdir, bucket=bucket)
        if cfg.output.format == "json":
            return JSONWriter(outdir, bucket=bucket)
        if cfg.output.format == "parquet":
            return ParquetWriter(outdir, bucket=bucket)
        raise GenerationError(f"Unsupported output format {cfg.output.format}")

    def _maybe_calibrate(self, cfg: GeneratorConfig) -> GeneratorConfig:
        """Calibrate configuration from reference dataset when requested.

        Complexity:
            Time: O(n); Memory: O(n) to load the reference dataframe.
        """

        if cfg.reference_fit is None or cfg.reference_fit.fit_from_path is None:
            return cfg
        path = cfg.reference_fit.fit_from_path
        if not path.exists():
            raise GenerationError(f"Reference dataset not found at {path}")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix in {".csv", ".gz"}:
            df = pd.read_csv(path)
        else:
            raise GenerationError("Unsupported reference dataset format")
        profiler = ReferenceProfiler()
        profile = profiler.fit(df, cfg.reference_fit)
        calibrator = ConfigCalibrator()
        calibrated_cfg = calibrator.calibrate(profile, cfg)
        self._fit_metadata = {
            "age_dist": profile.age_dist,
            "channel_dist": profile.channel_dist,
            "region_dist": profile.region_dist,
            "merchant_category_dist": profile.merchant_category_dist,
            "fraud_rate": profile.fraud_rate,
            "fraud_type_dist": profile.fraud_type_dist,
            "amount_log_mean": profile.amount_log_mean,
            "amount_log_sigma": profile.amount_log_sigma,
            "hour_hist": profile.hour_hist,
        }
        return calibrated_cfg
