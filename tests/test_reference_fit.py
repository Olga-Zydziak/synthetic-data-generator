from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudforge.config import GeneratorConfig, ReferenceFitConfig
from fraudforge.fit.reference import ConfigCalibrator, ReferenceProfiler


def sample_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "event_time": pd.date_range("2024-01-01", periods=100, freq="H"),
            "age_band": rng.choice(["A18_25", "A26_35"], size=100),
            "channel": rng.choice(["APP", "WEB"], size=100),
            "region": rng.choice(["NORTH", "SOUTH"], size=100),
            "merchant_category": rng.choice(["grocery", "travel"], size=100),
            "is_fraud": rng.choice([True, False], size=100, p=[0.1, 0.9]),
            "fraud_type": rng.choice(["CARD_NOT_PRESENT", "SKIMMING"], size=100),
            "amount": rng.lognormal(mean=3.0, sigma=0.7, size=100),
        }
    )


def test_reference_profile_and_calibration(tmp_path: Path) -> None:
    df = sample_dataframe()
    cfg = ReferenceFitConfig(fit_max_categories=5, fit_from_path=tmp_path / "dummy.parquet")
    df.to_parquet(cfg.fit_from_path)
    profiler = ReferenceProfiler()
    profile = profiler.fit(df, cfg)
    calibrator = ConfigCalibrator()

    base_cfg = GeneratorConfig.model_validate(
        {
            "records": 10,
            "age_dist": {"A18_25": 0.5, "A26_35": 0.5},
            "output": {"format": "csv", "outdir": str(tmp_path / "out"), "chunk_size": 10},
        }
    )

    calibrated = calibrator.calibrate(profile, base_cfg)
    assert calibrated.time_model is not None
    assert "hour_hist" in calibrated.time_model
    assert len(profile.hour_hist) == 24
