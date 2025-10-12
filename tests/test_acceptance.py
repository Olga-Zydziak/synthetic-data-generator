from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraudforge.config import GeneratorConfig
from fraudforge.generator import TransactionGenerator


def make_config(outdir: Path) -> GeneratorConfig:
    return GeneratorConfig.model_validate(
        {
            "records": 200,
            "seed": 123,
            "age_dist": {"A18_25": 0.25, "A26_35": 0.25, "A36_50": 0.25, "A50_PLUS": 0.25},
            "fraud_rate": 0.08,
            "fraud_type_dist": {
                "CARD_NOT_PRESENT": 0.5,
                "ACCOUNT_TAKEOVER": 0.2,
                "SKIMMING": 0.2,
                "AUTHORIZED_PUSH_PAYMENT": 0.1,
            },
            "causal_fraud": True,
            "causal_fraud_rate": 0.02,
            "output": {
                "format": "csv",
                "outdir": str(outdir),
                "chunk_size": 64,
            },
        }
    )


def test_generation_pipeline(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    generator = TransactionGenerator(cfg)
    metadata = generator.run()

    csv_path = tmp_path / "transactions.csv.gz"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert df.shape[0] == 200
    assert "is_casual_fraud" in df.columns
    assert metadata["counts"]["total_records"] == 200
    assert metadata["causal"]["causal_fraud_count"] > 0
    assert metadata["data_quality"]["dirty_rows"] == 0
    assert metadata["synth"]["backend"] == "none"
