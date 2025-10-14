from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraudforge.config import GeneratorConfig
from fraudforge.generator import TransactionGenerator


def test_dirty_json_output(tmp_path: Path) -> None:
    cfg = GeneratorConfig.model_validate(
        {
            "records": 60,
            "seed": 42,
            "age_dist": {"A18_25": 0.5, "A26_35": 0.5},
            "fraud_rate": 0.05,
            "fraud_type_dist": {"CARD_NOT_PRESENT": 1.0},
            "output": {"format": "json", "outdir": str(tmp_path), "chunk_size": 20},
            "data_quality": {
                "enabled": True,
                "row_dirty_rate": 0.6,
                "issue_dist": {
                    "MISSING_VALUES": 0.5,
                    "TYPOS_NOISE": 0.5,
                },
            },
        }
    )
    metadata = TransactionGenerator(cfg).run()
    jsonl = tmp_path / "transactions.jsonl.gz"
    df = pd.read_json(jsonl, lines=True, compression="gzip")
    dirty_share = metadata["data_quality"]["dirty_share"]
    assert df[df["is_dirty"]].shape[0] >= 1
    assert dirty_share > 0
