from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from fraudforge.cli import app


def test_cli_generate_json(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "--records",
            "50",
            "--age-dist",
            "A18_25:0.4,A26_35:0.3,A36_50:0.2,A50_PLUS:0.1",
            "--fraud-type-dist",
            "CARD_NOT_PRESENT:0.6,ACCOUNT_TAKEOVER:0.4",
            "--fraud-rate",
            "0.1",
            "--casual-fraud",
            "--causal-fraud-rate",
            "0.02",
            "--output-format",
            "json",
            "--outdir",
            str(tmp_path),
            "--chunk-size",
            "32",
        ],
    )
    assert result.exit_code == 0
    jsonl_path = tmp_path / "transactions.jsonl.gz"
    assert jsonl_path.exists()
    df = pd.read_json(jsonl_path, lines=True, compression="gzip")
    assert df.shape[0] == 50


def test_cli_dirty_toggle(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "--records",
            "30",
            "--age-dist",
            "A18_25:0.5,A26_35:0.5",
            "--fraud-type-dist",
            "CARD_NOT_PRESENT:1.0",
            "--fraud-rate",
            "0.05",
            "--dirty",
            "--dirty-rate",
            "0.5",
            "--dirty-issue-dist",
            "MISSING_VALUES:0.6,TYPOS_NOISE:0.4",
            "--output-format",
            "csv",
            "--outdir",
            str(tmp_path / "dirty"),
        ],
    )
    assert result.exit_code == 0
    csv_path = tmp_path / "dirty" / "transactions.csv.gz"
    assert csv_path.exists()
