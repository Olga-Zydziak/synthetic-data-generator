from __future__ import annotations

import pandas as pd
import pytest

from fraudforge.exceptions import MissingExtraError
from fraudforge.synth.factory import create_synthesizer


def test_none_synthesizer_noop() -> None:
    synth, info = create_synthesizer("none", calibrate_cols=[], condition_cols=[])
    df = pd.DataFrame(
        {
            "transaction_id": ["a"],
            "is_fraud": [False],
            "fraud_type": [None],
            "is_causal_fraud": [False],
            "scenario": ["baseline"],
            "is_casual_fraud": [False],
        }
    )
    result = synth.calibrate_columns(df, [], [])
    assert result.equals(df)
    assert info.backend == "none"


def test_missing_extra_error() -> None:
    with pytest.raises(MissingExtraError):
        create_synthesizer("sdv", calibrate_cols=[], condition_cols=[])
