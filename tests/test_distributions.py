from __future__ import annotations

from fraudforge.config import DataQualityConfig, DataQualityIssue, GeneratorConfig


def test_age_distribution_normalization() -> None:
    cfg = GeneratorConfig.model_validate(
        {
            "records": 10,
            "age_dist": {"A18_25": 1, "A26_35": 1, "A36_50": 1, "A50_PLUS": 1},
            "output": {"format": "csv", "outdir": "./tmp", "chunk_size": 10},
        }
    )
    assert abs(sum(cfg.age_dist.values()) - 1.0) < 1e-9


def test_data_quality_normalization() -> None:
    dq = DataQualityConfig(
        enabled=True,
        row_dirty_rate=0.5,
        issue_dist={DataQualityIssue.MISSING_VALUES: 2, DataQualityIssue.TYPOS_NOISE: 1},
    )
    assert abs(sum(dq.issue_dist.values()) - 1.0) < 1e-9



def test_data_quality_default_distribution_when_enabled() -> None:
    dq = DataQualityConfig(enabled=True, row_dirty_rate=0.1)
    assert abs(sum(dq.issue_dist.values()) - 1.0) < 1e-9
    assert set(dq.issue_dist.keys()) == set(DataQualityIssue)

