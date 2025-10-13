# fraudforge

fraudforge is a production-focused synthetic data generator for banking transaction datasets. The
package produces CSV, JSON Lines, or Parquet streams, applies optional dirty data issues, injects
causal fraud scenarios, and records comprehensive lineage metadata.

## Features

- Deterministic generation driven by NumPy's `PCG64` RNG.
- Configurable fraud type composition, channel/region distributions, and amount models.
- Optional causal-only fraud scenarios (Simpson's paradox and collider bias) with clear metadata
documentation.
- Dirty data injector for missing values, typos, outliers, duplicates, swaps, and timestamp jitter.
- Streaming writers for gzipped CSV, gzipped JSON Lines, and Parquet.
- Reference dataset profiler with optional differential privacy noise to bootstrap configuration.
- Pluggable synthesizer adapters with lazy optional dependencies.
- Typer-powered CLI with JSON metadata output.

## Quickstart

```bash
pip install -e .
fraudforge generate \
  --records 1000 \
  --age-dist "A18_25:0.3,A26_35:0.3,A36_50:0.25,A50_PLUS:0.15" \

  --fraud-type-dist "CARD_NOT_PRESENT:0.5,ACCOUNT_TAKEOVER:0.2,CARD_PRESENT_CLONED:0.3" \

  --fraud-type-dist "CARD_NOT_PRESENT:0.7,ACCOUNT_TAKEOVER:0.3" \

  --fraud-rate 0.03 \
  --causal-fraud \
  --causal-fraud-rate 0.01 \
  --output-format csv \
  --outdir ./out
```

The command creates `transactions.csv.gz` and `metadata.json` under `./out` and prints metadata to
stdout in JSON form.


## Configuration reference

- **Distributions** (`*_dist` fields) accept comma-separated `KEY:WEIGHT` pairs on the CLI and
  dictionaries in Python. Keys must match the enum values below; weights are automatically
  normalized to sum to 1.0.
- **Fraud types**: `CARD_NOT_PRESENT`, `ACCOUNT_TAKEOVER`, `SKIMMING`,
  `AUTHORIZED_PUSH_PAYMENT`, `CARD_PRESENT_CLONED`, `SYNTHETIC_IDENTITY`, `FRIENDLY_FRAUD`,
  `MONEY_MULE`, `SOCIAL_ENGINEERING`.
- **Causal scenarios** (toggle with `causal_fraud`): Simpson's paradox on region×channel and a
  collider-bias review process; both mark `is_causal_fraud` and are summarized in metadata.
- **Dirty data**: set `data_quality.enabled=True` (or `--dirty`). If no `issue_dist` is supplied the
  generator defaults to an even mix of all supported dirty-data injectors. Provide
  `ISSUE:PROB,...` to bias toward specific problems.
- **Synth calibration**: optional plugins (`--synth`) can recalibrate specific columns without
  touching labels; list columns with `--synth-calibrate-cols` and optionally preserve cohorts with
  `--synth-condition-cols`.

## Using in JupyterLab

1. Install the package into the same Python environment that backs your JupyterLab kernel:

   ```bash
   pip install -e .
   ```

2. Start JupyterLab and create a new Python 3.11+ notebook. In a fresh cell configure and run the
   generator programmatically:

   ```python
   from pathlib import Path

   from fraudforge.config import DataQualityConfig, GeneratorConfig, OutputOptions
   from fraudforge.generator import TransactionGenerator

   cfg = GeneratorConfig(
       records=5_000,
       fraud_rate=0.05,
       fraud_type_dist={
           "CARD_NOT_PRESENT": 0.4,
           "ACCOUNT_TAKEOVER": 0.2,
           "SYNTHETIC_IDENTITY": 0.2,
           "SOCIAL_ENGINEERING": 0.2,
       },
       age_dist={
           "A18_25": 0.3,
           "A26_35": 0.3,
           "A36_50": 0.25,
           "A50_PLUS": 0.15,
       },
       causal_fraud=True,
       causal_fraud_rate=0.02,
       output=OutputOptions(format="parquet", outdir=Path("./notebook_out"), chunk_size=10_000),
       data_quality=DataQualityConfig(enabled=True, row_dirty_rate=0.05),
   )

   metadata = TransactionGenerator(cfg).run()
   metadata
   ```

   Running the cell streams the dataset to `./notebook_out` inside your working directory and
   returns the finalized metadata dictionary for quick inspection.

3. Load any of the generated artifacts in subsequent cells—e.g. read the Parquet output for ad-hoc
   exploration:

   ```python
   import pandas as pd

   transactions = pd.read_parquet("./notebook_out/transactions.parquet")
   transactions.head()
   ```

The generator is deterministic for a fixed seed, so rerunning the notebook cell with the same
configuration will produce identical outputs. Adjust the configuration inside the notebook to
experiment with richer fraud type mixes (regular and causal), causal scenarios, dirty data options,
or synthesizer integrations.



## Development

Install development dependencies and run the quality checks:

```bash
pip install -r requirements-dev.txt
ruff check .
mypy --strict .
pytest
```

## License

MIT License.
