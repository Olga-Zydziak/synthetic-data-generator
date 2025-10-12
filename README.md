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
  --fraud-type-dist "CARD_NOT_PRESENT:0.7,ACCOUNT_TAKEOVER:0.3" \
  --fraud-rate 0.03 \
  --causal-fraud \
  --causal-fraud-rate 0.01 \
  --output-format csv \
  --outdir ./out
```

The command creates `transactions.csv.gz` and `metadata.json` under `./out` and prints metadata to
stdout in JSON form.

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
