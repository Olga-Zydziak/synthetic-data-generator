"""CSV streaming writer."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from ..storage import BucketExporter
from .writer_base import BaseWriter


class CSVWriter(BaseWriter):
    """Streams transactions into a gzipped CSV file."""

    def __init__(self, outdir: Path, bucket: BucketExporter | None = None) -> None:
        super().__init__(outdir, "transactions.csv.gz", bucket=bucket)
        self._handle = gzip.open(self.path, "wt", encoding="utf-8")
        self._wrote_header = False

    def write(self, df: pd.DataFrame) -> None:
        chunk = df.copy()
        chunk.loc[:, "event_time"] = chunk["event_time"].apply(
            lambda ts: pd.Timestamp(ts).isoformat()
        )
        chunk.loc[:, "dirty_issues"] = chunk["dirty_issues"].apply(json.dumps)
        chunk.to_csv(self._handle, index=False, header=not self._wrote_header)
        self._wrote_header = True

    def finalize(self, metadata: dict[str, object]) -> None:
        self._handle.close()
        super().finalize(metadata)
