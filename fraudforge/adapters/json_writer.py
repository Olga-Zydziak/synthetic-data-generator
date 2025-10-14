"""JSON Lines streaming writer."""

from __future__ import annotations

import gzip
from pathlib import Path

import orjson
import pandas as pd

from ..storage import BucketExporter
from .writer_base import BaseWriter, _json_default


class JSONWriter(BaseWriter):
    """Streams transactions into gzipped JSON Lines."""

    def __init__(self, outdir: Path, bucket: BucketExporter | None = None) -> None:
        super().__init__(outdir, "transactions.jsonl.gz", bucket=bucket)
        self._handle = gzip.open(self.path, "wb")

    def write(self, df: pd.DataFrame) -> None:
        chunk = df.copy()
        chunk.loc[:, "event_time"] = chunk["event_time"].apply(
            lambda ts: pd.Timestamp(ts).isoformat()
        )
        for record in chunk.to_dict(orient="records"):
            payload = orjson.dumps(record, default=_json_default)
            self._handle.write(payload + b"\n")

    def finalize(self, metadata: dict[str, object]) -> None:
        self._handle.close()
        super().finalize(metadata)
