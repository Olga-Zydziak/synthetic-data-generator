"""Parquet streaming writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


from ..storage import BucketExporter


from ..storage import BucketExporter


from .writer_base import BaseWriter


class ParquetWriter(BaseWriter):
    """Streams transactions to a Parquet file using pyarrow."""


    def __init__(self, outdir: Path, bucket: BucketExporter | None = None) -> None:
        """Initialize Parquet writer with output directory and optional bucket.
        
        Args:
            outdir: Directory for output files.
            bucket: Optional cloud storage exporter.
            
        Complexity:
            Time: O(1); Memory: O(1).
        """
        super().__init__(outdir, "transactions.parquet", bucket=bucket)
        self._writer: pq.ParquetWriter | None = None

    

    def write(self, df: pd.DataFrame) -> None:
        chunk = df.copy()
        chunk.loc[:, "event_time"] = chunk["event_time"].apply(
            lambda ts: pd.Timestamp(ts).to_pydatetime()
        )
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)

    def finalize(self, metadata: dict[str, object]) -> None:
        if self._writer is not None:
            self._writer.close()
        super().finalize(metadata)
