"""Shared writer utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd

from ..exceptions import WriterError

from ..storage import BucketExporter


def _json_default(value: object) -> str | list[object]:
    """Convert unsupported JSON types to serializable values.

    Args:
        value: Object provided by :mod:`json` or :mod:`orjson` default handler.

    Returns:
        A string or list representation that is JSON serializable.

    Raises:
        TypeError: If the value cannot be converted.

    Complexity:
        Time: O(k) for set conversion; Memory: O(k) for produced list.
    """

    if isinstance(value, pd.Timestamp):
        return str(value.isoformat())
    if isinstance(value, set | frozenset):
        return [cast(object, element) for element in value]
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


class BaseWriter:
    """Common functionality for streaming writers."""


    def __init__(
        self,
        outdir: Path,
        filename: str,
        bucket: BucketExporter | None = None,
    ) -> None:


    


        try:
            outdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem errors
            raise WriterError(f"Failed to create output directory: {exc}") from exc
        self._outdir = outdir.resolve()
        self._path = (self._outdir / filename).resolve()
        self._metadata_path = (self._outdir / "metadata.json").resolve()

        self._bucket = bucket



        self._bucket = bucket



    @property
    def path(self) -> Path:
        return self._path

    def finalize(self, metadata: dict[str, object]) -> None:
        try:
            with self._metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, default=_json_default)
        except OSError as exc:  # pragma: no cover - filesystem errors
            raise WriterError(f"Failed to write metadata: {exc}") from exc

        if self._bucket is not None:
            self._bucket.export(self._path, self._metadata_path)


        if self._bucket is not None:
            self._bucket.export(self._path, self._metadata_path)


