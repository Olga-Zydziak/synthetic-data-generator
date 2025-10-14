"""Storage helpers for exporting generated artifacts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .exceptions import WriterError

__all__ = ["BucketExporter"]


@dataclass(slots=True)
class BucketExporter:
    """Copies generated artifacts into a resolved bucket directory."""

    target_dir: Path

    def export(self, *paths: Path) -> None:
        """Copy files into the bucket destination.

        Args:
            paths: Paths to copy into the bucket directory.

        Raises:
            WriterError: If a filesystem error occurs during copying.

        Complexity:
            Time: O(k) for ``k`` files; Memory: O(1).
        """

        for source in paths:
            destination = self.target_dir / source.name
            try:
                shutil.copy2(source, destination)
            except OSError as exc:  # pragma: no cover - propagated to caller
                raise WriterError(f"Failed to export {source} to bucket: {exc}") from exc
