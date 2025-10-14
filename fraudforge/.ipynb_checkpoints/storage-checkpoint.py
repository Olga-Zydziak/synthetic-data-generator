"""Storage helpers for exporting generated artifacts."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .exceptions import WriterError

logger = logging.getLogger(__name__)

__all__ = ["BucketExporter"]


@dataclass(slots=True)
class BucketExporter:
    """Exports generated artifacts to local filesystem or Google Cloud Storage.
    
    Supports both local directories and GCS buckets with automatic detection.
    
    Attributes:
        target_dir: Local directory for staging files.
        gcs_path: Optional GCS destination (format: "bucket-name/prefix").
        gcs_client: Optional authenticated GCS client for uploads.
        
    Complexity:
        Local: O(k) for k files (shutil.copy2)
        GCS: O(k) for k files (streaming upload)
    """

    target_dir: Path
    gcs_path: str | None = None
    gcs_client: "storage.Client | None" = None  # type: ignore

    def _is_gcs_export(self) -> bool:
        """Check if GCS export is configured.
        
        Returns:
            bool: True if gcs_path is set and client is available.
            
        Complexity:
            Time: O(1); Memory: O(1).
        """
        return self.gcs_path is not None and self.gcs_client is not None

    def _upload_to_gcs(self, source: Path) -> None:
        """Upload a file to Google Cloud Storage.
        
        Args:
            source: Local file path to upload.
            
        Raises:
            WriterError: If GCS upload fails.
            
        Complexity:
            Time: O(n) where n is file size; Memory: O(1) streaming.
        """
        if self.gcs_client is None or self.gcs_path is None:
            raise WriterError("GCS client or path not configured")  # pragma: no cover

        try:
            # Parse bucket name and path
            parts = self.gcs_path.split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            
            # Construct destination path
            dest_path = f"{prefix}/{source.name}" if prefix else source.name
            
            # Upload file
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(dest_path)
            blob.upload_from_filename(str(source))
            
            logger.info(f"Uploaded {source.name} to gs://{bucket_name}/{dest_path}")
            
        except Exception as exc:
            raise WriterError(
                f"Failed to upload {source.name} to GCS: {exc}"
            ) from exc

    def export(self, *paths: Path) -> None:
        """Copy files to local target and optionally upload to GCS.
        
        Files are first copied to local target_dir (for staging/backup),
        then optionally uploaded to GCS if configured.
        
        Args:
            paths: Paths to export (must exist).
            
        Raises:
            WriterError: If filesystem or GCS operations fail.
            
        Complexity:
            Time: O(k) for k files; Memory: O(1) per file.
        """
        for source in paths:
            # Always copy to local staging area first
            destination = self.target_dir / source.name
            try:
                shutil.copy2(source, destination)
                logger.debug(f"Staged {source.name} to {destination}")
            except OSError as exc:
                raise WriterError(
                    f"Failed to stage {source} locally: {exc}"
                ) from exc
            
            # Upload to GCS if configured
            if self._is_gcs_export():
                self._upload_to_gcs(destination)