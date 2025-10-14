"""Smoke tests ensuring key public symbols are importable."""

from __future__ import annotations


def test_bucket_options_import_from_config() -> None:
    """`BucketOptions` should be exposed from the configuration module."""

    from fraudforge.config import BucketOptions  # noqa: PLC0415

    assert BucketOptions.__name__ == "BucketOptions"


def test_bucket_options_import_from_package_root() -> None:
    """`BucketOptions` should be exposed from the package root for convenience."""

    from fraudforge import BucketOptions  # noqa: PLC0415

    assert BucketOptions.__name__ == "BucketOptions"
