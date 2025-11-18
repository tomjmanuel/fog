"""AWS-related helpers shared across the fog package."""
from __future__ import annotations

from typing import Any


def lazy_boto3() -> Any:
    """Import and return ``boto3`` lazily.

    The project primarily depends on ``s3fs`` for data access, so boto3 is only
    required when uploading or deleting objects directly via the AWS API. This
    helper centralizes the optional import so callers can defer the dependency
    until the functionality is needed.
    """

    try:
        import boto3  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("boto3 is required for this operation.") from exc
    return boto3


__all__ = ["lazy_boto3"]

