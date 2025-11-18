"""Helpers for uploading rendered assets to S3.

These functions are currently unused but provide the scaffolding for
future automation that pushes daily renders to a CloudFront-backed bucket.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .aws import lazy_boto3


def build_s3_key(prefix: str, filename: str) -> str:
    """Join ``prefix`` and ``filename`` into an S3 object key."""
    sanitized = prefix.strip("/")
    if sanitized:
        return f"{sanitized}/{filename}"
    return filename


def upload_file_to_s3(
    local_path: Path,
    bucket: str,
    key: str,
    *,
    client: Any | None = None,
    extra_args: Mapping[str, Any] | None = None,
) -> str:
    """Upload ``local_path`` to ``s3://bucket/key`` and return the URI."""
    boto3 = lazy_boto3()
    s3 = client or boto3.client("s3")
    s3.upload_file(
        str(local_path),
        bucket,
        key,
        ExtraArgs=dict(extra_args or {}),
    )
    return f"s3://{bucket}/{key}"


def upload_render_batch(
    files: Mapping[str, Path],
    bucket: str,
    *,
    prefix: str = "",
    client: Any | None = None,
    overwrite: bool = True,
    extra_args: Mapping[str, Any] | None = None,
) -> list[str]:
    """Upload multiple rendered assets to S3.

    Parameters
    ----------
    files:
        Mapping of descriptive labels to local file paths, typically the
        outputs of ``render_scene_to_file``.
    bucket:
        Destination S3 bucket.
    prefix:
        Optional key prefix (e.g. ``daily/2024-01-01``).
    overwrite:
        When ``False``, existing objects are preserved by namespacing each
        render under its label.
    """
    uris: list[str] = []
    for label, path in files.items():
        name = Path(path).name
        key = build_s3_key(prefix, name) if overwrite else build_s3_key(
            prefix, f"{label}/{name}"
        )
        uri = upload_file_to_s3(
            path,
            bucket,
            key,
            client=client,
            extra_args=extra_args,
        )
        uris.append(uri)
    return uris


__all__ = [
    "build_s3_key",
    "upload_file_to_s3",
    "upload_render_batch",
]
