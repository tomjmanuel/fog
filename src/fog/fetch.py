"""Minimal helpers for downloading GOES-18 ABI channel 02 scenes."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

try:  # optional dependency at runtime
    import s3fs  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled lazily at runtime
    s3fs = None

BUCKET = "noaa-goes18"
PRODUCT = "ABI-L1b-RadC"
REGION = "M6"
CHANNEL = "C02"


def _fs():
    if s3fs is None:  # pragma: no cover - resolved only at runtime
        raise RuntimeError("s3fs is required to fetch data but is not installed.")
    return s3fs.S3FileSystem(anon=True)


def _scene_prefix(scene_time: datetime) -> str:
    scene = scene_time.astimezone(timezone.utc)
    return (
        f"{BUCKET}/{PRODUCT}/{scene:%Y/%j/%H}/"
        f"OR_{PRODUCT}-{REGION}{CHANNEL}_G18_"
    )


def _select_granule(scene_time: datetime) -> str:
    prefix = _scene_prefix(scene_time)
    fs = _fs()
    keys = sorted(fs.glob(f"{prefix}*"))
    if not keys:
        raise FileNotFoundError(
            f"No {CHANNEL} granules found for {scene_time.isoformat()}"
        )
    return keys[0]


def download_channel_02(scene_time: datetime, output_dir: Path) -> Path:
    """Download the C02 granule nearest ``scene_time`` and save it locally."""

    key = _select_granule(scene_time)
    fs = _fs()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = scene_time.astimezone(timezone.utc)
    filename = output_dir / f"goes18_{scene:%Y%m%dT%H%M%S}_{CHANNEL}.nc"
    with fs.open(f"s3://{key}", mode="rb") as remote:
        ds = xr.open_dataset(remote, engine="h5netcdf").load()
    ds.to_netcdf(filename)
    return filename


__all__ = ["download_channel_02"]
