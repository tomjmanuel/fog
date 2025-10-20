"""Tiny helper to grab GOES-18 channel 02 scenes."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import xarray as xr

try:  # optional dependency installed at runtime
    import s3fs  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - surfaced when function is called
    s3fs = None


ABI_BUCKET = "noaa-goes18"
ABI_PRODUCT = "ABI-L1b-RadC"
ABI_REGION = "M6"
ABI_CHANNEL = "C02"


def _filesystem():
    if s3fs is None:  # pragma: no cover - callers see clear error message
        raise RuntimeError("s3fs is required to fetch GOES data but is not installed")
    return s3fs.S3FileSystem(anon=True)


def _object_prefix(scene_time: datetime) -> str:
    return f"{ABI_PRODUCT}/{scene_time:%Y/%j/%H}/OR_{ABI_PRODUCT}-{ABI_REGION}"


def _list_channel_objects(scene_time: datetime) -> list[str]:
    fs = _filesystem()
    pattern = f"{ABI_BUCKET}/{_object_prefix(scene_time)}*{ABI_CHANNEL}_*.nc"
    return sorted(fs.glob(pattern))


def download_channel_02(scene_time: datetime, output_dir: Path) -> Path:
    """Download the first GOES-18 C02 granule found for ``scene_time``."""
    keys = _list_channel_objects(scene_time)
    if not keys:
        raise FileNotFoundError(
            f"No {ABI_CHANNEL} granules found for {scene_time.isoformat()}"
        )
    key = keys[0]
    uri = f"s3://{key}" if not key.startswith("s3://") else key
    ds = xr.open_dataset(_filesystem().open(uri, mode="rb"), engine="h5netcdf").load()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"goes18_{ABI_CHANNEL}_{scene_time:%Y%m%dT%H%M%S}.nc"
    ds.to_netcdf(path)
    return path


__all__ = ["download_channel_02"]
