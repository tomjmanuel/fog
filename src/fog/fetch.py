"""Data access utilities for GOES-18 ABI datasets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Tuple, Iterable, Dict

import numpy as np
import xarray as xr

try:  # optional heavy dependency at runtime
    import s3fs  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled lazily at runtime
    s3fs = None

from .config import GOESConfig
from .projection import project_xy_to_lonlat

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SectorDefinition:
    """Geographic bounds for a processing sector."""

    north: float
    south: float
    east: float
    west: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def intersects(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return (
            (lon >= self.west)
            & (lon <= self.east)
            & (lat >= self.south)
            & (lat <= self.north)
        )


SAN_FRANCISCO_SECTOR = SectorDefinition(
    north=38.390362,
    south=37.268569,
    east=-121.731963,
    west=-122.984661,
)


def list_scene_objects(
    config: GOESConfig,
    scene_time: datetime,
    product: str,
    *,
    channel: str | None = None,
) -> list[str]:
    start, end = config.valid_time_window(scene_time)
    prefix = config.object_key_prefix(scene_time, product=product)
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": None})
    candidates = fs.glob(f"{config.bucket}/{prefix}*")
    filtered = []
    for key in candidates:
        try:
            parts = key.split("_")
            # Extract timestamp: s20231821901187 -> first 14 chars
            timestamp = parts[3][1:14]
            scene = datetime.strptime(timestamp, "%Y%j%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except Exception:  # pragma: no cover - format errors are rare
            LOGGER.debug("Skipping unexpected key format: %s", key)
            continue
        if start <= scene <= end:
            if channel is None or f"{channel}_" in key:
                filtered.append(key)
    return sorted(filtered)


def open_dataset(
    config: GOESConfig,
    scene_time: datetime,
    product: str,
    *,
    channel: str | None = None,
) -> xr.Dataset:
    keys = list_scene_objects(config, scene_time, product, channel=channel)
    if not keys:
        raise FileNotFoundError(
            f"No {product} objects found for {scene_time.isoformat()}"
        )
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": None})
    # Select the single granule whose timestamp is closest to scene_time
    # to avoid stacking multiple scans.
    scene_time_utc = (
        scene_time.replace(tzinfo=timezone.utc)
        if scene_time.tzinfo is None
        else scene_time.astimezone(timezone.utc)
    )
    candidates: list[tuple[datetime, str]] = []
    for key in keys:
        parts = key.split("_")
        try:
            timestamp = parts[3][1:14]
            t = datetime.strptime(timestamp, "%Y%j%H%M%S").replace(
                tzinfo=timezone.utc
            )
            candidates.append((t, key))
        except Exception:
            continue
    if not candidates:
        # Fallback: use the first key
        chosen_key = keys[0]
    else:
        chosen_t, chosen_key = min(
            candidates,
            key=lambda tk: abs((tk[0] - scene_time_utc).total_seconds()),
        )
    uri = (
        f"s3://{chosen_key}"
        if not chosen_key.startswith("s3://")
        else chosen_key
    )
    delta_s = (
        abs((chosen_t - scene_time_utc).total_seconds()) if candidates else 0.0
    )
    LOGGER.info(
        "Opening nearest granule for %s: %s (Δ=%.1fs)",
        product,
        chosen_key,
        delta_s,
    )
    open_kwargs = {"engine": "h5netcdf"}
    ds = xr.open_dataset(fs.open(uri, mode="rb"), **open_kwargs)
    # Keep dataset lazy - don't load until after subsetting
    return ds


def fetch_ABI_L1b(
    channel: str,
    scene_time: datetime,
    sector: SectorDefinition,
    config: GOESConfig,
) -> xr.Dataset:
    product = config.product
    dataset = open_dataset(
        config,
        scene_time,
        product=product,
        channel=channel,
    )
    # Subset while lazy, then load only the subset
    subset = subset_sector(dataset, sector)
    return subset.load()


def subset_sector(dataset: xr.Dataset, sector: SectorDefinition) -> xr.Dataset:
    x = dataset.coords.get("x")
    y = dataset.coords.get("y")
    if x is None or y is None:
        raise ValueError("Dataset missing GOES projection coordinates")
    lon2d, lat2d = project_xy_to_lonlat(x, y, dataset=dataset)

    lon_min = float(np.nanmin(lon2d))
    lon_max = float(np.nanmax(lon2d))
    lat_min = float(np.nanmin(lat2d))
    lat_max = float(np.nanmax(lat2d))
    LOGGER.info(
        "Scene lon/lat bounds: lon[%.2f, %.2f], lat[%.2f, %.2f]; "
        "sector west=%.2f east=%.2f south=%.2f north=%.2f",
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        sector.west,
        sector.east,
        sector.south,
        sector.north,
    )

    ds = dataset.assign_coords(
        {
            "lon": (("y", "x"), lon2d),
            "lat": (("y", "x"), lat2d),
        }
    )
    lon_mask = (lon2d >= sector.west) & (lon2d <= sector.east)
    lat_mask = (lat2d >= sector.south) & (lat2d <= sector.north)
    mask = lon_mask & lat_mask
    if mask.ndim == 2:
        if not np.any(mask):
            # If nothing intersects, return the original dataset instead of
            # an empty slice
            LOGGER.warning(
                "Sector subsetting found no overlap; returning full dataset"
            )
            return ds
        valid_rows = np.any(mask, axis=1)
        valid_cols = np.any(mask, axis=0)
        return ds.isel(y=valid_rows, x=valid_cols)
    return ds


def download_channels(
    scene_time: datetime,
    output_dir: Path,
    *,
    config: GOESConfig | None = None,
) -> Dict[str, str]:
    """Download specified ABI L1b channels for a scene and save to NetCDF.
    Returns a mapping of channel -> saved file path.
    """
    cfg = config or GOESConfig()
    channel_list = ["C02"]
    sector = SAN_FRANCISCO_SECTOR
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}
    for channel in channel_list:
        ds = fetch_ABI_L1b(channel, scene_time, sector, cfg)
        fname = (
            f"goes18_{cfg.product}_{channel}_"
            f"{scene_time:%Y%m%dT%H%M%S}_SF.nc"
        )
        path = output_dir / fname
        # Ensure coordinate variables are saved as float to avoid integer
        # encoding warnings from xarray/netCDF.
        coord_float32: Dict[str, Dict[str, str]] = {}
        for v in ("x", "y", "lon", "lat"):
            if v in ds.variables:
                coord_float32[v] = {"dtype": "float32"}
                # Clear any inherited encodings that might force integer dtypes
                try:
                    ds[v].encoding.clear()
                except Exception:
                    pass
        ds.to_netcdf(path, encoding=coord_float32)
        saved[channel] = str(path)
    return saved


__all__ = [
    "SectorDefinition",
    "SAN_FRANCISCO_SECTOR",
    "fetch_ABI_L1b",
    "download_channels",
    "subset_sector",
]
