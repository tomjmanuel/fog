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

LOGGER = logging.getLogger(__name__)

ABI_PROJECTION = {
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31414,
    "inverse_flattening": 298.2572221,
    "latitude_of_projection_origin": 0.0,
    "longitude_of_projection_origin": -137.0,
    "sweep_angle_axis": "x",
}


@dataclass(slots=True)
class SectorDefinition:
    """Geographic bounds for a processing sector."""

    west: float
    south: float
    east: float
    north: float

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
    west=-123.0,
    south=36.5,
    east=-121.0,
    north=38.2,
)


@lru_cache(maxsize=1)
def _fs(config: GOESConfig):
    if s3fs is None:  # pragma: no cover - we warn at runtime
        raise RuntimeError(
            "s3fs is required to fetch data but is not installed."
        )
    return s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": None})


def list_scene_objects(
    config: GOESConfig,
    scene_time: datetime,
    product: str,
    *,
    channel: str | None = None,
) -> list[str]:
    start, end = config.valid_time_window(scene_time)
    prefix = config.object_key_prefix(scene_time, product=product)
    fs = _fs(config)
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
    chunks: Mapping[str, int] | None = None,
    channel: str | None = None,
) -> xr.Dataset:
    keys = list_scene_objects(config, scene_time, product, channel=channel)
    if not keys:
        raise FileNotFoundError(
            f"No {product} objects found for {scene_time.isoformat()}"
        )
    fs = _fs(config)
    uris = [
        f"s3://{key}" if not key.startswith("s3://") else key for key in keys
    ]
    LOGGER.info("Opening %s datasets: %d granules", product, len(uris))
    # Always open and eagerly load datasets into memory for robustness
    open_kwargs = {"engine": "h5netcdf"}
    loaded: list[xr.Dataset] = []
    for uri in uris:
        ds = xr.open_dataset(fs.open(uri, mode="rb"), **open_kwargs)
        loaded.append(ds.load())
    if len(loaded) > 1:
        return xr.concat(loaded, dim="y").load()
    return loaded[0]


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
    subset = subset_sector(dataset, sector)
    return subset.load()


def subset_sector(dataset: xr.Dataset, sector: SectorDefinition) -> xr.Dataset:
    x = dataset.coords.get("x")
    y = dataset.coords.get("y")
    if x is None or y is None:
        raise ValueError("Dataset missing GOES projection coordinates")
    lon2d, lat2d = abi_xy_to_lonlat(x.values, y.values)
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
        valid_rows = np.any(mask, axis=1)
        valid_cols = np.any(mask, axis=0)
        return ds.isel(y=valid_rows, x=valid_cols)
    return ds


def abi_xy_to_lonlat(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts from GOES satellite projection coordinates (x, y radians)
    to geographic coordinates (longitude, latitude in degrees).
    This uses the fixed-grid projection parameters defined
    in ABI_PROJECTION.
    """
    lon0 = np.deg2rad(
        ABI_PROJECTION["longitude_of_projection_origin"]
    )
    r_eq = ABI_PROJECTION["semi_major_axis"]  # Equatorial radius (~6378137 m)
    r_pol = ABI_PROJECTION["semi_minor_axis"]  # Polar radius (~6356752.31 m)
    H = 35786023.0  # Altitude of the satellite (≈35786023 m)

    x_rad = np.asarray(x)
    y_rad = np.asarray(y)
    if x_rad.ndim == 1 and y_rad.ndim == 1:
        x_rad, y_rad = np.meshgrid(x_rad, y_rad)

    cos_x = np.cos(x_rad)
    cos_y = np.cos(y_rad)
    sin_x = np.sin(x_rad)
    sin_y = np.sin(y_rad)

    a = (sin_x**2) + (cos_x**2) * (
        (cos_y**2) + ((r_eq**2) / (r_pol**2)) * (sin_y**2)
    )
    under_sqrt = (H * cos_x * cos_y) ** 2 - (a * (H**2 - r_eq**2))
    under_sqrt = np.maximum(under_sqrt, 0.0)
    rs = (H * cos_x * cos_y) - np.sqrt(under_sqrt)
    sx = rs * cos_y * sin_x
    sy = -rs * sin_y
    sz = rs * cos_y * cos_x

    lon = lon0 + np.arctan2(sx, sz)
    lat = np.arctan(
        (r_eq**2 / r_pol**2) * (sy / np.sqrt(sx**2 + sz**2))
    )

    return np.rad2deg(lon), np.rad2deg(lat)


def download_channels(
    scene_time: datetime,
    output_dir: Path,
    *,
    channels: Iterable[str] = ("C02", "C07", "C14"),
    sector: SectorDefinition = SAN_FRANCISCO_SECTOR,
    config: GOESConfig | None = None,
) -> Dict[str, str]:
    """Download specified ABI L1b channels for a scene and save to NetCDF.

    Returns a mapping of channel -> saved file path.
    """
    cfg = config or GOESConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}
    for channel in channels:
        ds = fetch_ABI_L1b(channel, scene_time, sector, cfg)
        fname = (
            f"goes18_{cfg.product}_{channel}_"
            f"{scene_time:%Y%m%dT%H%M%S}_SF.nc"
        )
        path = output_dir / fname
        ds.to_netcdf(path)
        saved[channel] = str(path)
    return saved


__all__ = [
    "SectorDefinition",
    "SAN_FRANCISCO_SECTOR",
    "fetch_ABI_L1b",
    "download_channels",
    "subset_sector",
    "abi_xy_to_lonlat",
]
