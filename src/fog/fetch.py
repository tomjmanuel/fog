"""Data access utilities for GOES-18 ABI datasets."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Tuple

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
    config: GOESConfig, scene_time: datetime, product: str, *, channel: str | None = None
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


def _ensure_cache_dir(cache_dir: str | None) -> Path | None:
    if not cache_dir:
        return None
    cache_path = Path(os.path.expanduser(cache_dir)).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _cache_path(cache_root: Path | None, key: str, bucket: str) -> Path | None:
    if cache_root is None:
        return None
    normalized = key
    if normalized.startswith("s3://"):
        normalized = normalized[len("s3://") :]
    if normalized.startswith(f"{bucket}/"):
        normalized = normalized[len(bucket) + 1 :]
    local_path = cache_root / normalized
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return local_path


def _ensure_local_copy(
    fs,
    key: str,
    bucket: str,
    cache_root: Path | None,
) -> tuple[str, bool]:
    if cache_root is None:
        uri = key if key.startswith("s3://") else f"s3://{key}"
        return uri, False

    local_path = _cache_path(cache_root, key, bucket)
    if local_path is None:
        uri = key if key.startswith("s3://") else f"s3://{key}"
        return uri, False

    if not local_path.exists():
        remote_path = key
        if remote_path.startswith("s3://"):
            remote_path = remote_path[len("s3://") :]
        LOGGER.info("Caching %s -> %s", remote_path, local_path)
        fs.get(remote_path, str(local_path))
    else:
        LOGGER.debug("Using cached copy for %s", local_path)
    return str(local_path), True


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
    cache_root = _ensure_cache_dir(config.cache_dir)
    uris_with_locality: list[tuple[str, bool]] = [
        _ensure_local_copy(fs, key, config.bucket, cache_root) for key in keys
    ]
    LOGGER.info("Opening %s datasets: %d granules", product, len(uris_with_locality))
    open_kwargs = {"engine": "h5netcdf", "chunks": chunks} if chunks else {"engine": "h5netcdf"}
    datasets: list[xr.Dataset] = []
    for uri, is_local in uris_with_locality:
        if is_local:
            datasets.append(xr.open_dataset(uri, **open_kwargs))
        else:
            datasets.append(xr.open_dataset(fs.open(uri, mode="rb"), **open_kwargs))
    return xr.concat(datasets, dim="y") if len(datasets) > 1 else datasets[0]


def fetch_ABI_L1b(channel: str, scene_time: datetime, sector: SectorDefinition, config: GOESConfig) -> xr.Dataset:
    product = config.product
    dataset = open_dataset(config, scene_time, product=product, channel=channel)
    subset = subset_sector(dataset, sector)
    # Channel is already filtered in open_dataset, no need to select by band
    return subset


def fetch_ABI_geolocation(scene_time: datetime, sector: SectorDefinition, config: GOESConfig) -> xr.Dataset:
    product = "ABI-L2-MCMIPC"
    dataset = open_dataset(config, scene_time, product=product)
    return subset_sector(dataset, sector)


def fetch_ABI_cloud_mask(scene_time: datetime, sector: SectorDefinition, config: GOESConfig) -> xr.DataArray:
    ds = fetch_ABI_geolocation(scene_time, sector, config)
    return ds["Cloud_Mask"]


def fetch_ABI_cloud_phase(scene_time: datetime, sector: SectorDefinition, config: GOESConfig) -> xr.DataArray:
    product = "ABI-L2-CMIPF"
    ds = open_dataset(config, scene_time, product=product)
    return subset_sector(ds, sector)["Phase_Retrieval"]


def fetch_ABI_LWP(scene_time: datetime, sector: SectorDefinition, config: GOESConfig) -> xr.DataArray:
    product = "ABI-L2-LWPRad"
    ds = open_dataset(config, scene_time, product=product)
    return subset_sector(ds, sector)["LWP"]


def fetch_surface_emissivity_maps(channel39: float, channel11: float, sector: SectorDefinition) -> Tuple[np.ndarray, np.ndarray]:
    shape = (100, 100)
    return np.full(shape, 0.95, dtype=np.float32), np.full(shape, 0.98, dtype=np.float32)


def fetch_NWP_surface_temperature(scene_time: datetime, sector: SectorDefinition) -> np.ndarray:
    shape = (100, 100)
    return np.full(shape, 285.0, dtype=np.float32)


def fetch_clear_sky_transmittance_and_radiance(channel: float, scene_time: datetime, sector: SectorDefinition):
    shape = (100, 100)
    tau = np.full(shape, 0.98, dtype=np.float32)
    ratm = np.full(shape, 1.0, dtype=np.float32)
    return tau, ratm


def subset_sector(dataset: xr.Dataset, sector: SectorDefinition) -> xr.Dataset:
    x = dataset.coords.get("x")
    y = dataset.coords.get("y")
    if x is None or y is None:
        raise ValueError("Dataset missing GOES projection coordinates")
    lon2d, lat2d = abi_xy_to_lonlat(x.values, y.values)
    ds = dataset.assign_coords({"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d)})
    lon_mask = (lon2d >= sector.west) & (lon2d <= sector.east)
    lat_mask = (lat2d >= sector.south) & (lat2d <= sector.north)
    mask = lon_mask & lat_mask
    if mask.ndim == 2:
        valid_rows = np.any(mask, axis=1)
        valid_cols = np.any(mask, axis=0)
        return ds.isel(y=valid_rows, x=valid_cols)
    return ds


def abi_xy_to_lonlat(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon0 = np.deg2rad(ABI_PROJECTION["longitude_of_projection_origin"])
    r_eq = ABI_PROJECTION["semi_major_axis"]
    r_pol = ABI_PROJECTION["semi_minor_axis"]
    H = 35786023.0

    x_rad = np.asarray(x)
    y_rad = np.asarray(y)
    if x_rad.ndim == 1 and y_rad.ndim == 1:
        x_rad, y_rad = np.meshgrid(x_rad, y_rad)

    cos_x = np.cos(x_rad)
    cos_y = np.cos(y_rad)
    sin_x = np.sin(x_rad)
    sin_y = np.sin(y_rad)

    a = (sin_x**2) + (cos_x**2) * ((cos_y**2) + ((r_eq**2) / (r_pol**2)) * (sin_y**2))
    under_sqrt = (H * cos_x * cos_y) ** 2 - (a * (H**2 - r_eq**2))
    under_sqrt = np.maximum(under_sqrt, 0.0)
    rs = (H * cos_x * cos_y) - np.sqrt(under_sqrt)
    sx = rs * cos_y * sin_x
    sy = -rs * sin_y
    sz = rs * cos_y * cos_x

    lon = lon0 + np.arctan2(sx, sz)
    lat = np.arctan((r_eq ** 2 / r_pol ** 2) * (sy / np.sqrt(sx ** 2 + sz ** 2)))

    return np.rad2deg(lon), np.rad2deg(lat)


__all__ = [
    "SectorDefinition",
    "SAN_FRANCISCO_SECTOR",
    "fetch_ABI_L1b",
    "fetch_ABI_geolocation",
    "fetch_ABI_cloud_mask",
    "fetch_ABI_cloud_phase",
    "fetch_ABI_LWP",
    "fetch_surface_emissivity_maps",
    "fetch_NWP_surface_temperature",
    "fetch_clear_sky_transmittance_and_radiance",
    "subset_sector",
    "abi_xy_to_lonlat",
]
