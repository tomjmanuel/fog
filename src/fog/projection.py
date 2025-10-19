"""Projection utilities for GOES ABI data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import xarray as xr
from pyproj import CRS

from .fetch import ABI_PROJECTION, SectorDefinition


@dataclass(slots=True)
class MapGrid:
    lon: np.ndarray
    lat: np.ndarray
    resolution: float

    def extent(self) -> Tuple[float, float, float, float]:
        return (self.lon.min(), self.lat.min(), self.lon.max(), self.lat.max())


ABI_CRS = CRS.from_dict(
    {
        "proj": "geos",
        "lon_0": ABI_PROJECTION["longitude_of_projection_origin"],
        "h": 35786023.0,
        "a": ABI_PROJECTION["semi_major_axis"],
        "b": ABI_PROJECTION["semi_minor_axis"],
        "sweep": ABI_PROJECTION["sweep_angle_axis"],
    }
)

GEODETIC_CRS = CRS.from_epsg(4326)


def project_dataset(dataset: xr.Dataset, sector: SectorDefinition) -> xr.Dataset:
    lon = dataset["longitude"] if "longitude" in dataset.coords else dataset["lon"]
    lat = dataset["latitude"] if "latitude" in dataset.coords else dataset["lat"]
    data_vars = {}
    for name, da in dataset.data_vars.items():
        data_vars[name] = da
    projected = xr.Dataset(data_vars=data_vars, coords={"lon": lon, "lat": lat})
    return projected.sel(lon=slice(sector.west, sector.east), lat=slice(sector.south, sector.north))


def build_high_res_grid(sector: SectorDefinition, resolution_km: float = 0.5) -> MapGrid:
    lon = np.arange(sector.west, sector.east, resolution_km / 100.0)
    lat = np.arange(sector.south, sector.north, resolution_km / 100.0)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return MapGrid(lon=lon2d, lat=lat2d, resolution=resolution_km)


def resample_to_grid(data: xr.DataArray, grid: MapGrid, method: str = "linear") -> xr.DataArray:
    lon = data.coords.get("lon")
    lat = data.coords.get("lat")
    if lon is None or lat is None:
        raise ValueError("DataArray must have lon/lat coordinates")
    interpolated = data.interp(lon=("y", grid.lon[0, :]), lat=("x", grid.lat[:, 0]), method=method)
    interpolated = interpolated.assign_coords({"lon": ("x", grid.lon[0, :]), "lat": ("y", grid.lat[:, 0])})
    return interpolated


__all__ = ["MapGrid", "project_dataset", "build_high_res_grid", "resample_to_grid"]
