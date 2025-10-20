"""Utilities for working with GOES geostationary projection metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class GeostationaryProjection:
    """Container for GOES geostationary projection parameters."""

    longitude_of_projection_origin: float
    perspective_point_height: float
    semi_major_axis: float
    semi_minor_axis: float
    sweep_angle_axis: str

    @property
    def crs(self) -> CRS:
        """Return a ``pyproj.CRS`` describing the geostationary projection."""

        proj4 = (
            " +".join(
                [
                    "proj=geos",
                    f"lon_0={self.longitude_of_projection_origin}",
                    f"h={self.perspective_point_height}",
                    f"a={self.semi_major_axis}",
                    f"b={self.semi_minor_axis}",
                    f"sweep={self.sweep_angle_axis}",
                    "units=m",
                ]
            )
        )
        return CRS.from_proj4("+" + proj4)


def _extract_projection(dataset: xr.Dataset) -> GeostationaryProjection:
    proj_var = dataset.variables.get("goes_imager_projection")
    if proj_var is None:
        raise ValueError(
            "Dataset is missing 'goes_imager_projection' metadata"
        )

    # Read attributes robustly from the variable's attrs with fallbacks.
    attrs = getattr(proj_var, "attrs", {}) or {}

    def _get_first_float(keys: Sequence[str]) -> float:
        for key in keys:
            if key in attrs and attrs[key] is not None:
                return float(attrs[key])
        raise KeyError("/".join(keys))

    def _get_first_str(keys: Sequence[str], default: str | None = None) -> str:
        for key in keys:
            if key in attrs and attrs[key] is not None:
                return str(attrs[key])
        if default is not None:
            return default
        raise KeyError("/".join(keys))

    try:
        # CF attribute names first, then common PROJ parameter fallbacks
        longitude = _get_first_float(
            [
                "longitude_of_projection_origin",
                "lon_0",
            ]
        )
        height = _get_first_float(
            [
                "perspective_point_height",
                "H",
                "h",
            ]
        )
        semi_major = _get_first_float(
            [
                "semi_major_axis",
                "a",
            ]
        )

        # Semi-minor may be missing; derive from inverse_flattening/flattening
        # when possible
        semi_minor: float
        try:
            semi_minor = _get_first_float(
                ["semi_minor_axis", "b"]
            )  # type: ignore[no-redef]
        except KeyError:
            inv_flat: float | None = None
            flat: float | None = None
            # Try common keys for inverse flattening or flattening
            for key in (
                "inverse_flattening",
                "rf",
                "1/f",
                "inverse_flattening_ratio",
            ):
                if key in attrs and attrs[key] not in (None, 0, "0"):
                    inv_flat = float(attrs[key])
                    break
            if inv_flat and inv_flat != 0.0:
                semi_minor = semi_major * (1.0 - 1.0 / inv_flat)
            else:
                for key in ("flattening", "f"):
                    if key in attrs and attrs[key] is not None:
                        flat = float(attrs[key])
                        break
                if flat is not None:
                    semi_minor = semi_major * (1.0 - flat)
                else:
                    # Fall back to WGS-84 flattening if nothing provided
                    wgs84_f = 1.0 / 298.257223563
                    semi_minor = semi_major * (1.0 - wgs84_f)

        sweep = _get_first_str(["sweep_angle_axis"], default="x")
        if sweep not in ("x", "y"):
            sweep = "x"
    except Exception as exc:  # pragma: no cover - metadata errors are rare
        raise ValueError("Invalid GOES projection metadata") from exc

    return GeostationaryProjection(
        longitude_of_projection_origin=longitude,
        perspective_point_height=height,
        semi_major_axis=semi_major,
        semi_minor_axis=semi_minor,
        sweep_angle_axis=sweep,
    )


def _xy_arrays(
    dataset: xr.Dataset,
    projection: GeostationaryProjection,
) -> Tuple[np.ndarray, np.ndarray]:
    x = dataset.coords.get("x")
    y = dataset.coords.get("y")
    if x is None or y is None:
        raise ValueError(
            "Dataset is missing 'x'/'y' projection coordinates"
        )

    x_vals = np.asarray(x.values)
    y_vals = np.asarray(y.values)
    if x_vals.size == 0 or y_vals.size == 0:
        raise ValueError("Projection coordinate arrays are empty")

    if x_vals.ndim == 1 and y_vals.ndim == 1:
        X, Y = np.meshgrid(x_vals, y_vals)
    else:
        X = np.asarray(x_vals)
        Y = np.asarray(y_vals)
        if X.shape != Y.shape:
            raise ValueError(
                "Projection coordinate arrays must share a shape"
            )

    X_m = X * projection.perspective_point_height
    Y_m = Y * projection.perspective_point_height
    return (X_m, Y_m)


def lonlat_grid(dataset: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Return longitude/latitude arrays derived from GOES projection
    metadata.
    """

    projection = _extract_projection(dataset)
    transformer = Transformer.from_crs(
        projection.crs,
        "EPSG:4326",
        always_xy=True,
    )
    x_m, y_m = _xy_arrays(dataset, projection)
    lon, lat = transformer.transform(x_m, y_m)
    return lon, lat


def extent_from_dataset(dataset: xr.Dataset) -> Sequence[float] | None:
    """Compute [lon_min, lon_max, lat_min, lat_max] extent for ``dataset``."""

    try:
        lon, lat = lonlat_grid(dataset)
    except ValueError:
        return None

    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))
    if not np.isfinite([lon_min, lon_max, lat_min, lat_max]).all():
        return None
    if lon_min == lon_max or lat_min == lat_max:
        return None
    return [lon_min, lon_max, lat_min, lat_max]


def project_xy_to_lonlat(
    x: np.ndarray,
    y: np.ndarray,
    dataset: xr.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project arbitrary GOES ``x``/``y`` arrays to lon/lat using
    ``dataset`` metadata.
    """

    projection = _extract_projection(dataset)
    transformer = Transformer.from_crs(
        projection.crs, "EPSG:4326", always_xy=True
    )
    x_vals = np.asarray(x)
    y_vals = np.asarray(y)
    if x_vals.ndim == 1 and y_vals.ndim == 1:
        X, Y = np.meshgrid(x_vals, y_vals)
    else:
        X = np.asarray(x_vals)
        Y = np.asarray(y_vals)
        if X.shape != Y.shape:
            raise ValueError("Projection coordinate arrays must share a shape")

    X_m = X * projection.perspective_point_height
    Y_m = Y * projection.perspective_point_height
    return transformer.transform(X_m, Y_m)


__all__ = [
    "GeostationaryProjection",
    "extent_from_dataset",
    "lonlat_grid",
    "project_xy_to_lonlat",
]
