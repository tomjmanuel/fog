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

        semi_minor = _get_first_float(
            ["semi_minor_axis", "b"]
        )

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

    # The GOES ABI "x"/"y" coordinates are scan angles in radians.
    # Convert scan angles to projection plane meters expected by PROJ's
    # geostationary projection using the tangent relationship.
    # This preserves spatial accuracy across the full disk.
    X_m = np.tan(X) * projection.perspective_point_height
    Y_m = np.tan(Y) * projection.perspective_point_height
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


def _edges_from_centers(values: np.ndarray) -> np.ndarray:
    """Compute cell edges from 1D center coordinates.

    Works for non-uniform spacing. Returns an array of length ``n+1`` where
    ``n`` is the length of ``values``.
    """
    centers = np.asarray(values)
    if centers.ndim != 1:
        raise ValueError("Center coordinates must be 1D arrays")
    n = centers.size
    if n == 0:
        raise ValueError("Empty coordinate array")
    if n == 1:
        # Create an arbitrary small half-width when only one pixel is present
        half = 0.5 * (np.abs(centers[0]) + 1.0) * 1e-6
        return np.array([centers[0] - half, centers[0] + half], dtype=float)

    mid = 0.5 * (centers[:-1] + centers[1:])
    edges = np.empty(n + 1, dtype=float)
    edges[1:n] = mid
    edges[0] = centers[0] - (mid[0] - centers[0])
    edges[n] = centers[-1] + (centers[-1] - mid[-1])
    return edges


def _densify_edges(edges: np.ndarray, factor: int) -> np.ndarray:
    """Densify an edge array by an integer ``factor``.

    If ``edges`` has length ``N+1``, the returned array has length
    ``N*factor + 1``. Interpolation is performed in the native coordinate
    space (scan angle in radians), preserving spatial precision.
    """
    if factor <= 1:
        return edges
    orig_len = edges.size - 1
    new_len = orig_len * factor + 1
    x_old = np.linspace(0.0, float(orig_len), edges.size)
    x_new = np.linspace(0.0, float(orig_len), new_len)
    return np.interp(x_new, x_old, edges)


def lonlat_edges_grid(
    dataset: xr.Dataset,
    upsample_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return lon/lat edge grids for warping with pcolormesh.

    The output arrays have shapes ``(ny+1, nx+1)``. When ``upsample_factor``
    is greater than 1, the returned grids are densified accordingly to match
    data that has been upsampled in pixel space by the same factor.
    """
    projection = _extract_projection(dataset)

    x = dataset.coords.get("x")
    y = dataset.coords.get("y")
    if x is None or y is None:
        raise ValueError("Dataset is missing 'x'/'y' projection coordinates")

    x_vals = np.asarray(x.values)
    y_vals = np.asarray(y.values)
    if x_vals.ndim != 1 or y_vals.ndim != 1:
        # If provided as 2D, we can't reliably build edges without additional
        # assumptions; fall back to center-based approach which still works
        # with pcolormesh but without exact cell edges.
        x_m, y_m = _xy_arrays(dataset, projection)
        transformer = Transformer.from_crs(
            projection.crs, "EPSG:4326", always_xy=True
        )
        lon, lat = transformer.transform(x_m, y_m)
        # Build pseudo-edges by padding centers
        ny, nx = lon.shape

        def pad_edges(arr: np.ndarray, axis: int) -> np.ndarray:
            # Average adjacent values to compute interior edges
            if axis == 1:
                mid = 0.5 * (arr[:, :-1] + arr[:, 1:])
                left = (2 * arr[:, :1]) - mid[:, :1]
                right = (2 * arr[:, -1:]) - mid[:, -1:]
                return np.concatenate([left, mid, right], axis=1)
            else:
                mid = 0.5 * (arr[:-1, :] + arr[1:, :])
                top = (2 * arr[:1, :]) - mid[:1, :]
                bottom = (2 * arr[-1:, :]) - mid[-1:, :]
                return np.concatenate([top, mid, bottom], axis=0)

        if upsample_factor > 1:
            # Densify centers before padding to edges
            from scipy.ndimage import zoom  # type: ignore

            lon = zoom(lon, upsample_factor, order=1)
            lat = zoom(lat, upsample_factor, order=1)

        lon_e = pad_edges(pad_edges(lon, axis=1), axis=0)
        lat_e = pad_edges(pad_edges(lat, axis=1), axis=0)
        return lon_e, lat_e

    # 1D coordinate case (typical for GOES)
    x_edges = _edges_from_centers(x_vals)
    y_edges = _edges_from_centers(y_vals)

    x_edges = _densify_edges(x_edges, upsample_factor)
    y_edges = _densify_edges(y_edges, upsample_factor)

    # Convert scan-angle edges (radians) to projection plane meters
    X_e, Y_e = np.meshgrid(x_edges, y_edges)
    Xm = np.tan(X_e) * projection.perspective_point_height
    Ym = np.tan(Y_e) * projection.perspective_point_height

    transformer = Transformer.from_crs(
        projection.crs, "EPSG:4326", always_xy=True
    )
    lon_e, lat_e = transformer.transform(Xm, Ym)
    return lon_e, lat_e


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
    "lonlat_edges_grid",
    "project_xy_to_lonlat",
]
