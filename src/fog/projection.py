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

    X, Y = np.meshgrid(x_vals, y_vals)

    X_m = X * projection.perspective_point_height
    Y_m = Y * projection.perspective_point_height
    return transformer.transform(X_m, Y_m)


__all__ = [
    "GeostationaryProjection",
    "project_xy_to_lonlat",
]
