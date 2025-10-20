"""Visualize GOES-18 ABI L1b channel files stored locally.

Usage (after installing the project):

    python -m fog.visualize --input-dir ./fog_data --upsample-factor 2

This will open a matplotlib window showing, for each available channel file
in the directory, the radiance and (when applicable) the brightness
temperature derived from the file's Planck calibration metadata.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize GOES-18 ABI L1b channels: radiance and brightness\n"
            "temperature (when available)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded NetCDF channel files",
    )
    parser.add_argument(
        "--upsample-factor",
        type=int,
        default=1,
        help=(
            "Upsample each dataset by this integer factor using bilinear "
            "interpolation before plotting"
        ),
    )
    return parser


def _find_channel_files(directory: Path) -> list[Path]:
    """Return a sorted list of channel NetCDF files in ``directory``.

    The function is permissive about filenames; anything containing
    "_Cxx_" will be treated as a channel file.
    """
    paths: list[Path] = []
    for path in sorted(directory.glob("*.nc")):
        name = path.name
        if "_C" in name and name.lower().endswith(".nc"):
            paths.append(path)
    return paths


def _extract_channel_id(path: Path) -> str:
    # Try to parse "_Cxx_" from the filename; fall back to basename
    name = path.name
    try:
        start = name.index("_C") + 1
        channel = name[start:start + 3]
        if channel.startswith("C") and channel[1:].isdigit():
            return channel
    except ValueError:
        pass
    return name


def _select_radiance_variable(dataset: xr.Dataset) -> xr.DataArray:
    """Pick the most likely radiance variable from the dataset.

    L1b ABI typically stores radiance in a variable named "Rad".
    If "Rad" is not present, we select the first floating-point 2D variable.
    """
    if "Rad" in dataset.data_vars:
        return dataset["Rad"]
    for var_name, da in dataset.data_vars.items():
        if da.ndim == 2 and np.issubdtype(da.dtype, np.floating):
            return da
    raise KeyError("No suitable radiance-like variable found")


def _compute_brightness_temperature(
    radiance: xr.DataArray, attrs: Mapping[str, float]
) -> xr.DataArray | None:
    """Convert radiance to brightness temperature using Planck calibration.

    The GOES-ABI L1b metadata includes the following attributes:
    - planck_fk1, planck_fk2: Planck function coefficients
    - planck_bc1, planck_bc2: calibration slope/offset corrections

    The standard conversion is:
        T_unbiased = fk2 / ln((fk1 / Rad) + 1)
        BT = bc1 + bc2 * T_unbiased

    If the required attributes are missing, ``None`` is returned.
    """
    try:
        fk1 = float(attrs["planck_fk1"])  # type: ignore[index]
        fk2 = float(attrs["planck_fk2"])  # type: ignore[index]
        bc1 = float(attrs.get("planck_bc1", 0.0))
        bc2 = float(attrs.get("planck_bc2", 1.0))
    except Exception:
        return None

    # Guard against division-by-zero and invalid values
    safe_rad = xr.where(radiance <= 0, np.nan, radiance)
    t_unbiased = fk2 / np.log((fk1 / safe_rad) + 1.0)
    bt = bc1 + (bc2 * t_unbiased)
    bt.name = "BrightnessTemperature"
    bt.attrs.update({"units": "K"})
    return bt


def _maybe_upsample(da: xr.DataArray, factor: int) -> xr.DataArray:
    """Upsample a 2D array by an integer factor using pixel coordinates.

    This avoids any reliance on projection coordinates and works even when
    coordinate variables are missing or empty.
    """
    if factor <= 1 or da.ndim < 2:
        return da
    # Use the last two dims as y, x by convention
    ydim, xdim = da.dims[-2:]
    ny, nx = int(da.sizes[ydim]), int(da.sizes[xdim])
    new_nx = max(1, int(nx * factor))
    new_ny = max(1, int(ny * factor))
    # Ensure pixel coordinate variables exist and are non-empty
    if (
        xdim not in da.coords
        or int(getattr(da.coords.get(xdim), "size", 0)) == 0
    ):
        da = da.assign_coords({xdim: np.arange(nx)})
    if (
        ydim not in da.coords
        or int(getattr(da.coords.get(ydim), "size", 0)) == 0
    ):
        da = da.assign_coords({ydim: np.arange(ny)})
    xi = np.linspace(0, nx - 1, new_nx)
    yi = np.linspace(0, ny - 1, new_ny)
    return da.interp({xdim: xi, ydim: yi})


def _extent_from_projection(dataset: xr.Dataset) -> Sequence[float] | None:
    """Derive lon/lat extent using GOES geostationary projection metadata."""

    proj_var = dataset.variables.get("goes_imager_projection")
    x = dataset.coords.get("x")
    y = dataset.coords.get("y")
    if proj_var is None or x is None or y is None:
        return None

    if int(getattr(x, "size", 0)) == 0 or int(getattr(y, "size", 0)) == 0:
        return None

    try:
        H = float(proj_var.perspective_point_height)
        a = float(proj_var.semi_major_axis)
        b = float(proj_var.semi_minor_axis)
        lon0 = float(proj_var.longitude_of_projection_origin)
        sweep = str(getattr(proj_var, "sweep_angle_axis", "x"))
    except Exception:
        return None

    # GOES fixed grid coordinates are scan angles (radians). Convert to meters
    x_m = np.array([x.values[0], x.values[-1]]) * H
    y_m = np.array([y.values[0], y.values[-1]]) * H

    # Build CRS and transform corner coordinates to lon/lat
    crs_geos = CRS.from_proj4(
        f"+proj=geos +lon_0={lon0} +h={H} +a={a} +b={b} +sweep={sweep} +units=m"
    )
    transformer = Transformer.from_crs(crs_geos, "EPSG:4326", always_xy=True)
    X, Y = np.meshgrid(x_m, y_m)
    lon_c, lat_c = transformer.transform(X, Y)

    lon_min = float(np.nanmin(lon_c))
    lon_max = float(np.nanmax(lon_c))
    lat_min = float(np.nanmin(lat_c))
    lat_max = float(np.nanmax(lat_c))
    if not np.isfinite([lon_min, lon_max, lat_min, lat_max]).all():
        return None
    return [lon_min, lon_max, lat_min, lat_max]


def _extent_from_lonlat(dataset: xr.Dataset) -> Sequence[float] | None:
    lon = dataset.coords.get("lon")
    lat = dataset.coords.get("lat")
    if lon is not None and lat is not None:
        lon_vals = np.asarray(lon.values)
        lat_vals = np.asarray(lat.values)
        if lon_vals.size and lat_vals.size:
            lon_min = float(np.nanmin(lon_vals))
            lon_max = float(np.nanmax(lon_vals))
            lat_min = float(np.nanmin(lat_vals))
            lat_max = float(np.nanmax(lat_vals))
            if (
                np.isfinite([lon_min, lon_max, lat_min, lat_max]).all()
                and lon_min != lon_max
                and lat_min != lat_max
            ):
                return [lon_min, lon_max, lat_min, lat_max]

    return _extent_from_projection(dataset)


def visualize_directory(input_dir: Path, upsample_factor: int = 1) -> None:
    import matplotlib.pyplot as plt

    files = _find_channel_files(input_dir)
    if not files:
        raise FileNotFoundError(
            f"No NetCDF channel files found in {input_dir}"
        )

    n = len(files)
    ncols = 2
    fig, axes = plt.subplots(
        n, ncols, figsize=(12, max(3 * n, 3)), squeeze=False
    )

    for row_index, path in enumerate(files):
        # Always load fully to avoid lazy/dask surprises
        ds = xr.open_dataset(path, engine="h5netcdf").load()
        rad = _select_radiance_variable(ds)
        # Handle empty arrays gracefully (e.g., no coverage in subset files)
        ydim, xdim = rad.dims[-2:]
        ny, nx = int(rad.sizes[ydim]), int(rad.sizes[xdim])
        ax_r = axes[row_index, 0]
        ax_t = axes[row_index, 1]
        ch = _extract_channel_id(path)
        if ny == 0 or nx == 0:
            ax_r.set_title(f"{ch} (empty)")
            ax_r.text(0.5, 0.5, "No data in region", ha="center", va="center")
            ax_r.set_xticks([])
            ax_r.set_yticks([])
            ax_t.set_visible(False)
            continue
        rad = _maybe_upsample(rad, upsample_factor)
        bt = _compute_brightness_temperature(rad, {**ds.attrs, **rad.attrs})
        extent = _extent_from_lonlat(ds)

        # Radiance plot
        r = rad.values
        r_vmin = np.nanpercentile(r, 2.0)
        r_vmax = np.nanpercentile(r, 98.0)
        im_r = ax_r.imshow(
            r,
            origin="upper",
            cmap="viridis",
            vmin=r_vmin,
            vmax=r_vmax,
            extent=extent,
            aspect="auto",
        )
        ax_r.set_title(f"{ch} Radiance")
        cbar_r = fig.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
        cbar_r.set_label(rad.attrs.get("units", ""))
        if extent is not None:
            ax_r.set_xlabel("Longitude")
            ax_r.set_ylabel("Latitude")

        # Temperature plot (if available)
        if bt is not None:
            t = bt.values
            t_vmin = np.nanpercentile(t, 2.0)
            t_vmax = np.nanpercentile(t, 98.0)
            im_t = ax_t.imshow(
                t,
                origin="upper",
                cmap="inferno",
                vmin=t_vmin,
                vmax=t_vmax,
                extent=extent,
                aspect="auto",
            )
            ax_t.set_title(f"{ch} Brightness Temperature (K)")
            cbar_t = fig.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)
            cbar_t.set_label("K")
            if extent is not None:
                ax_t.set_xlabel("Longitude")
        else:
            ax_t.set_visible(False)

    fig.suptitle(f"GOES-18 ABI L1b | {input_dir}")
    fig.tight_layout()
    plt.show()


def main(argv: Iterable[str] | None = None) -> None:
    parser = _arg_parser()
    args = parser.parse_args(argv)
    visualize_directory(args.input_dir, upsample_factor=args.upsample_factor)


if __name__ == "__main__":
    main()
