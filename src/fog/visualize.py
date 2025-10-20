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

from .projection import extent_from_dataset, lonlat_edges_grid


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

    # Always perform interpolation in pixel-index space to avoid mixing with
    # geolocation/scan-angle coordinates that can cause NaNs.
    da_indexed = da.assign_coords({xdim: np.arange(nx), ydim: np.arange(ny)})
    xi = np.linspace(0, nx - 1, new_nx)
    yi = np.linspace(0, ny - 1, new_ny)
    return da_indexed.interp({xdim: xi, ydim: yi})


def _extent_from_projection(dataset: xr.Dataset) -> Sequence[float] | None:
    """Derive lon/lat extent using GOES geostationary projection metadata."""

    return extent_from_dataset(dataset)


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

    for path in files:
        # Always load fully to avoid lazy/dask surprises
        ds = xr.open_dataset(path, engine="h5netcdf").load()
        rad = _select_radiance_variable(ds)
        # Handle empty arrays gracefully (e.g., no coverage in subset files)
        ydim, xdim = rad.dims[-2:]
        ny, nx = int(rad.sizes[ydim]), int(rad.sizes[xdim])
        ch = _extract_channel_id(path)

        if ny == 0 or nx == 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.set_title(f"{ch} (empty)")
            ax.text(0.5, 0.5, "No data in region", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.suptitle(
                f"GOES-18 ABI L1b | {path.name}"
            )
            fig.tight_layout()
            continue

        rad = _maybe_upsample(rad, upsample_factor)
        bt = _compute_brightness_temperature(rad, {**ds.attrs, **rad.attrs})
        extent = _extent_from_lonlat(ds)

        if bt is not None:
            fig, (ax_r, ax_t) = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, ax_r = plt.subplots(1, 1, figsize=(7, 5))

        # Radiance plot
        r = rad.values
        r_vmin = np.nanpercentile(r, 2.0)
        r_vmax = np.nanpercentile(r, 98.0)
        try:
            lon_e, lat_e = lonlat_edges_grid(
                ds, upsample_factor=upsample_factor
            )
            im_r = ax_r.pcolormesh(
                lon_e,
                lat_e,
                r,
                cmap="viridis",
                vmin=r_vmin,
                vmax=r_vmax,
                shading="auto",
            )
            ax_r.set_xlabel("Longitude")
            ax_r.set_ylabel("Latitude")
        except Exception:
            # Fallback to extent-based imshow if projection metadata is missing
            im_r = ax_r.imshow(
                r,
                origin="upper",
                cmap="viridis",
                vmin=r_vmin,
                vmax=r_vmax,
                extent=extent,
                aspect="auto",
            )
            if extent is not None:
                ax_r.set_xlabel("Longitude")
                ax_r.set_ylabel("Latitude")
        ax_r.set_title(f"{ch} Radiance")
        cbar_r = fig.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
        cbar_r.set_label(rad.attrs.get("units", ""))

        # Temperature plot (if available)
        if bt is not None:
            t = bt.values
            t_vmin = np.nanpercentile(t, 2.0)
            t_vmax = np.nanpercentile(t, 98.0)
            used_t_pcolormesh = False
            try:
                lon_e, lat_e = lonlat_edges_grid(
                    ds, upsample_factor=upsample_factor
                )
                im_t = ax_t.pcolormesh(
                    lon_e,
                    lat_e,
                    t,
                    cmap="inferno",
                    vmin=t_vmin,
                    vmax=t_vmax,
                    shading="auto",
                )
                used_t_pcolormesh = True
                ax_t.set_xlabel("Longitude")
            except Exception:
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
            if extent is not None and not used_t_pcolormesh:
                # If we fell back to imshow, label via extent availability
                ax_t.set_xlabel("Longitude")

        fig.suptitle(f"GOES-18 ABI L1b | {path.name}")
        fig.tight_layout()

    plt.show()


def main(argv: Iterable[str] | None = None) -> None:
    parser = _arg_parser()
    args = parser.parse_args(argv)
    visualize_directory(args.input_dir, upsample_factor=args.upsample_factor)


if __name__ == "__main__":
    main()
