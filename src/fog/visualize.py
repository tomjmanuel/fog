"""Visualize GOES-18 ABI L1b channel files stored locally.

Usage (after installing the project):

    python -m fog.visualize --input-dir ./fog_data

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
from .config import OverlayConfig, load_overlay_config


def _cells_within_bbox(
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    bbox: Sequence[float],
) -> np.ndarray:
    """Return a mask selecting grid cells that intersect ``bbox``."""

    lon_min, lon_max, lat_min, lat_max = bbox
    # Compute cell centers from the supplied edge arrays. ``lonlat_edges_grid``
    # returns arrays with shape (ny + 1, nx + 1); the radiance grid has shape
    # (ny, nx). We average the four surrounding edges to obtain the center.
    lon_center = (
        lon_edges[:-1, :-1]
        + lon_edges[1:, :-1]
        + lon_edges[:-1, 1:]
        + lon_edges[1:, 1:]
    ) * 0.25
    lat_center = (
        lat_edges[:-1, :-1]
        + lat_edges[1:, :-1]
        + lat_edges[:-1, 1:]
        + lat_edges[1:, 1:]
    ) * 0.25

    return (
        (lon_center >= lon_min)
        & (lon_center <= lon_max)
        & (lat_center >= lat_min)
        & (lat_center <= lat_max)
    )


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
        "--base-image",
        type=Path,
        help="Optional high-resolution base image for overlay",
    )
    parser.add_argument(
        "--overlay-config",
        type=Path,
        help="JSON configuration describing the base image bounding box",
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


def _extent_from_lonlat(dataset: xr.Dataset) -> Sequence[float] | None:
    lon = dataset.coords.get("lon")
    lat = dataset.coords.get("lat")
    lon_vals = np.asarray(lon.values)
    lat_vals = np.asarray(lat.values)
    lon_min = float(np.nanmin(lon_vals))
    lon_max = float(np.nanmax(lon_vals))
    lat_min = float(np.nanmin(lat_vals))
    lat_max = float(np.nanmax(lat_vals))
    return [lon_min, lon_max, lat_min, lat_max]


def _alpha_from_values(values: np.ndarray) -> np.ndarray:
    """Generate alpha values for radiance data using a lookup table."""

    table_values = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 200.0])
    table_alpha = np.array([0.2, 0.40, 0.60, 1.0, 1.0, 1.0])
    clipped = np.clip(values, table_values[0], table_values[-1])
    return np.interp(clipped, table_values, table_alpha)


def visualize_directory(
    input_dir: Path,
    base_image_path: Path | None = None,
    overlay_config: OverlayConfig | None = None,
) -> None:
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")


    files = _find_channel_files(input_dir)
    if not files:
        raise FileNotFoundError(
            f"No NetCDF channel files found in {input_dir}"
        )

    base_image = None
    base_extent: Sequence[float] | None = None
    if base_image_path is not None:
        if overlay_config is None:
            raise ValueError("Overlay config is required when providing a base image")
        base_image = plt.imread(base_image_path)
        base_extent = overlay_config.bounding_box

    for path in files:
        # Always load fully to avoid lazy/dask surprises
        ds = xr.open_dataset(path, engine="h5netcdf").load()
        rad = _select_radiance_variable(ds)
        # Handle empty arrays gracefully (e.g., no coverage in subset files)
        ydim, xdim = rad.dims[-2:]
        ny, nx = int(rad.sizes[ydim]), int(rad.sizes[xdim])
        ch = _extract_channel_id(path)

        extent = _extent_from_lonlat(ds)

        fig, ax_r = plt.subplots(1, 2, figsize=(10, 6))
        ax_r[0].imshow(
            base_image,
            extent=base_extent,
            origin="lower",
            aspect="equal",
        )
        ax_r[0].set_xlim(base_extent[0], base_extent[1])
        ax_r[0].set_ylim(base_extent[3], base_extent[2])

        # Radiance plot
        lon_e, lat_e = lonlat_edges_grid(ds)
        # mask = _cells_within_bbox(lon_e, lat_e, base_extent)
        # data_to_plot = np.where(mask, r, np.nan)
        # alpha_mask = _alpha_from_values(r)
        # alpha_values = np.where(mask, alpha_mask, 0.0)

        ax_r[1].imshow(
            rad.values,
            extent=extent,
            aspect="equal",
        )
        # ax_r[1].set_xlim(base_extent[1], base_extent[0])
        # ax_r[1].set_ylim(extent[3], extent[2])
   

        fig.suptitle(f"GOES-18 ABI L1b | {path.name}")
        fig.tight_layout()

    plt.show()


def main(argv: Iterable[str] | None = None) -> None:
    parser = _arg_parser()
    args = parser.parse_args(argv)

    # overlay config has the bounding box for now (lat lon)
    overlay_conf = load_overlay_config(args.overlay_config) if args.overlay_config else None

    visualize_directory(
        args.input_dir,
        base_image_path=args.base_image,
        overlay_config=overlay_conf,
    )


if __name__ == "__main__":
    main()
