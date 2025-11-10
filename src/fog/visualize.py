"""Visualize GOES-18 ABI L1b channel files stored locally."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from .fetch import SectorDefinition, SAN_FRANCISCO_SECTOR


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
    return parser


def _alpha_from_values(values: np.ndarray) -> np.ndarray:
    """Generate alpha values for radiance data using a lookup table."""
    # TODO: make this handle differnt brightnesses appropriately
    nan_mask = np.isnan(values)
    values[nan_mask] = 0.0

    table_values = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 200.0])
    table_alpha = np.array([0.2, 0.60, 0.80, 0.85, 0.90, 1.0])
    clipped = np.clip(values, table_values[0], table_values[-1])
    return np.interp(clipped, table_values, table_alpha)


def resample_radiance_to_base_image(radiance: xr.DataArray, base_image: np.ndarray, base_image_sector: SectorDefinition) -> np.ndarray:
    # build coordinates that represent the base image
    base_x = np.linspace(base_image_sector.west, base_image_sector.east, base_image.shape[1])
    base_y = np.linspace(base_image_sector.north, base_image_sector.south, base_image.shape[0])
    base_X, base_Y = np.meshgrid(base_x, base_y)

    pts = np.vstack([radiance.coords["lon"].values.ravel(), radiance.coords["lat"].values.ravel()]).T
    values = radiance.values.ravel()

    # Optionally drop NaNs in source
    m = np.isfinite(values) & np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    pts = pts[m]
    values = values[m]

    return griddata(pts, values, (base_X, base_Y), method="cubic")


def visualize_directory(
    input_dir: Path,
    base_image_path: Path,
    base_image_sector: SectorDefinition,
) -> None:
    plt.style.use("dark_background")

    files = list(input_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No NetCDF channel files found in {input_dir}"
        )

    base_image = plt.imread(base_image_path)

    for path in files:
        # Always load fully to avoid lazy/dask surprises
        ds = xr.open_dataset(path, engine="h5netcdf").load()

        # Radiance has not been projected (resampled)
        # But the lon lat values have been added in subset_sector
        # "lon" and "lat" are the projected lon lat values
        radiance = ds["Rad"]
        radiance_resampled = resample_radiance_to_base_image(radiance, base_image, base_image_sector)

        extent_for_imshow = (
            base_image_sector.west,
            base_image_sector.east,
            base_image_sector.south,
            base_image_sector.north,
        )

        alpha_mask = _alpha_from_values(radiance_resampled)

        fig, ax_r = plt.subplots(1, 2, figsize=(12, 6))
        ax_r[0].imshow(
            base_image,
            aspect="equal",
            extent=extent_for_imshow,
            cmap="gray",
        )

        ax_r[0].imshow(
            radiance_resampled,
            aspect="equal",
            cmap="gray",
            alpha=alpha_mask,
            extent=extent_for_imshow,
        )

        ax_r[1].imshow(
            radiance_resampled,
            aspect="equal",
            cmap="gray",
            extent=extent_for_imshow,
        )
   
        fig.tight_layout()

    plt.show()


def main(argv: Iterable[str] | None = None) -> None:
    parser = _arg_parser()
    args = parser.parse_args(argv)

    visualize_directory(
        args.input_dir,
        base_image_path=args.base_image,
        base_image_sector=SAN_FRANCISCO_SECTOR,
    )


if __name__ == "__main__":
    main()
