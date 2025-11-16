"""Visualize GOES-18 ABI L1b channel files stored locally."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import xarray as xr
import matplotlib.pyplot as plt
from .fetch import SectorDefinition, SAN_FRANCISCO_SECTOR
from .rendering import (
    create_overlay_and_raw_images,
    resample_radiance_to_base_image,
)


def _default_resource(path_name: str) -> Path:
    return Path(__file__).resolve().parents[2] / path_name


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
        default=_default_resource("resources/San_Francisco_Bay.jpg"),
        help="High-resolution base image for overlay",
    )
    parser.add_argument(
        "--coastline-image",
        type=Path,
        default=_default_resource("resources/San_Francisco_Bay_Edges.jpg"),
        help="Binary coastline mask aligned with the base image",
    )
    return parser


def visualize_directory(
    input_dir: Path,
    base_image_path: Path,
    base_image_sector: SectorDefinition,
    coastline_image_path: Path,
) -> None:
    plt.style.use("dark_background")

    files = list(input_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No NetCDF channel files found in {input_dir}"
        )

    base_image = plt.imread(base_image_path)
    coastline_image = plt.imread(coastline_image_path)

    for path in files:
        # Always load fully to avoid lazy/dask surprises
        ds = xr.open_dataset(path, engine="h5netcdf").load()

        # Radiance has not been projected (resampled)
        # But the lon lat values have been added in subset_sector
        # "lon" and "lat" are the projected lon lat values
        radiance = ds["Rad"]
        radiance_resampled = resample_radiance_to_base_image(
            radiance, base_image, base_image_sector
        )
        scene_title = path.name.replace(".nc", "")
        overlay_image, raw_image = create_overlay_and_raw_images(
            base_image,
            radiance_resampled,
            coastline_image,

        )

        overlay_image.save(f"{scene_title}_overlay.png")
        raw_image.save(f"{scene_title}_raw.png")


def main(argv: Iterable[str] | None = None) -> None:
    parser = _arg_parser()
    args = parser.parse_args(argv)

    visualize_directory(
        args.input_dir,
        base_image_path=args.base_image,
        base_image_sector=SAN_FRANCISCO_SECTOR,
        coastline_image_path=args.coastline_image,
    )


if __name__ == "__main__":
    main()
