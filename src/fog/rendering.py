"""Reusable rendering utilities for GOES fog visualizations."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from PIL import Image

from .fetch import SectorDefinition

RADIANCE_LOWER_CLIP = 6
RADIANCE_UPPER_CLIP = 140
BASE_IMAGE_ALPHA = 0.37


def resample_radiance_to_base_image(
    radiance: xr.DataArray,
    base_image: np.ndarray,
    base_image_sector: SectorDefinition,
) -> np.ndarray:
    """Interpolate radiance values onto the pixel grid of ``base_image``."""
    base_x = np.linspace(
        base_image_sector.west,
        base_image_sector.east,
        base_image.shape[1],
    )
    base_y = np.linspace(
        base_image_sector.north,
        base_image_sector.south,
        base_image.shape[0],
    )
    base_X, base_Y = np.meshgrid(base_x, base_y)

    pts = np.vstack(
        [
            radiance.coords["lon"].values.ravel(),
            radiance.coords["lat"].values.ravel(),
        ]
    ).T
    values = radiance.values.ravel()

    mask = (
        np.isfinite(values)
        & np.isfinite(pts[:, 0])
        & np.isfinite(pts[:, 1])
    )
    pts = pts[mask]
    values = values[mask]

    return griddata(pts, values, (base_X, base_Y), method="cubic")


def sector_extent(sector: SectorDefinition) -> Tuple[float, float, float, float]:
    """Return a matplotlib extent tuple for a sector."""
    return (
        sector.west,
        sector.east,
        sector.south,
        sector.north,
    )


def create_radiance_with_coastline(
    base_image: np.ndarray,
    radiance_resampled: np.ndarray,
    coastline_image: np.ndarray,
) -> Image.Image:
    """
    Create a PIL RGB image of the radiance, with the coastline mask added to the blue channel.
    """

    if base_image.shape[:2] != coastline_image.shape[:2]:
        raise ValueError(
            "Coastline image must have the same height/width as the base image."
        )
    if radiance_resampled.shape[:2] != base_image.shape[:2]:
        raise ValueError(
            "Radiance image must match the base image resolution."
        )

    # Prepare radiance as grayscale for R and G channels
    radiance_upper_clip_this_image = max(RADIANCE_UPPER_CLIP, np.max(radiance_resampled))
    radiance_resampled = np.nan_to_num(radiance_resampled, nan=0)
    radiance_rescaled = radiance_resampled - RADIANCE_LOWER_CLIP
    radiance_rescaled *= 255 / (radiance_upper_clip_this_image - RADIANCE_LOWER_CLIP)
    radiance_clipped = np.clip(radiance_rescaled, 0, 255).astype(np.uint8)

    # Coastline mask in blue channel: max out blue (255) where coastline is present
    # coastline_mask: True where coastline_image == 0 (~land). False otherwise.
    coastline_mask = (coastline_image < 10)[:, :, 0]

    red_channel = radiance_clipped * (1 - BASE_IMAGE_ALPHA) + base_image[:, :, 0] * BASE_IMAGE_ALPHA
    green_channel = radiance_clipped * (1 - BASE_IMAGE_ALPHA) + base_image[:, :, 1] * BASE_IMAGE_ALPHA + coastline_mask * 255
    blue_channel = radiance_clipped * (1 - BASE_IMAGE_ALPHA) + base_image[:, :, 2] * BASE_IMAGE_ALPHA + coastline_mask * 255

    rgb = np.stack(
        [
            np.clip(red_channel, 0, 255),
            np.clip(green_channel, 0, 255),
            np.clip(blue_channel, 0, 255),
        ],
        axis=-1
    )

    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def render_scene_to_file(
    dataset: xr.Dataset,
    base_image: np.ndarray,
    coastline_image: np.ndarray,
    output_path: Path,
    *,
    sector: SectorDefinition,
    dpi: int = 400,
) -> Path:
    """Render ``dataset`` radiance over ``base_image`` and save to disk."""
    radiance = dataset["Rad"]
    resampled = resample_radiance_to_base_image(
        radiance, base_image, sector
    )
    radiance_with_coastline = create_radiance_with_coastline(
        base_image,
        resampled,
        coastline_image,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    radiance_with_coastline.save(output_path)
    return output_path


__all__ = [
    "resample_radiance_to_base_image",
    "sector_extent",
    "create_radiance_with_coastline",
    "render_scene_to_file",
]
