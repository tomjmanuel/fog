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


def _alpha_from_values(values: np.ndarray) -> np.ndarray:
    """Generate alpha values for radiance data using a lookup table."""
    table_values = np.array([0.0, 10.0, 20.0, 40.0])
    table_alpha = np.array([0.1, 0.5, 1.0, 1.0])
    clean = np.nan_to_num(values, nan=0.0).astype(float, copy=False)
    clipped = np.clip(clean, table_values[0], table_values[-1])
    return np.interp(clipped, table_values, table_alpha)


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


def create_overlay_and_raw_images(
    base_image: np.ndarray,
    radiance_resampled: np.ndarray,
) -> tuple[Image.Image, Image.Image]:
    """Create two PIL images: overlay and raw radiance, using numpy alpha blending."""

    # Calculate alpha mask (0-1)
    alpha = _alpha_from_values(radiance_resampled)

    # Direct numpy overlay (radiance on base using alpha mask)
    radiance_resampled[np.isnan(radiance_resampled)] = 0

    # normalize radiance to 0-255
    radiance_normalized = (radiance_resampled - radiance_resampled.min()) / (radiance_resampled.max() - radiance_resampled.min()) * 255
    overlay_im_data = (base_image * (1 - alpha) + radiance_normalized * alpha).clip(0, 255).astype(np.uint8)

    # Convert to PIL Images
    overlay_pil = Image.fromarray(overlay_im_data, mode="L")
    raw_pil = Image.fromarray(radiance_normalized.astype(np.uint8), mode="L")
    return overlay_pil, raw_pil


def render_scene_to_file(
    dataset: xr.Dataset,
    base_image: np.ndarray,
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
    overlay_image, raw_image = create_overlay_and_raw_images(
        base_image,
        resampled,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fname_raw = output_path.with_name(
        output_path.stem + "_radiance.png"
    )

    overlay_image.save(output_path)
    raw_image.save(fname_raw)

    return output_path


__all__ = [
    "resample_radiance_to_base_image",
    "sector_extent",
    "create_overlay_and_raw_images",
    "render_scene_to_file",
]
