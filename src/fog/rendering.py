"""Reusable rendering utilities for GOES fog visualizations."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.interpolate import griddata

from .fetch import SectorDefinition


def _alpha_from_values(values: np.ndarray) -> np.ndarray:
    """Generate alpha values for radiance data using a lookup table."""
    table_values = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 200.0])
    table_alpha = np.array([0.2, 0.60, 0.80, 0.85, 0.90, 1.0])
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


def create_overlay_figure(
    base_image: np.ndarray,
    radiance_resampled: np.ndarray,
    sector: SectorDefinition,
    *,
    figsize: Tuple[int, int] = (12, 6),
    title: str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Create the two-panel overlay figure used across the project."""
    extent = sector_extent(sector)
    alpha_mask = _alpha_from_values(radiance_resampled)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(
        base_image,
        aspect="equal",
        extent=extent,
        cmap="gray",
    )
    axes[0].imshow(
        radiance_resampled,
        aspect="equal",
        cmap="gray",
        alpha=alpha_mask,
        extent=extent,
    )
    axes[0].set_title("Overlay")

    axes[1].imshow(
        radiance_resampled,
        aspect="equal",
        cmap="gray",
        extent=extent,
    )
    axes[1].set_title("Radiance")

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    return fig, (axes[0], axes[1])


def render_scene_to_file(
    dataset: xr.Dataset,
    base_image: np.ndarray,
    output_path: Path,
    *,
    sector: SectorDefinition,
    title: str | None = None,
    dpi: int = 200,
    figsize: Tuple[int, int] = (12, 6),
) -> Path:
    """Render ``dataset`` radiance over ``base_image`` and save to disk."""
    radiance = dataset["Rad"]
    resampled = resample_radiance_to_base_image(
        radiance, base_image, sector
    )
    fig, _ = create_overlay_figure(
        base_image,
        resampled,
        sector,
        figsize=figsize,
        title=title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


__all__ = [
    "resample_radiance_to_base_image",
    "sector_extent",
    "create_overlay_figure",
    "render_scene_to_file",
]
