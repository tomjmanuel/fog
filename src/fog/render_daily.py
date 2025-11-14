"""CLI utility to render low/high-resolution composites for the current day."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import xarray as xr
from rich.console import Console

from .config import default_config
from .fetch import SAN_FRANCISCO_SECTOR, SectorDefinition, download_channels
from .rendering import render_scene_to_file

console = Console()


@dataclass(frozen=True)
class RenderPreset:
    """Describe a single render target."""

    name: str
    base_image: Path
    dpi: int = 200
    figsize: tuple[int, int] = (12, 6)
    display_name: str | None = None

    @property
    def title(self) -> str:
        return self.display_name or self.name.replace("_", " ").title()


def _default_scene_time() -> datetime:
    now = datetime.now(timezone.utc)
    rounded_minute = (now.minute // 10) * 10
    return now.replace(
        minute=rounded_minute,
        second=0,
        microsecond=0,
    )


def _parse_scene_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Scene time must be ISO-8601, e.g. 2024-01-01T15:00:00+00:00"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_presets(
    low_base: Path,
    high_base: Path,
) -> list[RenderPreset]:
    return [
        RenderPreset(
            name="low_res",
            base_image=low_base,
            display_name="Low Resolution",
            dpi=150,
        ),
        RenderPreset(
            name="high_res",
            base_image=high_base,
            display_name="High Resolution",
            dpi=220,
            figsize=(14, 7),
        ),
    ]


def _load_dataset(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path, engine="h5netcdf")
    try:
        return ds.load()
    finally:
        ds.close()


def _ensure_base_image(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Base image not found: {path}")
    return path


def render_scene_for_presets(
    scene_time: datetime,
    *,
    presets: Iterable[RenderPreset],
    data_dir: Path,
    render_dir: Path,
    sector: SectorDefinition = SAN_FRANCISCO_SECTOR,
) -> Mapping[str, Path]:
    cfg = default_config()
    saved = download_channels(scene_time, data_dir, config=cfg)
    channel_path = saved.get("C02")
    if channel_path is None:
        raise RuntimeError("C02 channel was not downloaded; cannot render.")
    dataset = _load_dataset(Path(channel_path))

    date_folder = render_dir / scene_time.strftime("%Y-%m-%d")
    date_folder.mkdir(parents=True, exist_ok=True)
    timestamp_str = scene_time.strftime("%Y%m%dT%H%M%SZ")

    results: dict[str, Path] = {}
    for preset in presets:
        base_image_path = _ensure_base_image(preset.base_image)
        base_image = plt.imread(base_image_path)
        output_name = f"{timestamp_str}_{preset.name}.png"
        output_path = date_folder / output_name
        title = f"{preset.title} · {scene_time:%Y-%m-%d %H:%MZ}"
        render_scene_to_file(
            dataset,
            base_image,
            output_path,
            sector=sector,
            title=title,
            dpi=preset.dpi,
            figsize=preset.figsize,
        )
        results[preset.name] = output_path
    return results


def _default_base(path_name: str) -> Path:
    # Resolve relative to project root so CLI works from anywhere inside repo.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path_name


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the current day's GOES scene and render low/high "
            "resolution composites."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene-time",
        type=_parse_scene_time,
        default=None,
        help="ISO-8601 time for the render (defaults to 'now').",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory where NetCDF downloads will be stored.",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=Path("renders"),
        help="Directory for rendered PNG files.",
    )
    parser.add_argument(
        "--base-image-low",
        type=Path,
        default=_default_base("San_Francisco_Bay.jpg"),
        help="Low-resolution base image for overlays.",
    )
    parser.add_argument(
        "--base-image-high",
        type=Path,
        default=_default_base("San_Francisco_Bay_full_size.jpg"),
        help="High-resolution base image for overlays.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    scene_time = args.scene_time or _default_scene_time()
    console.log(f"Rendering GOES scene for {scene_time.isoformat()}...")

    presets = build_presets(
        args.base_image_low,
        args.base_image_high,
    )
    paths = render_scene_for_presets(
        scene_time,
        presets=presets,
        data_dir=args.data_dir,
        render_dir=args.render_dir,
    )

    for name, path in paths.items():
        console.log(f"{name}: {path}")
    console.log("Render complete. Upload helpers are defined in fog.s3_uploader.")


if __name__ == "__main__":
    main()
