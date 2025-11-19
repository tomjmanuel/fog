"""CLI utility to render low/high-resolution composites for the current time."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Tuple
from time import sleep

from astral import LocationInfo
from astral.sun import sun

import matplotlib.pyplot as plt
import xarray as xr
from rich.console import Console

from .config import default_config
from .fetch import SAN_FRANCISCO_SECTOR, SectorDefinition, download_channels
from .rendering import render_scene_to_file
from .s3_uploader import upload_render_batch

LOOP_INTERVAL_MINUTES = 8

console = Console()

SF_LOCATION = LocationInfo(
    name="San Francisco",
    region="USA",
    timezone="America/Los_Angeles",
    latitude=37.7749,
    longitude=-122.4194,
)


@dataclass(frozen=True)
class RenderPreset:
    """Describe a single render target."""

    name: str
    base_image: Path
    coastline_image: Path
    dpi: int = 400
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
    base_image_path: Path,
    coastline_image_path: Path,
) -> list[RenderPreset]:
    """Leaving this in case we want to add more presets later."""
    return [
        RenderPreset(
            name="standard_render",
            base_image=base_image_path,
            coastline_image=coastline_image_path,
            display_name="Standard Render",
            dpi=400,
        )
    ]


def _load_dataset(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path, engine="h5netcdf")
    try:
        return ds.load()
    finally:
        ds.close()


def _ensure_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _ensure_base_image(path: Path) -> Path:
    return _ensure_path(path, "Base image")


def _ensure_coastline_image(path: Path) -> Path:
    return _ensure_path(path, "Coastline image")


def render_scene_for_presets(
    scene_time: datetime,
    *,
    presets: Iterable[RenderPreset],
    data_dir: Path,
    render_dir: Path,
    sector: SectorDefinition = SAN_FRANCISCO_SECTOR,
) -> Tuple[Mapping[str, Path], datetime]:
    cfg = default_config()
    saved, actual_scene_time = download_channels(scene_time, data_dir, config=cfg)
    channel_path = saved.get("C02")
    if channel_path is None:
        raise RuntimeError("C02 channel was not downloaded; cannot render.")
    dataset = _load_dataset(Path(channel_path))

    date_folder = render_dir / actual_scene_time.strftime("%Y-%m-%d")
    date_folder.mkdir(parents=True, exist_ok=True)
    timestamp_str = actual_scene_time.strftime("%Y%m%dT%H%M%SZ")

    results: dict[str, Path] = {}
    for preset in presets:
        base_image_path = _ensure_base_image(preset.base_image)
        coastline_image_path = _ensure_coastline_image(preset.coastline_image)
        base_image = plt.imread(base_image_path)
        coastline_image = plt.imread(coastline_image_path)
        output_name = f"{timestamp_str}_{preset.name}.png"
        output_path = date_folder / output_name
        render_scene_to_file(
            dataset,
            base_image,
            coastline_image,
            output_path,
            sector=sector,
            dpi=preset.dpi,
        )
        results[preset.name] = output_path
    return results, actual_scene_time


def _default_base(path_name: str) -> Path:
    # Resolve relative to project root so CLI works from anywhere inside repo.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path_name


def _sun_window(target_date: date) -> Tuple[datetime, datetime]:
    """Return sunrise/sunset timestamps in UTC for the location/date."""
    s = sun(
        SF_LOCATION.observer,
        date=target_date,
        tzinfo=timezone.utc,
    )

    # sunset will cross a day boundary, so adjust the day by one day (approximately correct)
    sunrise = s["sunrise"] + timedelta(minutes=30)
    sunset = s["sunset"] + timedelta(days=1) + timedelta(minutes=30)
    return sunrise, sunset


def is_daylight(scene_time: datetime) -> bool:
    # get the sunrise and sunset times for the scene time
    sunrise, sunset = _sun_window(scene_time.date())
    return sunrise < scene_time < sunset


def _render_once(args: argparse.Namespace) -> None:
    scene_time = args.scene_time or _default_scene_time()

    # check if the scene time is within daylight
    # if not, return
    if not is_daylight(scene_time):
        console.log(f"Scene time {scene_time.isoformat()} is not within daylight.")
        return

    console.log(f"Rendering GOES scene for {scene_time.isoformat()}...")

    presets = build_presets(
        args.base_image,
        args.coastline_image,
    )
    paths, actual_scene_time = render_scene_for_presets(
        scene_time,
        presets=presets,
        data_dir=args.data_dir,
        render_dir=args.render_dir,
    )
    if actual_scene_time != scene_time:
        console.log(
            "Resolved scene time differs from requested time: "
            f"{actual_scene_time.isoformat()}"
        )
    for name, path in paths.items():
        console.log(f"{name}: {path}")

    if args.s3_bucket:
        prefix = args.s3_prefix or actual_scene_time.strftime("%Y-%m-%d")
        console.log(f"Uploading renders to s3://{args.s3_bucket}/{prefix}...")
        uris = upload_render_batch(
            paths,
            args.s3_bucket,
            prefix=prefix,
        )
        for uri in uris:
            console.log(f"Uploaded {uri}")
    console.log("Render complete.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the current day's GOES scene and render an overlay image."
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
        "--base-image",
        type=Path,
        default=_default_base("resources/San_Francisco_Bay.jpg"),
        help="Base image for overlays.",
    )
    parser.add_argument(
        "--coastline-image",
        type=Path,
        default=_default_base("resources/San_Francisco_Bay_Edges.jpg"),
        help="Binary coastline mask to overlay on the base image.",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="Name of the bucket where renders should be uploaded.",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="",
        help="Optional key prefix (defaults to the render date).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously render and upload every ten minutes.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    while True:
        _render_once(args)
        if not args.loop:
            break
        args.scene_time = None
        console.log("Waiting ten minutes for the next render...")
        sleep(LOOP_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()
