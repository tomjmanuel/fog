#!/usr/bin/env python3
"""Render every GOES scene between today's sunrise and sunset in San Francisco."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Tuple
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import sun
from rich.console import Console

from fog.render_now import (
    _default_base,
    build_presets,
    render_scene_for_presets,
)

SF_LOCATION = LocationInfo(
    name="San Francisco",
    region="USA",
    timezone="America/Los_Angeles",
    latitude=37.7749,
    longitude=-122.4194,
)

console = Console()


def _local_today() -> date:
    local_now = datetime.now(ZoneInfo(SF_LOCATION.timezone))
    return local_now.date()


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Date must be formatted as YYYY-MM-DD."
        ) from exc


def _sun_window(target_date: date) -> Tuple[datetime, datetime]:
    """Return sunrise/sunset timestamps in UTC for the location/date."""
    s = sun(
        SF_LOCATION.observer,
        date=target_date,
        tzinfo=timezone.utc,
    )

    # sunset will cross a day boundary, so adjust the day by one day (approximately correct)
    sunrise = s["sunrise"] + timedelta(minutes=30)
    sunset = s["sunset"] + timedelta(days=1)
    return sunrise, sunset


def _generate_scene_times(
    start: datetime,
    end: datetime,
    *,
    interval_minutes: int,
) -> List[datetime]:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    total: List[datetime] = []
    current = start
    delta = timedelta(minutes=interval_minutes)
    while current <= end:
        total.append(current)
        current += delta
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fog.render_daily across all daylight GOES scenes for today "
            "in San Francisco."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--date",
        type=_parse_date,
        default=None,
        help="Override the local date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=20,
        help="Spacing in minutes between successive renders.",
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
        help="Binary coastline mask to overlay on renders.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    target_date = args.date or _local_today()
    sunrise, sunset = _sun_window(target_date)
    console.log(
        f"Rendering scenes from sunrise ({sunrise.isoformat()}) "
        f"to sunset ({sunset.isoformat()}) UTC for {target_date}."
    )

    if sunrise >= sunset:
        raise RuntimeError(
            "Sunrise occurs after sunset; check the selected date/location."
        )

    scene_times = _generate_scene_times(
        sunrise,
        sunset,
        interval_minutes=args.interval_minutes,
    )
    console.log(f"Found {len(scene_times)} scene windows.")

    presets = build_presets(
        args.base_image,
        args.coastline_image,
    )

    successes = 0
    for idx, scene_time in enumerate(scene_times, start=1):
        console.log(
            f"[{idx}/{len(scene_times)}] Rendering {scene_time.isoformat()}..."
        )
        try:
            render_scene_for_presets(
                scene_time,
                presets=presets,
                data_dir=args.data_dir,
                render_dir=args.render_dir,
            )
            successes += 1
        except Exception as exc:  # pragma: no cover - best effort batch run
            console.print(f"[red]Failed for {scene_time}: {exc}")

    console.log(
        f"Completed {successes}/{len(scene_times)} renders. "
        "Use fog.s3_uploader to publish the outputs when ready."
    )


if __name__ == "__main__":
    main()
