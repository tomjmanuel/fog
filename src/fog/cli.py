"""Command line interface for downloading GOES-18 ABI channels."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

from rich.console import Console

from .config import default_config
from .fetch import (
    SAN_FRANCISCO_SECTOR,
    SectorDefinition,
    download_channels,
)

console = Console()


def _parse_time(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into a ``datetime`` instance."""
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(
            "scene time must be an ISO-8601 timestamp, e.g. 2023-07-01T12:30"
        ) from exc


def _parse_sector(value: str) -> SectorDefinition:
    """
    Parse a comma-separated bounding box into a
    :class:`SectorDefinition`.
    """
    parts = value.split(",")
    if len(parts) != 4:
        msg = "sector must be 'west,south,east,north'"
        raise argparse.ArgumentTypeError(msg)

    try:
        west, south, east, north = (float(v) for v in parts)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        msg = "sector bounds must be numeric"
        raise argparse.ArgumentTypeError(msg) from exc

    return SectorDefinition(west=west, south=south, east=east, north=north)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "GOES-18 downloader: save C02, C07, C14 for SF sector"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene-time",
        required=True,
        type=_parse_time,
        help="ISO-8601 time of the desired scan (UTC)",
    )
    parser.add_argument(
        "--sector",
        type=_parse_sector,
        default=SAN_FRANCISCO_SECTOR,
        help="Bounding box expressed as west,south,east,north",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write NetCDF files",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for downloading ABI channels."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = default_config()
    console.log("Downloading ABI channels C02, C07, C14 for SF sector...")
    saved = download_channels(
        args.scene_time,
        args.output_dir,
        channels=("C01", "C02", "C03", "C07", "C13", "C14"),
        sector=args.sector,
        config=cfg,
    )
    for ch, path in sorted(saved.items()):
        console.log(f"Saved {ch}: {path}")


if __name__ == "__main__":
    main()
