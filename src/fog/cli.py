"""Command line interface for fog probability processing."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

from rich.console import Console

from .config import default_config
from .fetch import SAN_FRANCISCO_SECTOR, SectorDefinition
from .probability import build_fog_probability

console = Console()


def _parse_time(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into a ``datetime`` instance."""
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(
            "scene time must be an ISO-8601 timestamp, e.g. 2023-07-01T12:30:00"
        ) from exc


def _parse_sector(value: str) -> SectorDefinition:
    """Parse a comma-separated bounding box into a :class:`SectorDefinition`."""
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("sector must be 'west,south,east,north'")

    try:
        west, south, east, north = (float(v) for v in parts)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError("sector bounds must be numeric") from exc

    return SectorDefinition(west=west, south=south, east=east, north=north)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="GOES-18 fog probability utilities",
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
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional local cache directory for downloaded products",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for fog probability processing."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = default_config()
    if args.cache_dir:
        cfg.cache_dir = str(args.cache_dir)

    console.log("Fetching GOES data...")
    p_fog, diagnostics = build_fog_probability(args.scene_time, args.sector, cfg)
    console.log(f"Probability array shape: {p_fog.shape}")
    console.log(f"Diagnostics available: {sorted(diagnostics)}")


if __name__ == "__main__":
    main()
