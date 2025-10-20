"""Command line interface for downloading GOES-18 ABI channel 02."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .fetch import download_channel_02


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argparse formats message
        raise argparse.ArgumentTypeError(
            "scene time must be an ISO-8601 timestamp, e.g. 2023-07-01T12:30"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download GOES-18 ABI channel 02",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene-time",
        required=True,
        type=_parse_time,
        help="ISO-8601 time of the desired scan (UTC)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the NetCDF file",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    path = download_channel_02(args.scene_time, args.output_dir)
    print(f"Saved channel 02 granule to {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
