"""Command line wrapper for downloading GOES-18 channel 02."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .fetch import download_channel_02


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download GOES-18 channel 02 for a given scene time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene-time",
        required=True,
        type=datetime.fromisoformat,
        help="ISO-8601 time of the desired scan (UTC)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the NetCDF file will be saved",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = download_channel_02(args.scene_time, args.output_dir)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
