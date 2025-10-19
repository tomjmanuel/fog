"""Command line interface for fog probability processing."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import default_config
from .fetch import SAN_FRANCISCO_SECTOR, SectorDefinition
from .probability import build_fog_probability

app = typer.Typer(help="GOES-18 fog probability utilities")
console = Console()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _sector_option(value: Optional[str]) -> SectorDefinition:
    if value is None:
        return SAN_FRANCISCO_SECTOR
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter("Sector must be 'west,south,east,north'")
    return SectorDefinition(*parts)


@app.command()
def fog(
    scene_time: str = typer.Argument(..., help="ISO8601 time of the desired scan"),
    sector: Optional[str] = typer.Option(None, help="Bounding box west,south,east,north"),
    cache_dir: Optional[Path] = typer.Option(None, help="Optional local cache directory"),
):
    cfg = default_config()
    if cache_dir:
        cfg.cache_dir = str(cache_dir)
    bbox = _sector_option(sector)
    console.log("Fetching GOES data...")
    p_fog, diagnostics = build_fog_probability(_parse_time(scene_time), bbox, cfg)
    console.log(f"Probability array shape: {p_fog.shape}")
    console.log(f"Diagnostics available: {diagnostics}")


if __name__ == "__main__":
    app()
