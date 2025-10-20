"""Minimal GOES-18 downloader for SF sector."""

from .config import GOESConfig
from .fetch import (
    SAN_FRANCISCO_SECTOR,
    SectorDefinition,
    download_channels,
)

__all__ = [
    "GOESConfig",
    "SAN_FRANCISCO_SECTOR",
    "SectorDefinition",
    "download_channels",
]
