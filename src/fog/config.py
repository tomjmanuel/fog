"""Configuration helpers for GOES-18 fog processing."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Mapping, Tuple


@dataclass(slots=True, frozen=True)
class GOESConfig:
    """Runtime configuration for fetching and processing GOES scenes.

    Attributes
    ----------
    product : str
        ABI product short name. Defaults to the mesoscale FD G16 key.
    bucket : str
        Cloud bucket containing GOES L1b files.
    region : str
        GOES scan mode / region identifier (e.g. "M6" for mode 6, "F" for full-disk).
    channels : tuple[str, ...]
        ABI channel identifiers required for the fog algorithm.
    cache_dir : str | None
        Optional local cache directory for downloaded granules.
    max_concurrent : int
        Maximum concurrent fetches when reading remote files.
    timeout : int
        Timeout in seconds for remote object reads.
    preload_minutes : int
        Search tolerance around the requested scene time when locating
        a granule (some scans can straddle times).
    """

    product: str = "ABI-L1b-RadC"
    bucket: str = "noaa-goes18"
    region: str = "M6"
    channels: Tuple[str, ...] = ("C02",)
    cache_dir: str | None = None
    max_concurrent: int = 4
    timeout: int = 300
    preload_minutes: int = 10
    additional_products: None = None
    # additional_products: Tuple[str, ...] = (
    #     "ABI-L2-MCMIPC",  # cloud mask
    #     "ABI-L2-CMIPF",  # cloud phase
    #     "ABI-L2-LWPRad",  # LWP
    # )

    def channel_list(self) -> List[str]:
        return list(self.channels)

    def product_prefixes(self) -> Iterable[str]:
        yield self.product
        # yield from self.additional_products

    def valid_time_window(self, scene_time: datetime) -> Tuple[datetime, datetime]:
        scene_time = scene_time.astimezone(timezone.utc)
        delta = timedelta(minutes=self.preload_minutes)
        return scene_time - delta, scene_time + delta

    def object_key_prefix(
        self, scene_time: datetime, product: str | None = None, region: str | None = None
    ) -> str:
        scene_time = scene_time.astimezone(timezone.utc)
        product = product or self.product
        prefix = f"{product}/{scene_time:%Y/%j/%H}/OR_{product}-"
        region = region or self.region
        if region:
            prefix += region
        return prefix


def default_config() -> GOESConfig:
    """Return a default configuration tailored for San Francisco fog."""
    return GOESConfig()


@dataclass(slots=True, frozen=True)
class OverlayConfig:
    """Configuration for overlaying high resolution base imagery."""

    bounding_box: tuple[float, float, float, float]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float] | Mapping[str, object]) -> "OverlayConfig":
        try:
            lon_min = float(mapping["lon_min"])
            lon_max = float(mapping["lon_max"])
            lat_min = float(mapping["lat_min"])
            lat_max = float(mapping["lat_max"])
        except Exception as exc:  # pragma: no cover - validation path
            raise ValueError(
                "OverlayConfig mapping must contain lon_min, lon_max, lat_min, lat_max"
            ) from exc
        if lon_min >= lon_max or lat_min >= lat_max:
            raise ValueError("Bounding box must have lon_min < lon_max and lat_min < lat_max")
        return cls((lon_min, lon_max, lat_min, lat_max))


def load_overlay_config(path: Path) -> OverlayConfig:
    """Load overlay configuration from a JSON file."""

    data = json.loads(Path(path).read_text())
    bbox = data.get("bounding_box")
    if not isinstance(bbox, dict):
        raise ValueError("Overlay config must define a 'bounding_box' object")
    return OverlayConfig.from_mapping(bbox)


__all__ = ["GOESConfig", "OverlayConfig", "default_config", "load_overlay_config"]
