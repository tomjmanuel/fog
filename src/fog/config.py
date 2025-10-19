"""Configuration helpers for GOES-18 fog processing."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Tuple


@dataclass(slots=True)
class GOESConfig:
    """Runtime configuration for fetching and processing GOES scenes.

    Attributes
    ----------
    product : str
        ABI product short name. Defaults to the mesoscale FD G16 key.
    bucket : str
        Cloud bucket containing GOES L1b files.
    region : str
        GOES region identifier ("F", "M1", "M2").
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
    region: str = "M1"
    channels: Tuple[str, ...] = ("C02", "C07", "C14")
    cache_dir: str | None = None
    max_concurrent: int = 4
    timeout: int = 300
    preload_minutes: int = 10
    additional_products: Tuple[str, ...] = (
        "ABI-L2-MCMIPC",  # cloud mask
        "ABI-L2-CMIPF",  # cloud phase
        "ABI-L2-LWPRad",  # LWP
    )

    def channel_list(self) -> List[str]:
        return list(self.channels)

    def product_prefixes(self) -> Iterable[str]:
        yield self.product
        yield from self.additional_products

    def valid_time_window(self, scene_time: datetime) -> Tuple[datetime, datetime]:
        scene_time = scene_time.astimezone(timezone.utc)
        delta = timedelta(minutes=self.preload_minutes)
        return scene_time - delta, scene_time + delta

    def object_key_prefix(
        self, scene_time: datetime, product: str | None = None, region: str | None = None
    ) -> str:
        scene_time = scene_time.astimezone(timezone.utc)
        product = product or self.product
        region = region or self.region
        return f"{product}/{scene_time:%Y/%j/%H}/OR_{product}-G18"


def default_config() -> GOESConfig:
    """Return a default configuration tailored for San Francisco fog."""
    return GOESConfig()


__all__ = ["GOESConfig", "default_config"]
