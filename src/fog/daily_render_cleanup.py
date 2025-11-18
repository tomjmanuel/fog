"""Utilities for removing outdated renders from the ``fog-app-renders`` bucket."""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Iterator, Sequence
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import sun
from rich.console import Console

from .aws import lazy_boto3

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
SUNSET_LOOKUP_LOCATION = LocationInfo(
    name="San Francisco",
    region="USA",
    timezone="America/Los_Angeles",
    latitude=37.7749,
    longitude=-122.4194,
)

TIMESTAMP_PATTERN = re.compile(r"(?P<ts>\d{8}T\d{6}Z)")

console = Console()


@dataclass(frozen=True)
class RenderObject:
    key: str
    timestamp_utc: datetime

    @property
    def timestamp_local(self) -> datetime:
        return self.timestamp_utc.astimezone(LOCAL_TZ)


def _parse_timestamp_from_key(key: str) -> datetime | None:
    match = TIMESTAMP_PATTERN.search(key.rsplit("/", 1)[-1])
    if not match:
        return None
    raw = match.group("ts")
    ts = datetime.strptime(raw, "%Y%m%dT%H%M%SZ")
    return ts.replace(tzinfo=timezone.utc)


def iter_daily_render_objects(*, bucket: str, prefix: str) -> Iterator[RenderObject]:
    boto3 = lazy_boto3()
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            timestamp = _parse_timestamp_from_key(key)
            if not timestamp:
                continue
            yield RenderObject(key=key, timestamp_utc=timestamp)


def _sunset_for_date(target_date: date) -> datetime:
    solar_data = sun(
        SUNSET_LOOKUP_LOCATION.observer,
        date=target_date,
        tzinfo=LOCAL_TZ,
    )
    return solar_data["sunset"]


def _cutoff_for_day(target_date: date, now: datetime | None = None) -> datetime:
    now_local = (now or datetime.now(tz=LOCAL_TZ)).astimezone(LOCAL_TZ)
    sunset = _sunset_for_date(target_date)
    anchor = min(now_local, sunset)
    return anchor - timedelta(hours=2)


def _group_days(renders: Iterable[RenderObject]) -> dict[date, list[RenderObject]]:
    buckets: dict[date, list[RenderObject]] = {}
    for render in renders:
        buckets.setdefault(render.timestamp_local.date(), []).append(render)
    return buckets


def determine_deletions(renders: Iterable[RenderObject], *, now: datetime | None = None) -> list[RenderObject]:
    by_day = _group_days(renders)
    if not by_day:
        return []

    ordered_days = sorted(by_day)
    latest_day = ordered_days[-1]
    if len(ordered_days) > 1:
        deletions: list[RenderObject] = []
        for day in ordered_days[:-1]:
            deletions.extend(by_day[day])
        return deletions

    cutoff = _cutoff_for_day(latest_day, now=now)
    return [render for render in by_day[latest_day] if render.timestamp_local < cutoff]


def delete_renders(renders: Sequence[RenderObject], *, bucket: str) -> None:
    if not renders:
        return
    boto3 = lazy_boto3()
    s3 = boto3.client("s3")
    batch: list[str] = []
    for render in renders:
        batch.append(render.key)
        if len(batch) == 1000:
            _delete_batch(s3, bucket, batch)
            batch.clear()
    if batch:
        _delete_batch(s3, bucket, batch)


def _delete_batch(s3_client, bucket: str, keys: Sequence[str]) -> None:
    s3_client.delete_objects(
        Bucket=bucket,
        Delete={"Objects": [{"Key": key} for key in keys]},
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove outdated daily renders from S3 based on sunset/age heuristics."
        )
    )
    parser.add_argument(
        "--bucket",
        default="fog-app-renders",
        help="S3 bucket containing the renders.",
    )
    parser.add_argument(
        "--prefix",
        default="daily/",
        help="Key prefix containing the renders (default: daily/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List keys that would be deleted without removing them.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    renders = list(iter_daily_render_objects(bucket=args.bucket, prefix=args.prefix))
    if not renders:
        console.log("No renders found; nothing to delete.")
        return 0

    to_delete = determine_deletions(renders)
    if not to_delete:
        console.log("All renders are within the retention window.")
        return 0

    console.log(f"Identified {len(to_delete)} renders to delete from {args.bucket}.")
    for render in to_delete:
        console.log(
            f"  {render.key} ({render.timestamp_local.isoformat()})"
        )

    if args.dry_run:
        console.log("Dry-run enabled; skipping deletion.")
        return 0

    delete_renders(to_delete, bucket=args.bucket)
    console.log("Deletion complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

