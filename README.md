# Minimal GOES-18 Downloader (SF sector)

This repository contains a tiny utility to download GOES-18 ABI Level-1b channel 2 data
for the San Francisco sector and save it locally as NetCDF.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
python -m fog.cli \
  --scene-time 2023-07-01T12:30:00 \
  --output-dir ./data
```

Options:
- `--sector`: override bounding box as `west,south,east,north` if needed. Defaults to SF sector.

This will save a NetCDF file for channel C02 in the specified directory.

### Visualizer overlay configuration

When plotting the fog visualizations you can overlay channel 2 radiance on a
high-resolution base image by supplying an overlay configuration JSON file:

```bash
python -m fog.visualize \
  --scene-time 2023-07-01T12:30:00 \
  --overlay-config examples/channel2_overlay_config.json \
  --base-image /path/to/high_res_base.png
```

An example overlay configuration is provided in
`examples/channel2_overlay_config.json`. The bounding box corresponds to a
high-resolution image spanning the latitude/longitude corners `(37.268569,
-122.984661)` and `(38.390362, -121.731963)`.

## Project Structure

```
src/fog/
  __init__.py      # Public API (downloader)
  config.py        # GOES configuration
  fetch.py         # Data access + subsetting + download function
  cli.py           # CLI entry point
```
