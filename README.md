# Minimal GOES-18 Downloader (SF sector)

Download GOES-18 ABI Level-1b channel 2 data for the bay area and save it locally as NetCDF.
Project the data to be viewed from directly above my house. 
Upsample it to 1 voxel per kilometer and overlay it on a high res base image.

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
  --input-dir /path/to/downloaded/data/\
  --base-image /path/to/high_res_base.png
```

## Project Structure

```
src/fog/
  __init__.py      # Public API (downloader)
  config.py        # GOES configuration
  fetch.py         # Data access + subsetting + download function
  cli.py           # CLI entry point
```
