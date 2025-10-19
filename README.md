# GOES-18 Fog Probability Toolkit

This repository provides a starting point for working with GOES-18 Advanced Baseline Imager (ABI)
Level-1b/Level-2 datasets to estimate the probability of fog over the San Francisco Bay Area.
It focuses on efficient data access, geospatial subsetting, and a modular implementation of
a fog-probability algorithm inspired by NOAA's low stratus/fog product.

## Features

- Minimal configuration layer for selecting GOES products, channels, and geographic sectors.
- Lazy, on-demand access to the NOAA GOES-18 S3 archive using `s3fs` and `xarray`.
- Geographic subsetting helpers tuned for the San Francisco coastline.
- Projection utilities and high-resolution grid generation for enhanced visualization.
- A direct translation of the provided fog probability pseudo-code with placeholders for lookup tables.
- CLI entrypoint (`fog-cli fog <scene-time>`) to fetch data and compute fog probabilities for a given scene.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the optional development dependencies for linting/testing:

```bash
pip install -e .[dev]
```

## Usage

```bash
fog-cli fog 2023-07-01T12:30:00
```

Use the `--sector` option to specify a custom bounding box (`west,south,east,north`).
A `--cache-dir` option is available to point to a writable directory for caching remote granules.

## Project Structure

```
src/fog/
    __init__.py             # Public API
    config.py               # GOES configuration dataclass
    fetch.py                # Data access helpers (S3 + subsetting)
    projection.py           # Projection, grid, and resampling utilities
    probability.py          # Fog probability algorithm + diagnostics
    cli.py                  # Typer-based CLI entrypoint
```

## Next Steps

The lookup tables used by the probability algorithm are placeholders that return constant values.
To make the product operational, replace `load_LUT` with logic that loads precomputed tables from disk
or a database. Additional validation (quality flags, land/sea masking) can be layered on top of the
provided scaffolding.
