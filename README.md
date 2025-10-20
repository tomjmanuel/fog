# GOES-18 Channel 02 Downloader

A tiny utility for fetching the GOES-18 ABI Level-1b channel 02 granule nearest a
requested scene time and saving it locally as NetCDF.

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

This command saves a single `C02` NetCDF file in the specified directory.
