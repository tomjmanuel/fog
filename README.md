# Minimal GOES-18 Downloader (SF sector)

Download GOES-18 ABI Level-1b channel 2 data for the bay area and save it locally as NetCDF.
Project the data to be viewed from directly above my house. 
Upsample it to 20 m per voxel and overlay it on a high res base image.

## Install

python -m venv .venv
source .venv/bin/activate
pip install -e .

## Command line interface for downloading a specific time

python -m fog.cli \
  --scene-time 2023-07-01T12:30:00 \
  --output-dir ./data

This will save a NetCDF file for channel C02 in the specified directory.

### Daily render pipeline

Generate a cloud + fog overlay on the base image for the current time and save
them to `./renders`:

python -m fog.render_now

### Render all times from a day

To iterate across every daylight scene (sunrise → sunset) in San Francisco for
a given day (defaults to today), run the helper script (it simply reuses the renderer above in a loop):

python render_all_of_today.py --interval-minutes 10

## Project Structure
src/fog/
  __init__.py      # Public API (downloader)
  config.py        # GOES configuration
  fetch.py         # Data access + subsetting + download function
  cli.py           # CLI entry point
  render_now.py    # CLI utility to render composites for the current day
  rendering.py     # Render functions
  s3_uploader.py   # functions for uploading to s3 bucket

