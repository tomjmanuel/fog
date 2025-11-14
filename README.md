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

### Daily render pipeline

Generate both low- and high-resolution composites for the current day and save
them to `./renders`:

```bash
python -m fog.render_daily \
  --data-dir ./data \
  --render-dir ./renders \
  --base-image San_Francisco_Bay.jpg \
```

The command downloads the requested GOES scene (defaults to “now”), renders two
PNGs that reuse the dual-panel layout from `fog.visualize`, and stores them
under `renders/YYYY-MM-DD/`. Helper functions for uploading the resulting files
to S3/CloudFront live in `fog.s3_uploader` (they are implemented but not yet
invoked by the CLI).

### Batch daylight rendering

To iterate across every daylight scene (sunrise → sunset) in San Francisco for
today, run the helper script (it simply reuses the renderer above in a loop):

```bash
python render_all_of_today.py --interval-minutes 10
```

Override the date or base images with the corresponding CLI flags if you want to
re-run a prior day or experiment with different overlays.

## Project Structure

```
src/fog/
  __init__.py      # Public API (downloader)
  config.py        # GOES configuration
  fetch.py         # Data access + subsetting + download function
  cli.py           # CLI entry point
  render_daily.py  # CLI utility to render composites for the current day

```
