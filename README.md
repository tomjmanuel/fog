# fog

A tiny helper that downloads a single GOES-18 ABI channel 02 granule for a
requested scene time.

## Usage

Install the package (or run with `python -m fog.cli`) and provide the scene time
and an output directory:

```bash
python -m fog.cli --scene-time 2023-07-01T12:30 --output-dir ./data
```

The command downloads the first available C02 granule for the hour containing
`scene-time` and saves it as a NetCDF file in the given directory.
