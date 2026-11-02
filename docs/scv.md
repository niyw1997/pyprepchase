# Using pyprepchase from solar-chromosphere-velocity

`solar-chromosphere-velocity` (`scv`) reads a CHASE cube and inverts it. Its input is a cube in
detector coordinates, so it inherits the per-frame pointing jitter. `pyprepchase` can be enabled as
an optional preprocessing step; it is off by default and has no hard dependency on `scv`.

## Option A: cache a corrected cube, then feed it to scv

```python
from pyprepchase.scv import prepare

info = prepare("RSM20250610T010407_0000_HA.fits", cache_dir="~/.cache/pyprepchase")
# info["path"] is a corrected FITS cube with CRPIX moved to the daily reference

from solar_chromosphere_velocity import ChromosphereConfig, ChromosphereProcessor
cfg = ChromosphereConfig(input_dir="", output_dir="out")
res = ChromosphereProcessor(cfg).process_file(info["path"])
```

The cache key includes the source file size, mtime and the applied shift, so a second call is a
dictionary lookup. `prepare()` never opens the source for writing.

## Option B: enable it inside scv

Build scv with the extra:

```bash
pip install -e ".[prep]"      # pulls pyprepchase
```

then

```python
cfg = ChromosphereConfig(input_dir="data", output_dir="out")
cfg.use_pyprepchase = True
cfg.pyprepchase_cache_dir = "~/.cache/pyprepchase"
```

or from the command line:

```bash
process-chase data out --use-pyprepchase --pyprepchase-cache ~/.cache/pyprepchase
```

If `pyprepchase` is not installed, `scv` logs a warning and continues with the raw file. The
preprocessed file name carries a hash of the model, so switching the reference grid or the CHASE
file does not silently reuse a stale cache.

## What the step changes

* the spatial `CRPIX1`/`CRPIX2` of the cube (moved to the daily median reference);
* the pixel data (cubic resampling by the per-frame `(dx, dy)`);
* nothing else: spectral WCS, units and data type are preserved.

## What it costs

One cube is 118 x 2313 x 2304 float32, about 2.4 GB in memory during the shift. Expect 20 to 60 s
per file on a laptop-class disk, a few seconds on a warm cache. The first call of a day also reads
the headers of all frames of that day (seconds).
