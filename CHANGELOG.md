# Changelog

## 0.1.0 (2026-11-03)

First release.

- `JitterModel`: per-frame `(dx, dy)` from `CRPIX1`/`CRPIX2`, reference `median` (default),
  `first` or `crpix:x,y`; JSON round-trip; statistics (median/p95/max, steps, autocorrelation).
- `limb`: robust circle fit and `limb - CRPIX` audit, with a summary of the scatter.
- `products`: cubic sub-pixel shift, optional crop, WCS rewrite, FITS/npz writers.
- `line_core_channel`: derive the core channel from `CRVAL3`/`CDELT3` (H-alpha and Fe I).
- `quality`: `gap_before`, `repoint`, `limb_mismatch`, `first`/`last` flags.
- `PrepPipeline.run_day`: products + `manifest.json` (config, stats, audit, flags) + QC figure.
- CLI `pyprep-chase`: `index`, `model`, `audit`, `apply`, `run`.
- Per-step timing recorded in `manifest["steps"]`.
- `pyprepchase.scv`: optional adapter for solar-chromosphere-velocity, with a hash-keyed cache.
- 52 tests on synthetic CHASE cubes; CI on Python 3.9, 3.11 and 3.12.
