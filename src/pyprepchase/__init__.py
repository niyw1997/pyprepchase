"""pyprepchase: preprocessing for CHASE/HIS Level-1 data.

The package turns a directory of CHASE `_HA.fits` / `_FE.fits` spectral cubes into
analysis-ready products:

* a per-frame pointing model from the `CRPIX1/CRPIX2` header cards,
* sub-pixel resampling onto a common reference grid,
* optional line-core extraction,
* provenance metadata and quick-look QC figures.

Quick start::

    from pyprepchase import PrepConfig, PrepPipeline

    cfg = PrepConfig(root="/path/to/Chase/Lev1")
    cfg.output.outdir = "./prep"
    PrepPipeline(cfg).run_day("2025-06-10")

Command line::

    pyprep-chase run --date 2025-06-10 --channels 68,69,70,71,72

Inputs are never modified: every FITS file is opened read-only and every derived
product goes to ``config.output.outdir``.
"""
from __future__ import annotations

from .config import PrepConfig, JitterOptions, LineCoreOptions, OutputOptions
from .frames import Frame, discover, read_header, read_plane
from .jitter import JitterModel
from .limb import audit, fit_circle, limb_center
from .pipeline import PrepPipeline, run_day
from .quality import frame_flags
from .products import shift_image

__version__ = "0.1.0"

__all__ = [
    "PrepConfig",
    "JitterOptions",
    "LineCoreOptions",
    "OutputOptions",
    "Frame",
    "discover",
    "read_header",
    "read_plane",
    "JitterModel",
    "audit",
    "fit_circle",
    "limb_center",
    "frame_flags",
    "shift_image",
    "PrepPipeline",
    "run_day",
    "__version__",
]
