"""Optional adapter so ``solar-chromosphere-velocity`` can use pyprepchase.

The integration is deliberately lazy and cache-based:

* scv only imports this module when its ``preprocess`` option is enabled;
* the corrected cube is written once per (file, model) into a cache directory;
* scv then loads the cached FITS like any other input.

If pyprepchase is not installed, :func:`available` returns ``False`` and scv can
fall back to its own loader with a warning.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

import numpy as np

from .jitter import JitterModel
from .frames import discover, read_header
from .products import crop_array, shift_cube


def available() -> bool:
    """True when the optional runtime dependencies are importable."""
    try:
        import astropy  # noqa: F401
        import scipy  # noqa: F401
    except Exception:
        return False
    return True


def _cache_key(path: str, model: JitterModel, i: int) -> str:
    stat = os.stat(path)
    raw = f"{os.path.abspath(path)}|{stat.st_size}|{int(stat.st_mtime)}|" \
          f"{model.ref1:.4f}|{model.ref2:.4f}|{model.shift(i)[0]:.4f}|{model.shift(i)[1]:.4f}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def prepare(path: str, cache_dir: str = "~/chase/prep_cache", hdu: int = 1,
            channels: Optional[list] = None, crop=None, overwrite: bool = False,
            verbose: bool = False) -> Dict[str, Any]:
    """Return a pointing-corrected copy of one CHASE cube.

    Args:
        path: Original ``_HA.fits`` (read-only, never modified).
        cache_dir: Where corrected cubes are cached.
        hdu: HDU with the spectral cube.
        channels: Channels to keep (``None`` = all).
        crop: Optional ``(x0, y0, w, h)``.
        overwrite: Recompute even if the cache is fresh.

    Returns:
        ``dict(path=..., shift_px=[dx, dy], cached=bool, reference_crpix=[...])``
    """
    from astropy.io import fits

    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    frames = discover(os.path.dirname(path), date="")
    if not frames:
        raise FileNotFoundError(path)
    model = JitterModel.build(frames, reference="median")
    index = next((f.index for f in frames if os.path.abspath(f.path) == os.path.abspath(path)), None)
    if index is None:
        raise FileNotFoundError(f"{path} not in its own directory listing")
    frame = model.frames[index]
    key = _cache_key(path, model, index)
    stem = os.path.basename(path).replace(".fits", "")
    out = os.path.join(cache_dir, f"{stem}_{key}_prep.fits")
    if os.path.exists(out) and not overwrite:
        return dict(path=out, cached=True, shift_px=list(model.shift(index)),
                    reference_crpix=[model.ref1, model.ref2])

    with fits.open(path, memmap=False, mode="readonly") as hdul:
        data = np.asarray(hdul[hdu].data, dtype=np.float32)
        head = hdul[hdu].header.copy()
    if channels is not None:
        data = data[[int(c) for c in channels]]
    dy, dx = model.shift(index)
    data = shift_cube(data, dy, dx)
    data = crop_array(data, crop)

    head["CRPIX1"] = model.ref1 - (0 if crop is None else crop[0])
    head["CRPIX2"] = model.ref2 - (0 if crop is None else crop[1])
    head.add_history(f"pyprepchase: jitter shift (dy,dx)=({dy:+.4f},{dx:+.4f}) px")
    head.add_history(f"pyprepchase: reference CRPIX=({model.ref1:.4f},{model.ref2:.4f})")
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32), head).writeto(out, overwrite=True)
    if verbose:
        print(f"[pyprepchase] {os.path.basename(path)} -> {out} shift=({dx:+.2f},{dy:+.2f}) px")
    return dict(path=out, cached=False, shift_px=list(model.shift(index)),
                reference_crpix=[model.ref1, model.ref2])


def describe(path: str) -> Dict[str, Any]:
    """Model summary for one file (useful for logging from scv)."""
    frames = discover(os.path.dirname(path), date="")
    model = JitterModel.build(frames, reference="median")
    index = next((f.index for f in frames if os.path.abspath(f.path) == os.path.abspath(path)), 0)
    return dict(n_frames=len(model), shift_px=list(model.shift(index)),
                shift_arcsec=list(model.shift_arcsec(index)),
                reference_crpix=[model.ref1, model.ref2], stats=model.stats())
