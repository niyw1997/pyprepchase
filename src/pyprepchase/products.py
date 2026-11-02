"""Sub-pixel resampling and product writers.

Input files are never modified.  Every product written here carries the shift
that was applied, so a downstream reader can reconstruct the original geometry.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import numpy as np


def shift_image(img: np.ndarray, dy: float, dx: float, order: int = 3) -> np.ndarray:
    """Translate an array's content by ``(dx, dy)`` pixels.

    Uses a spline of ``order`` (3 by default).  Integer and sub-pixel shifts take
    the same path so a series does not change character when a shift happens to be
    a whole pixel.
    """
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return img
    from scipy import ndimage as ndi

    return ndi.shift(img, shift=(dy, dx), order=order, mode="nearest", prefilter=True)


def shift_cube(cube: np.ndarray, dy: float, dx: float, order: int = 3) -> np.ndarray:
    """Translate the two spatial axes of a ``(channel, y, x)`` cube."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return cube
    from scipy import ndimage as ndi

    return ndi.shift(cube, shift=(0.0, dy, dx), order=order, mode="nearest", prefilter=True)


def crop_array(arr: np.ndarray, crop: Optional[Sequence[int]]) -> np.ndarray:
    """Crop ``(x0, y0, w, h)`` from the last two axes."""
    if crop is None:
        return arr
    x0, y0, w, h = (int(v) for v in crop)
    return arr[..., y0:y0 + h, x0:x0 + w]


def write_fits(path: str, data: np.ndarray, header: Dict[str, Any],
               history: Sequence[str] = (), overwrite: bool = False) -> str:
    """Write a float32 FITS product with an explicit WCS header."""
    from astropy.io import fits

    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    for key, value in header.items():
        if value is not None:
            hdu.header[key] = value
    for line in history:
        hdu.header["HISTORY"] = line
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    hdu.writeto(path, overwrite=overwrite)
    return path


def write_npz(path: str, data: np.ndarray, meta: Dict[str, Any]) -> str:
    """Write a compressed npz with the array plus flat metadata keys."""
    payload = {"data": np.asarray(data, dtype=np.float32)}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)):
            payload[key] = value
        else:
            payload[key] = np.array(value)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def product_header(frame, ref1: float, ref2: float, crop, channel: Optional[int],
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Minimal WCS for a corrected product (CHASE has no CTYPE/CROTA)."""
    x0, y0 = (0, 0) if crop is None else (int(crop[0]), int(crop[1]))
    header = dict(
        CRPIX1=ref1 - x0,
        CRPIX2=ref2 - y0,
        CRVAL1=0.0,
        CRVAL2=0.0,
        CDELT1=frame.cdelt1,
        CDELT2=frame.cdelt2,
        CUNIT1="arcsec",
        CUNIT2="arcsec",
        DATE_OBS=frame.date_obs,
        BUNIT="DN",
        ORIGIN="pyprepchase",
    )
    if channel is not None:
        header["CHANNEL"] = int(channel)
    if extra:
        header.update({k: v for k, v in extra.items() if v is not None})
    return header


def line_core_channel(frame, line: str = "HA") -> int:
    """Derive the line-core channel index from ``CRVAL3`` and ``CDELT3``."""
    from .config import LINE_WAVELENGTHS

    lam0 = LINE_WAVELENGTHS[line.upper()]
    if not np.isfinite(frame.crval3) or not np.isfinite(frame.cdelt3) or frame.cdelt3 == 0:
        raise ValueError("frame has no usable CRVAL3/CDELT3; pass an explicit channel")
    ch = int(round((lam0 - frame.crval3) / frame.cdelt3))
    if frame.naxis3 and not 0 <= ch < frame.naxis3:
        raise ValueError(f"line-core channel {ch} outside 0..{frame.naxis3 - 1}")
    return ch
