"""Synthetic CHASE days for the test-suite (no real data needed)."""
from __future__ import annotations

import os

import numpy as np
import pytest

CHANNELS = 118
CRVAL3 = 6559.40
CDELT3 = 0.0484
CORE = int(round((6562.83 - CRVAL3) / CDELT3))


def make_frame(path, crpix1, crpix2, t_sec, shape=(96, 96), seed=0, rsun=90.0,
               cdelt=1.04, cube=True):
    """Write one synthetic CHASE cube whose disk centre sits on ``CRPIX``."""
    from astropy.io import fits

    rng = np.random.default_rng(seed)
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - crpix1, yy - crpix2)
    disk = np.exp(-(rr / rsun) ** 4)
    # a couple of "filaments": elongated dark lanes fixed in solar coordinates
    lanes = np.zeros_like(disk)
    for x0, y0, sx, sy, amp in ((0.2, 0.3, 0.05, 0.35, 0.8), (-0.4, -0.1, 0.35, 0.04, 0.5)):
        cx = crpix1 + x0 * rsun
        cy = crpix2 + y0 * rsun
        lanes += amp * np.exp(-(((xx - cx) / (sx * rsun)) ** 2 +
                                ((yy - cy) / (sy * rsun)) ** 2))
    base = 100.0 + 900.0 * (disk - np.clip(lanes, 0, 1) * disk)
    hh, rem = divmod(int(t_sec), 3600)
    mm, ss = divmod(rem, 60)
    header = fits.Header()
    header["CRPIX1"] = crpix1
    header["CRPIX2"] = crpix2
    header["CDELT1"] = cdelt
    header["CDELT2"] = cdelt
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRVAL3"] = CRVAL3
    header["CDELT3"] = CDELT3
    header["RSUN_OBS"] = rsun * cdelt
    header["R_SUN"] = rsun
    header["DATE-OBS"] = f"2025-06-10T{hh:02d}:{mm:02d}:{ss:02d}"
    header["EXPTIME"] = 0.01
    if cube:
        data = np.empty((CHANNELS, ny, nx), dtype=np.int16)
        for c in range(CHANNELS):
            line = np.exp(-((c - CORE) ** 2) / (2 * 1.5 ** 2))
            data[c] = (base * (0.55 + 0.45 * line) +
                       rng.normal(0, 1.0, size=(ny, nx))).astype(np.int16)
    else:
        data = base.astype(np.int16)
    hdu = fits.PrimaryHDU()
    with fits.HDUList([hdu, fits.ImageHDU(data, header, name="SCI")]) as hdul:
        hdul.writeto(path, overwrite=True)
    return path


@pytest.fixture
def fake_day(tmp_path):
    """Five frames with an alternating pointing offset of 8 pixels."""
    root = tmp_path / "root"
    day = root / "2025" / "06" / "10"
    day.mkdir(parents=True)
    offsets = [(0.0, 0.0), (8.0, 6.0), (0.5, 0.2), (8.2, 6.1), (0.3, 0.1)]
    for i, (dx, dy) in enumerate(offsets):
        t = 3847 + 72 * i
        hh, rem = divmod(int(t), 3600)
        mm, ss = divmod(rem, 60)
        make_frame(str(day / f"RSM20250610T{hh:02d}{mm:02d}{ss:02d}_0000_HA.fits"),
                   64.0 + dx, 48.0 + dy, t, seed=i)
    return str(root)


@pytest.fixture
def fake_flat_day(tmp_path):
    """The same frames but flat in one directory (used by the scv adapter)."""
    day = tmp_path / "flat"
    day.mkdir()
    for i, (dx, dy) in enumerate([(0.0, 0.0), (7.0, 5.0), (0.2, 0.1)]):
        t = 3847 + 72 * i
        hh, rem = divmod(int(t), 3600)
        mm, ss = divmod(rem, 60)
        make_frame(str(day / f"RSM20250610T{hh:02d}{mm:02d}{ss:02d}_0000_HA.fits"),
                   64.0 + dx, 48.0 + dy, t, seed=10 + i)
    return str(day)
