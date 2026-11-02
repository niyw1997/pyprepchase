"""Frame discovery and read-only FITS access for CHASE Level-1 data."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

#: One CHASE Level-1 spectral cube, e.g. ``RSM20250610T010407_0000_HA.fits``.
FNAME_RE = re.compile(r"RSM(\d{8})T(\d{2})(\d{2})(\d{2})_(\d+)_([A-Za-z]+)\.fits$")


@dataclass
class Frame:
    """Geometry and provenance of one CHASE frame.

    All angles are degrees, all pixel quantities are in detector pixels.
    ``time`` is seconds since 00:00 UT of the observation day.
    """

    index: int
    time: float
    path: str
    date_obs: str = ""
    crpix1: float = 0.0
    crpix2: float = 0.0
    cdelt1: float = 1.04
    cdelt2: float = 1.04
    crval3: float = float("nan")
    cdelt3: float = float("nan")
    rsun_obs: float = float("nan")
    r_sun: float = float("nan")
    naxis1: int = 0
    naxis2: int = 0
    naxis3: int = 0
    exptime: float = float("nan")
    line: str = ""

    @property
    def arcsec_per_pixel(self) -> float:
        return abs(self.cdelt1)

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def _date_dirs(root: str, date: str) -> List[str]:
    """Candidate directories for one day, in priority order."""
    y, m, d = (int(part) for part in date.split("-"))
    return [
        os.path.join(root, f"{y:04d}", f"{m:02d}", f"{d:02d}"),
        os.path.join(root, f"{y:04d}", str(m), str(d)),
        os.path.join(root, date),
        root,
    ]


def _parse_time(date_obs: str, filename: str) -> float:
    """Seconds since 00:00 UT; falls back to the timestamp in the filename."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S"):
        try:
            t = datetime.strptime(date_obs.strip(), fmt)
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
        except (ValueError, AttributeError):
            continue
    match = FNAME_RE.search(os.path.basename(filename))
    if match:
        hh, mm, ss = (int(match.group(i)) for i in (2, 3, 4))
        return hh * 3600 + mm * 60 + ss
    raise ValueError(f"cannot determine time of {filename!r} (DATE-OBS={date_obs!r})")


def read_header(path: str, hdu: int = 1) -> Dict[str, object]:
    """Read the WCS cards of one frame without touching the pixel data."""
    from astropy.io import fits

    with fits.open(path, memmap=True, mode="readonly") as hdul:
        hdr = hdul[hdu].header
        get = hdr.get
        return dict(
            crpix1=float(get("CRPIX1", 0.0)),
            crpix2=float(get("CRPIX2", 0.0)),
            cdelt1=abs(float(get("CDELT1", 1.04))),
            cdelt2=abs(float(get("CDELT2", 1.04))),
            crval3=float(get("CRVAL3", np.nan)),
            cdelt3=float(get("CDELT3", np.nan)),
            rsun_obs=float(get("RSUN_OBS", np.nan)),
            r_sun=float(get("R_SUN", np.nan)),
            naxis1=int(get("NAXIS1", 0)),
            naxis2=int(get("NAXIS2", 0)),
            naxis3=int(get("NAXIS3", 0)),
            exptime=float(get("EXPTIME", np.nan)),
            date_obs=str(get("DATE-OBS", "")).strip(),
        )


def read_plane(path: str, channel: int = 70, hdu: int = 1) -> np.ndarray:
    """Read one channel of a cube (or the whole HDU0 image) as float32."""
    from astropy.io import fits

    with fits.open(path, memmap=False, mode="readonly") as hdul:
        if hdu == 0:
            return np.asarray(hdul[0].data, dtype=np.float32)
        return np.asarray(hdul[hdu].data[channel], dtype=np.float32)


def discover(root: str, date: str = "", pattern: str = "*_HA.fits") -> List[Frame]:
    """Find and describe the frames of one day.

    Args:
        root: Directory holding the CHASE tree (or one day, or one flat folder).
        date: ``YYYY-MM-DD``.  Ignored when *root* itself contains the FITS files.
        pattern: Filename glob.

    Returns:
        Frames sorted by time.  Empty list when nothing is found.
    """
    from fnmatch import fnmatch

    candidates = _date_dirs(root, date) if date else [root]
    day_dir = None
    names: List[str] = []
    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        names = sorted(n for n in os.listdir(cand) if fnmatch(n, pattern))
        if names:
            day_dir = cand
            break
    if day_dir is None:
        return []

    frames: List[Frame] = []
    for i, name in enumerate(names):
        path = os.path.join(day_dir, name)
        try:
            hdr = read_header(path)
        except Exception:  # pragma: no cover - unreadable file
            continue
        try:
            t = _parse_time(hdr["date_obs"], name)
        except ValueError:
            match = FNAME_RE.search(name)
            if not match:
                continue
            t = int(match.group(2)) * 3600 + int(match.group(3)) * 60 + int(match.group(4))
        line = name.rsplit("_", 1)[-1].replace(".fits", "").upper()
        frames.append(Frame(index=i, time=t, path=path, line=line, **hdr))
    frames.sort(key=lambda f: f.time)
    for i, f in enumerate(frames):
        f.index = i
    return frames
