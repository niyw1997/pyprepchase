"""Solar-limb circle fitting, used to audit the header pointing model.

The limb is an absolute geometric reference: its centre does not depend on solar
activity, so it can tell whether ``CRPIX`` really follows the image.  On the
CHASE data tested so far the residual ``limb - CRPIX`` is constant to 0.10 to
0.21 pixel across a day.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .frames import Frame, read_plane
from .jitter import JitterModel


def fit_circle(x: Sequence[float], y: Sequence[float], n_iter: int = 6,
               clip: float = 3.0) -> Dict[str, float]:
    """Robust algebraic (Kasa) circle fit with sigma clipping.

    Returns ``cx, cy, r, rms, n_points``.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.ones(x.size, bool)
    cx = cy = r = 0.0
    for _ in range(n_iter):
        xx, yy = x[keep], y[keep]
        if xx.size < 50:
            break
        a = np.c_[2 * xx, 2 * yy, np.ones(xx.size)]
        b = xx ** 2 + yy ** 2
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-6)))
        res = np.hypot(xx - cx, yy - cy) - r
        med = np.median(res)
        mad = np.median(np.abs(res - med)) * 1.4826
        idx = np.where(keep)[0]
        new_keep = np.zeros_like(keep)
        new_keep[idx[np.abs(res - med) < max(clip * mad, 0.05)]] = True
        if new_keep.sum() == keep.sum() or new_keep.sum() < 50:
            keep = new_keep
            break
        keep = new_keep
    res = np.hypot(x[keep] - cx, y[keep] - cy) - r
    return dict(cx=cx, cy=cy, r=r, rms=float(np.sqrt((res ** 2).mean())),
                n_points=int(keep.sum()))


def limb_center(img: np.ndarray, downsample: int = 2, threshold: float = 0.45,
                smooth: float = 2.0) -> Dict[str, float]:
    """Detect the solar disk edge and fit a circle.

    Args:
        img: 2D intensity image (disk brighter than sky).
        downsample: Bin factor for speed; the result is scaled back.
        threshold: Fraction between sky and the 99.5th percentile.
        smooth: Gaussian sigma in binned pixels.

    Returns:
        ``cx, cy, r`` in full-resolution pixels plus fit diagnostics.
    """
    from scipy import ndimage as ndi

    ds = max(int(downsample), 1)
    small = ndi.gaussian_filter(img[::ds, ::ds].astype(np.float32, copy=False), smooth)
    border = np.r_[small[:8, :].ravel(), small[-8:, :].ravel(),
                   small[:, :8].ravel(), small[:, -8:].ravel()]
    sky = float(np.median(border))
    hi = float(np.percentile(small, 99.5))
    mask = small > sky + threshold * (hi - sky)
    mask = ndi.binary_fill_holes(mask)
    labels, n = ndi.label(mask)
    if n > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        mask = labels == int(np.argmax(counts))
    mask = ndi.binary_erosion(mask, np.ones((5, 5)))
    edge = ndi.binary_dilation(mask, np.ones((3, 3))) & ~mask
    yy, xx = np.where(edge)
    fit = fit_circle(xx, yy)
    return dict(cx=fit["cx"] * ds, cy=fit["cy"] * ds, r=fit["r"] * ds,
                rms=fit["rms"] * ds, n_points=fit["n_points"],
                sky=sky, threshold=threshold)


def frame_limb(frame: Frame, channel: int = 70, hdu: int = 1, **kwargs) -> Dict[str, float]:
    """Limb fit for one frame."""
    img = read_plane(frame.path, channel=channel, hdu=hdu)
    return limb_center(img, **kwargs)


def audit(model: JitterModel, n_frames: Optional[int] = None, channel: int = 70,
          hdu: int = 1, **kwargs) -> List[dict]:
    """Compare ``limb - CRPIX`` on a sample of frames spread over the day."""
    frames = model.frames
    if not frames:
        return []
    if n_frames is None:
        n_frames = max(5, int(round(0.03 * len(frames))))
    n_frames = min(max(int(n_frames), 1), len(frames))
    idx = np.unique(np.linspace(0, len(frames) - 1, n_frames).round().astype(int))
    out: List[dict] = []
    for i in idx:
        f = frames[int(i)]
        fit = frame_limb(f, channel=channel, hdu=hdu, **kwargs)
        out.append(dict(index=int(i), time=f.time, file=os.path.basename(f.path),
                        cx=fit["cx"], cy=fit["cy"], r=fit["r"],
                        dx=fit["cx"] - f.crpix1, dy=fit["cy"] - f.crpix2,
                        drift_r=fit["r"] - f.r_sun if np.isfinite(f.r_sun) else float("nan"),
                        rms=fit["rms"], n_points=fit["n_points"]))
    return out


def audit_summary(rows: Sequence[dict]) -> Dict[str, float]:
    """Aggregate an audit table: the scatter is the calibration accuracy."""
    if not rows:
        return {}
    dx = np.array([r["dx"] for r in rows], float)
    dy = np.array([r["dy"] for r in rows], float)
    return dict(n=len(rows), mean_dx=float(dx.mean()), mean_dy=float(dy.mean()),
                std_dx=float(dx.std()), std_dy=float(dy.std()),
                range_dx=float(dx.max() - dx.min()), range_dy=float(dy.max() - dy.min()))
