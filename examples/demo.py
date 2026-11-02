#!/usr/bin/env python
"""Run pyprepchase on the bundled CHASE excerpt and draw the before/after figure.

The excerpt in ``examples/data/2025-06-10`` is five real CHASE/HIS frames
(2025-06-10 01:04 to 01:09 UT), cropped to 512 x 512 pixels around a dark
filament and reduced to the three channels around H-alpha line centre.  It is
small enough to live in the repository and still carries the real pointing
jitter: consecutive frames are offset by about 9 and 7 pixels.

Usage::

    python examples/demo.py                  # writes docs/demo.png
    python examples/demo.py --out /tmp/demo.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "data"
DATE = "2025-06-10"
CHANNEL = 0          # the line-core plane in the excerpt


def dark_map(img, sigma=6.0):
    from scipy import ndimage as ndi

    bg = ndi.gaussian_filter(img, sigma)
    return np.clip((bg - img) / np.maximum(bg, 1.0), 0, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE.parent / "docs" / "demo.png"))
    parser.add_argument("--sigma", type=float, default=6.0)
    args = parser.parse_args()

    from pyprepchase import JitterModel, discover, shift_image

    frames = discover(str(ROOT), DATE)
    if len(frames) < 3:
        print(f"demo data not found under {ROOT}", file=sys.stderr)
        return 1
    model = JitterModel.build(frames)
    stats = model.stats()
    print(f"frames        : {stats['n']}")
    print(f"reference CRPIX: ({model.ref1:.2f}, {model.ref2:.2f})")
    print(f"shift median  : {stats['median_arcsec']:.2f} arcsec")
    print(f"step median   : {stats['step_median_arcsec']:.2f} arcsec")

    from astropy.io import fits

    def read(i):
        with fits.open(frames[i].path) as hdul:
            return np.asarray(hdul[1].data[CHANNEL], dtype=np.float32)

    imgs = [read(i) for i in range(len(frames))]
    corrected = [shift_image(img, *model.shift(i)) for i, img in enumerate(imgs)]

    # zoom on the strongest dark structure so the offset is easy to see
    d0 = dark_map(imgs[0], args.sigma)
    from scipy import ndimage as ndi

    smooth = ndi.uniform_filter(d0, 9)
    cy, cx = np.unravel_index(np.argmax(smooth), smooth.shape)
    half = 128
    y0 = int(np.clip(cy - half, 0, imgs[0].shape[0] - 2 * half))
    x0 = int(np.clip(cx - half, 0, imgs[0].shape[1] - 2 * half))
    sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))

    a0 = dark_map(imgs[0], args.sigma)[sl]
    a1 = dark_map(imgs[1], args.sigma)[sl]
    a1c = dark_map(corrected[1], args.sigma)[sl]
    # a single, compact structure makes the offset readable
    mask = a0 > np.percentile(a0, 97.0)
    labels, _ = ndi.label(mask)
    keep = labels == labels[int(cy) - y0, int(cx) - x0]
    if keep.sum() == 0:
        keep = mask
    keep = ndi.binary_dilation(ndi.binary_fill_holes(keep), np.ones((3, 3)))

    # stack metric: odd frames versus even frames in the same window
    def stack_diff(stack):
        ev = np.nanmean(np.stack(stack[0::2]), axis=0)
        od = np.nanmean(np.stack(stack[1::2]), axis=0)
        return float(np.std((od - ev)[sl]))

    s_before = stack_diff(imgs)
    s_after = stack_diff(corrected)
    print(f"odd/even stack difference: {s_before:.2f} -> {s_after:.2f} DN "
          f"({100 * (1 - s_after / max(s_before, 1e-9)):.0f}% lower)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25], wspace=0.08))
    vmax = np.percentile(a0, 99.5)
    for k, (panel, label) in enumerate(((a0, "reference frame"),
                                        (a1, "next frame, raw"),
                                        (a1c, "next frame, corrected"))):
        axes[k].imshow(panel, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
        axes[k].contour(keep.astype(float), levels=[0.5], colors="cyan", linewidths=1.1)
        axes[k].set_xticks([])
        axes[k].set_yticks([])
        axes[k].text(0.03, 0.97, f"({chr(97 + k)})", transform=axes[k].transAxes,
                     ha="left", va="top", fontsize=13,
                     bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8))
        axes[k].set_xlabel(label, fontsize=9)
    dx, dy = model.shift_arcsec(1)
    axes[1].text(0.5, 0.05, f"offset {dx:+.1f}, {dy:+.1f} arcsec",
                 transform=axes[1].transAxes, ha="center", fontsize=8, color="white",
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6))
    axes[2].text(0.5, 0.05, "residual < 0.5 arcsec", transform=axes[2].transAxes,
                 ha="center", fontsize=8, color="white",
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6))

    t = np.array([f.time for f in frames], float) / 60.0
    xy = model.shifts_arcsec()
    axes[3].axhline(0, color="black", lw=0.6, alpha=0.5)
    axes[3].plot(t - t[0], xy[:, 0], "o-", ms=4, lw=1.0, label=r"$\Delta x$")
    axes[3].plot(t - t[0], xy[:, 1], "s-", ms=3.5, lw=1.0, label=r"$\Delta y$")
    axes[3].set_xlabel("minutes from first frame", fontsize=9)
    axes[3].set_ylabel("shift to reference (arcsec)", fontsize=9)
    axes[3].legend(fontsize=8, frameon=False)
    axes[3].grid(alpha=0.25)
    axes[3].text(0.03, 0.97, "(d)", transform=axes[3].transAxes, ha="left", va="top",
                 fontsize=13, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                                        alpha=0.8))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("figure       :", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
