"""Per-frame pointing model built from the CHASE ``CRPIX`` cards.

The CHASE Level-1 cubes are delivered in detector coordinates: the solar disk
moves by several pixels from frame to frame (up to ~14 arcsec on a normal day),
and ``CRPIX1/CRPIX2`` move with it.  Ignoring that, any fixed-pixel analysis
sees a sawtooth of the same size.  This module turns the cards into an explicit
shift model that can be recorded, inspected and applied.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .frames import Frame


@dataclass
class Shift:
    """Shift of one frame, in detector pixels."""

    index: int
    time: float
    dx: float
    dy: float

    def arcsec(self, cdelt: float) -> Tuple[float, float]:
        return self.dx * cdelt, self.dy * cdelt

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.dx, self.dy))


class JitterModel:
    """Translate each frame onto a common reference grid.

    ``shift`` follows ``scipy.ndimage.shift`` semantics: the returned ``(dy, dx)``
    is what has to be applied to the frame so that its disk centre lands on the
    reference ``CRPIX``.
    """

    def __init__(self, frames: Sequence[Frame], ref1: float, ref2: float,
                 reference: str = "median"):
        self.frames: List[Frame] = list(frames)
        self.ref1 = float(ref1)
        self.ref2 = float(ref2)
        self.reference = reference
        self.shifts: List[Shift] = [
            Shift(index=f.index, time=f.time,
                  dx=self.ref1 - f.crpix1,
                  dy=self.ref2 - f.crpix2)
            for f in self.frames
        ]

    # ------------------------------------------------------------------ builders
    @classmethod
    def build(cls, frames: Sequence[Frame], reference: str = "median") -> "JitterModel":
        """Build the model; ``reference`` is ``median``, ``first`` or ``crpix:x,y``."""
        if not frames:
            raise ValueError("no frames to model")
        xs = np.array([f.crpix1 for f in frames], float)
        ys = np.array([f.crpix2 for f in frames], float)
        if reference == "median":
            ref1, ref2 = float(np.median(xs)), float(np.median(ys))
        elif reference == "first":
            ref1, ref2 = float(xs[0]), float(ys[0])
        elif reference.startswith("crpix:"):
            ref1, ref2 = (float(v) for v in reference.split(":", 1)[1].split(","))
        else:
            raise ValueError(f"unknown reference {reference!r}")
        return cls(frames, ref1, ref2, reference=reference)

    # ------------------------------------------------------------------- queries
    def __len__(self) -> int:
        return len(self.frames)

    def shift(self, i: int) -> Tuple[float, float]:
        s = self.shifts[i]
        return s.dy, s.dx

    def shift_arcsec(self, i: int) -> Tuple[float, float]:
        f = self.frames[i]
        return self.shifts[i].arcsec(abs(f.cdelt1))

    def cdelt(self) -> float:
        return abs(self.frames[0].cdelt1) if self.frames else 1.04

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict:
        """Summary of the jitter for one day (all angles in arcsec)."""
        cdelt = self.cdelt()
        dx = np.array([s.dx for s in self.shifts]) * cdelt
        dy = np.array([s.dy for s in self.shifts]) * cdelt
        t = np.array([s.time for s in self.shifts], float)
        mag = np.hypot(dx, dy)
        step = np.hypot(np.diff(dx), np.diff(dy))
        dt = np.diff(t)
        return dict(
            n=len(self.shifts),
            span_hours=float((t[-1] - t[0]) / 3600.0) if len(t) > 1 else 0.0,
            dt_median=float(np.median(dt)) if len(dt) else float("nan"),
            median_arcsec=float(np.median(mag)),
            p95_arcsec=float(np.percentile(mag, 95)),
            max_arcsec=float(mag.max()),
            step_median_arcsec=float(np.median(step)) if len(step) else float("nan"),
            step_p95_arcsec=float(np.percentile(step, 95)) if len(step) else float("nan"),
            step_max_arcsec=float(step.max()) if len(step) else float("nan"),
            rsun_span_arcsec=float(np.nanmax([f.rsun_obs for f in self.frames])
                                   - np.nanmin([f.rsun_obs for f in self.frames])),
            ac1=float(np.corrcoef(dx[:-1], dx[1:])[0, 1]) if len(dx) > 3 else float("nan"),
            ac2=float(np.corrcoef(dx[:-2], dx[2:])[0, 1]) if len(dx) > 4 else float("nan"),
        )

    # ---------------------------------------------------------------- (de)serial
    def to_dict(self) -> dict:
        return dict(reference=self.reference, ref_crpix=[self.ref1, self.ref2],
                    cdelt=self.cdelt(), frames=[f.to_dict() for f in self.frames],
                    shifts=[s.__dict__ for s in self.shifts], stats=self.stats())

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str) -> "JitterModel":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        frames = [Frame(**f) for f in data["frames"]]
        model = cls(frames, *data["ref_crpix"], reference=data.get("reference", "median"))
        model.shifts = [Shift(**s) for s in data["shifts"]]
        return model

    def shifts_arcsec(self) -> np.ndarray:
        cdelt = self.cdelt()
        return np.array([[s.dx * cdelt, s.dy * cdelt] for s in self.shifts])
