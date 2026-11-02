"""Per-frame quality flags.

Flags never change the pixels; they tell downstream code which frames to trust,
to bridge, or to drop.  They are written into the manifest so a reprocessing run
reproduces the same selection.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .jitter import JitterModel


def frame_flags(model: JitterModel, max_step_arcsec: float = 20.0,
                gap_factor: float = 3.0,
                audit_rows: Optional[Sequence[dict]] = None,
                audit_tolerance: float = 1.0) -> List[Dict[str, object]]:
    """Return one flag record per frame.

    Flags:
        ``gap_before``: the previous frame is more than ``gap_factor`` times the
            median cadence earlier (orbit gap, not a frame-to-frame pair).
        ``repoint``: the step arriving at this frame exceeds ``max_step_arcsec``;
            treat it as a deliberate repoint, not jitter.
        ``limb_mismatch``: the limb audit disagrees with ``CRPIX`` by more than
            ``audit_tolerance`` pixels (bad header or a fast transient).
        ``first`` / ``last``: day boundaries, useful when building baselines.
    """
    frames = model.frames
    n = len(frames)
    if n == 0:
        return []
    t = np.array([f.time for f in frames], float)
    cdelt = model.cdelt()
    step = np.hypot(*np.diff(model.shifts_arcsec(), axis=0).T) if n > 1 else np.array([])
    dt = np.diff(t)
    cadence = float(np.median(dt)) if len(dt) else 0.0

    mismatch = {}
    if audit_rows:
        for row in audit_rows:
            if np.hypot(row["dx"], row["dy"]) > audit_tolerance:
                mismatch[int(row["index"])] = float(np.hypot(row["dx"], row["dy"]))

    flags: List[Dict[str, object]] = []
    for i, f in enumerate(frames):
        rec: Dict[str, object] = dict(index=i, time=f.time,
                                      file=f.path.rsplit("/", 1)[-1], flags=[])
        if i == 0:
            rec["flags"].append("first")
        if i == n - 1 and n > 1:
            rec["flags"].append("last")
        if i > 0 and cadence > 0 and dt[i - 1] > gap_factor * cadence:
            rec["flags"].append("gap_before")
            rec["gap_seconds"] = float(dt[i - 1])
        if i > 0 and i - 1 < len(step) and step[i - 1] > max_step_arcsec:
            rec["flags"].append("repoint")
            rec["step_arcsec"] = float(step[i - 1])
        if i in mismatch:
            rec["flags"].append("limb_mismatch")
            rec["limb_offset_px"] = mismatch[i]
        rec["shift_arcsec"] = [float(v) for v in model.shifts_arcsec()[i]]
        flags.append(rec)
    return flags


def usable(flags: Sequence[Dict[str, object]], drop: Sequence[str] = ("repoint",)) -> List[int]:
    """Indices of frames that carry none of the ``drop`` flags."""
    out = []
    for rec in flags:
        if not (set(drop) & set(rec["flags"])):
            out.append(int(rec["index"]))
    return out
