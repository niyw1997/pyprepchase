"""Configuration objects for :mod:`pyprepchase`.

Everything is a plain dataclass so it can be built in code, from JSON, or from the
command line.  Nested options are kept small and explicit on purpose: a
preprocessing step that silently changes the science products is worse than no
preprocessing at all.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple

#: Rest wavelengths (angstrom) of the CHASE/HIS science lines.
LINE_WAVELENGTHS = {"HA": 6562.83, "FE": 6569.22}

#: Default suffix of CHASE Level-1 spectral cubes.
DEFAULT_GLOB = "*_HA.fits"


@dataclass
class JitterOptions:
    """How the pointing model is built and applied.

    Attributes:
        enabled: Apply the jitter correction at all.
        reference: ``median`` (default), ``first``, or ``crpix:x,y``.
        mode: ``shift`` (translate only) or ``none`` (record, do not resample).
        order: Spline order for the sub-pixel shift (3 = cubic).
        max_step_arcsec: Consecutive-frame steps above this are flagged as
            repointing rather than jitter.
        audit_fraction: Fraction of frames to check against the limb circle.
        audit_min_frames: Lower bound for the audit sample size.
    """

    enabled: bool = True
    reference: str = "median"
    mode: str = "shift"
    order: int = 3
    max_step_arcsec: float = 20.0
    audit_fraction: float = 0.03
    audit_min_frames: int = 5


@dataclass
class LineCoreOptions:
    """Line-core extraction for the 118-channel CHASE cubes.

    Attributes:
        enabled: Collapse the cube to a line-core image.
        line: ``HA`` or ``FE``.
        half_width: Number of channels on each side of the core to average.
            ``0`` means the single core channel.
        channel: Override the channel index (``None`` = derive from CRVAL3).
    """

    enabled: bool = False
    line: str = "HA"
    half_width: int = 0
    channel: Optional[int] = None


@dataclass
class OutputOptions:
    """Where and how products are written."""

    outdir: str = "~/chase/prep"
    format: str = "fits"  # fits | npz
    crop: Optional[Tuple[int, int, int, int]] = None  # x0, y0, w, h
    overwrite: bool = False
    save_qc: bool = True
    manifest_name: str = "manifest.json"


@dataclass
class PrepConfig:
    """Top-level configuration.

    Attributes:
        root: Directory with the CHASE Level-1 day folders.
        date: Day in ``YYYY-MM-DD``.
        pattern: Filename pattern of the cubes.
        channels: Explicit channel list (overrides ``line_core``).
        start: First frame to process (the model still uses the whole day).
        count: Number of frames to process; ``0`` means all remaining.
        jitter: Pointing-model options.
        line_core: Line-core options.
        output: Output options.
        hdu: HDU holding the spectral cube (1 for CHASE Level-1).
    """

    root: str = "/data2/Chase/Lev1"
    date: str = ""
    pattern: str = DEFAULT_GLOB
    channels: Tuple[int, ...] = ()
    start: int = 0
    count: int = 0
    jitter: JitterOptions = field(default_factory=JitterOptions)
    line_core: LineCoreOptions = field(default_factory=LineCoreOptions)
    output: OutputOptions = field(default_factory=OutputOptions)
    hdu: int = 1

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrepConfig":
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        for key, klass in (("jitter", JitterOptions), ("line_core", LineCoreOptions),
                           ("output", OutputOptions)):
            if isinstance(data.get(key), dict):
                sub = {k: v for k, v in data[key].items()
                       if k in {f.name for f in fields(klass)}}
                data[key] = klass(**sub)
        if "channels" in data and data["channels"] is not None:
            data["channels"] = tuple(int(c) for c in data["channels"])
        return cls(**data)

    @classmethod
    def load(cls, path: str) -> "PrepConfig":
        """Load from a JSON or TOML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if path.endswith(".toml"):
            import tomllib  # Python >= 3.11

            with open(path, "rb") as fh:
                return cls.from_dict(tomllib.load(fh))
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, ensure_ascii=False)
        return path

    # --------------------------------------------------------------- validation
    def validate(self) -> None:
        if self.jitter.mode not in ("shift", "none"):
            raise ValueError(f"jitter.mode must be 'shift' or 'none', got {self.jitter.mode!r}")
        if self.jitter.reference not in ("median", "first") and not self.jitter.reference.startswith("crpix:"):
            raise ValueError("jitter.reference must be 'median', 'first' or 'crpix:x,y'")
        if self.line_core.line.upper() not in LINE_WAVELENGTHS:
            raise ValueError(f"unknown line {self.line_core.line!r}")
        if self.output.format not in ("fits", "npz"):
            raise ValueError("output.format must be 'fits' or 'npz'")
        if self.output.crop is not None and len(self.output.crop) != 4:
            raise ValueError("output.crop must be (x0, y0, w, h)")


def resolve_outdir(path: str) -> str:
    """Expand ``~`` and make the directory if needed."""
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    return path
