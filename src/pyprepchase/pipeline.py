"""The preprocessing pipeline: index, model, audit, flags, products, manifest."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import PrepConfig, resolve_outdir
from .frames import Frame, discover, read_header
from .jitter import JitterModel
from .limb import audit as limb_audit, audit_summary
from .products import (crop_array, line_core_channel, product_header, shift_cube,
                       shift_image, write_fits, write_npz)
from .quality import frame_flags


class PrepPipeline:
    """Run the CHASE preprocessing for one day.

    Example::

        cfg = PrepConfig(root="/data2/Chase/Lev1", outdir="~/chase/prep")
        cfg.line_core.enabled = True
        manifest = PrepPipeline(cfg).run_day("2025-06-10")
    """

    def __init__(self, config: PrepConfig):
        config.validate()
        self.config = config
        self._frames: Optional[List[Frame]] = None

    # --------------------------------------------------------------------- steps
    def frames(self) -> List[Frame]:
        if self._frames is None:
            self._frames = discover(self.config.root, self.config.date,
                                    self.config.pattern)
        return self._frames

    def model(self) -> JitterModel:
        frames = self.frames()
        if not frames:
            raise FileNotFoundError(
                f"no frames matching {self.config.pattern!r} under {self.config.root!r}"
                + (f" for {self.config.date}" if self.config.date else ""))
        return JitterModel.build(frames, reference=self.config.jitter.reference)

    def audit(self, model: Optional[JitterModel] = None,
              n_frames: Optional[int] = None) -> List[dict]:
        model = model or self.model()
        if n_frames is None:
            n_frames = max(self.config.jitter.audit_min_frames,
                           int(round(self.config.jitter.audit_fraction * len(model))))
        return limb_audit(model, n_frames=n_frames)

    def flags(self, model: JitterModel, audit_rows: Sequence[dict]) -> List[dict]:
        return frame_flags(model, max_step_arcsec=self.config.jitter.max_step_arcsec,
                           audit_rows=audit_rows)

    # ------------------------------------------------------------------ products
    def channels_for(self, frame: Frame) -> List[int]:
        """Channels to read: explicit list, else the line core (+/- half width)."""
        if self.config.channels:
            return [int(c) for c in self.config.channels]
        opts = self.config.line_core
        core = opts.channel if opts.channel is not None else line_core_channel(frame, opts.line)
        hw = max(int(opts.half_width), 0)
        return list(range(max(core - hw, 0), core + hw + 1))

    def process_frame(self, i: int, model: JitterModel, outdir: str,
                      audit_rows: Sequence[dict] = ()) -> Dict[str, Any]:
        frame = model.frames[i]
        channels = self.channels_for(frame)
        arrays = []
        for ch in channels:
            from .frames import read_plane

            arrays.append(read_plane(frame.path, channel=ch, hdu=self.config.hdu))
        cube = np.stack(arrays)
        dy, dx = model.shift(i) if self.config.jitter.enabled else (0.0, 0.0)
        if self.config.jitter.mode == "shift":
            cube = shift_cube(cube, dy, dx, order=self.config.jitter.order)
        cube = crop_array(cube, self.config.output.crop)
        data = cube[0] if len(channels) == 1 else cube

        stem = os.path.basename(frame.path).replace(".fits", "")
        crop = self.config.output.crop
        history = [
            f"pyprepchase {self._version()}",
            f"shift_pix=({dy:+.4f},{dx:+.4f}) from CRPIX",
            f"reference_crpix=({model.ref1:.4f},{model.ref2:.4f})",
            f"channels={','.join(str(c) for c in channels)}",
        ]
        if self.config.output.format == "fits":
            path = os.path.join(outdir, stem + "_prep.fits")
            write_fits(path, data,
                       product_header(frame, model.ref1, model.ref2, crop,
                                      channels[0] if len(channels) == 1 else None,
                                      extra=dict(NDCHAN=len(channels))),
                       history=history, overwrite=self.config.output.overwrite)
        else:
            path = os.path.join(outdir, stem + "_prep.npz")
            write_npz(path, data, dict(channels=channels, shift_dy=dy, shift_dx=dx,
                                       ref_crpix1=model.ref1, ref_crpix2=model.ref2,
                                       date=frame.date_obs, source=frame.path))
        return dict(index=i, time=frame.time, source=frame.path, product=path,
                    channels=channels, shift_px=[dx, dy],
                    shift_arcsec=list(model.shift_arcsec(i)))

    # --------------------------------------------------------------------- driver
    def run_day(self, date: Optional[str] = None, do_audit: bool = True) -> Dict[str, Any]:
        if date:
            self.config.date = date
            self._frames = None
        steps: List[Dict[str, Any]] = []

        def step(name: str, t0: float, info: Optional[Dict[str, Any]] = None) -> float:
            steps.append(dict(name=name, seconds=round(time.perf_counter() - t0, 3),
                              **(info or {})))
            return time.perf_counter()

        t0 = time.perf_counter()
        model = self.model()
        t0 = step("index+model", t0, dict(n=len(model)))

        outdir = resolve_outdir(self.config.output.outdir)
        audit_rows = []
        if do_audit and self.config.jitter.audit_fraction > 0:
            audit_rows = self.audit(model)
        t0 = step("audit", t0, dict(frames=len(audit_rows)))

        flags = self.flags(model, audit_rows)
        t0 = step("flags", t0, dict(frames=len(flags)))

        start = max(int(self.config.start), 0)
        stop = len(model) if self.config.count <= 0 else min(start + int(self.config.count),
                                                            len(model))
        products = []
        for i in range(start, stop):
            products.append(self.process_frame(i, model, outdir, audit_rows))
        t0 = step("products", t0, dict(written=len(products), first=start, last=max(stop - 1, start)))

        manifest: Dict[str, Any] = dict(
            package="pyprepchase",
            version=self._version(),
            date=self.config.date or model.frames[0].date_obs[:10],
            root=self.config.root,
            outdir=outdir,
            config=self.config.to_dict(),
            stats=model.stats(),
            audit=dict(rows=audit_rows, summary=audit_summary(audit_rows)),
            flags=flags,
            products=products,
            steps=steps,
        )
        manifest_path = os.path.join(outdir, self.config.output.manifest_name)
        steps.append(dict(name="manifest", seconds=0.0))
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, ensure_ascii=False)
        manifest["manifest"] = manifest_path

        if self.config.output.save_qc:
            qc = os.path.join(outdir, "qc_jitter.png")
            if self.qc_figure(model, audit_rows, qc):
                manifest["qc"] = qc
        return manifest

    # ------------------------------------------------------------------------ QC
    def qc_figure(self, model: JitterModel, audit_rows: Sequence[dict], path: str) -> bool:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - optional dependency
            return False

        from matplotlib import font_manager

        have = {f.name for f in font_manager.fontManager.ttflist}
        for name in ("WenQuanYi Micro Hei", "Noto Sans CJK SC", "PingFang SC"):
            if name in have:
                matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                break
        matplotlib.rcParams["axes.unicode_minus"] = False

        t = np.array([f.time for f in model.frames], float) / 3600.0
        xy = model.shifts_arcsec()
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6),
                                 gridspec_kw=dict(width_ratios=[2.0, 1.0], wspace=0.25))
        axes[0].plot(t, xy[:, 0], ".", ms=2.6, color="#14508c", label=r"$\Delta x$")
        axes[0].plot(t, xy[:, 1], ".", ms=2.6, color="#aa481a", label=r"$\Delta y$")
        axes[0].set_xlabel("UT (h)")
        axes[0].set_ylabel("per-frame shift (arcsec)")
        axes[0].legend(fontsize=8, frameon=False)
        axes[0].grid(alpha=0.25)
        if audit_rows:
            axes[1].plot([r["index"] for r in audit_rows], [r["dx"] for r in audit_rows],
                         "o", ms=4, color="#14508c", label=r"$\Delta x$")
            axes[1].plot([r["index"] for r in audit_rows], [r["dy"] for r in audit_rows],
                         "s", ms=3.5, color="#aa481a", label=r"$\Delta y$")
            axes[1].set_xlabel("frame index")
            axes[1].set_ylabel("limb - CRPIX (px)")
            axes[1].legend(fontsize=8, frameon=False)
            axes[1].grid(alpha=0.25)
        else:
            mag = np.hypot(xy[:, 0], xy[:, 1])
            axes[1].hist(mag, bins=40, color="#14508c", alpha=0.85)
            axes[1].set_xlabel("shift magnitude (arcsec)")
            axes[1].set_ylabel("frames")
            axes[1].grid(alpha=0.25)
        for ax, label in zip(axes, "ab"):
            ax.text(0.02, 0.97, f"({label})", transform=ax.transAxes, ha="left", va="top",
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                           ec="none", alpha=0.75))
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fig.tight_layout()
            except Exception:  # pragma: no cover - matplotlib layout corner case
                pass
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return True

    # -------------------------------------------------------------------- helpers
    @staticmethod
    def _version() -> str:
        from . import __version__

        return __version__


def run_day(date: str, config: Optional[PrepConfig] = None, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper: :class:`PrepPipeline` + :meth:`PrepPipeline.run_day`."""
    cfg = config or PrepConfig(date=date)
    cfg.date = date
    return PrepPipeline(cfg).run_day(**kwargs)
