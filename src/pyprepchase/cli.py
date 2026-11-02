"""Command line interface: ``pyprep-chase``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .config import PrepConfig, resolve_outdir
from .frames import discover
from .jitter import JitterModel
from .limb import audit as limb_audit, audit_summary
from .pipeline import PrepPipeline


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default="", required=True,
                        help="CHASE Level-1 tree (or one day folder)")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--pattern", default="*_HA.fits", help="filename glob")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyprep-chase",
        description="Preprocess CHASE/HIS Level-1 cubes: pointing model, "
                    "sub-pixel registration, line-core extraction.")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="list frames and their geometry")
    _common(p)
    p.add_argument("--json", action="store_true", help="machine-readable output")

    p = sub.add_parser("model", help="build the per-frame pointing model")
    _common(p)
    p.add_argument("--reference", default="median", help="median | first | crpix:x,y")
    p.add_argument("--out", default="", help="directory for the model JSON")
    p.add_argument("--no-qc", action="store_true")

    p = sub.add_parser("audit", help="check CRPIX against the solar limb")
    _common(p)
    p.add_argument("--n", type=int, default=0, help="frames to sample (0 = auto)")

    p = sub.add_parser("apply", help="write corrected cubes/images")
    _common(p)
    p.add_argument("--channels", default="", help="comma separated, e.g. 68,69,70")
    p.add_argument("--crop", default="", help="x0,y0,w,h")
    p.add_argument("--outdir", default="./prep")
    p.add_argument("--format", default="fits", choices=["fits", "npz"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--line-core", action="store_true",
                   help="derive the channel from CRVAL3/CDELT3")
    p.add_argument("--half-width", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0)

    p = sub.add_parser("run", help="full pipeline (model + audit + products + manifest)")
    _common(p)
    p.add_argument("--channels", default="")
    p.add_argument("--crop", default="")
    p.add_argument("--outdir", default="./prep")
    p.add_argument("--format", default="fits", choices=["fits", "npz"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--line-core", action="store_true")
    p.add_argument("--half-width", type=int, default=0)
    p.add_argument("--no-audit", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--config", default="", help="JSON/TOML config file")

    return parser


def _config_from_args(args) -> PrepConfig:
    if getattr(args, "config", ""):
        cfg = PrepConfig.load(args.config)
        cfg.root = args.root
        cfg.date = args.date
        cfg.pattern = args.pattern
    else:
        cfg = PrepConfig(root=args.root, date=args.date, pattern=args.pattern)
    if getattr(args, "channels", ""):
        cfg.channels = tuple(int(c) for c in args.channels.split(",") if c.strip())
    if getattr(args, "crop", ""):
        cfg.output.crop = tuple(int(v) for v in args.crop.split(","))
    if getattr(args, "outdir", ""):
        cfg.output.outdir = args.outdir
    if getattr(args, "format", ""):
        cfg.output.format = args.format
    if getattr(args, "overwrite", False):
        cfg.output.overwrite = True
    if getattr(args, "line_core", False):
        cfg.line_core.enabled = True
    if getattr(args, "half_width", 0):
        cfg.line_core.half_width = int(args.half_width)
    if getattr(args, "reference", ""):
        cfg.jitter.reference = args.reference
    if getattr(args, "start", 0):
        cfg.start = int(args.start)
    if getattr(args, "count", 0):
        cfg.count = int(args.count)
    return cfg


def cmd_index(args) -> int:
    frames = discover(args.root, args.date, args.pattern)
    if not frames:
        print(f"no frames under {args.root} for {args.date}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps([f.to_dict() for f in frames], indent=1))
        return 0
    print(f"{'i':>4} {'time':>9} {'CRPIX1':>9} {'CRPIX2':>9} {'CDELT':>6} "
          f"{'CRVAL3':>10} {'RSUN':>8}  file")
    for f in frames:
        hh, rem = divmod(int(f.time), 3600)
        mm, ss = divmod(rem, 60)
        print(f"{f.index:4d} {hh:02d}:{mm:02d}:{ss:02d} {f.crpix1:9.2f} {f.crpix2:9.2f} "
              f"{f.cdelt1:6.2f} {f.crval3:10.3f} {f.rsun_obs:8.2f}  {os.path.basename(f.path)}")
    return 0


def cmd_model(args) -> int:
    frames = discover(args.root, args.date, args.pattern)
    if not frames:
        print(f"no frames under {args.root} for {args.date}", file=sys.stderr)
        return 1
    model = JitterModel.build(frames, reference=args.reference)
    stats = model.stats()
    print(f"[{args.date}] {stats['n']} frames, reference CRPIX="
          f"({model.ref1:.2f}, {model.ref2:.2f})")
    print("  shift: median {:.2f}\", p95 {:.2f}\", max {:.2f}\"; "
          "step median {:.2f}\"".format(stats["median_arcsec"], stats["p95_arcsec"],
                                        stats["max_arcsec"], stats["step_median_arcsec"]))
    outdir = resolve_outdir(args.out or args.root)
    path = os.path.join(outdir, f"jitcal_{args.date.replace('-', '')}.json")
    model.save(path)
    print("  model ->", path)
    if not args.no_qc:
        png = os.path.join(outdir, f"jitcal_{args.date.replace('-', '')}.png")
        if PrepPipeline(PrepConfig(root=args.root, date=args.date)).qc_figure(model, [], png):
            print("  qc    ->", png)
    return 0


def cmd_audit(args) -> int:
    frames = discover(args.root, args.date, args.pattern)
    if not frames:
        print(f"no frames under {args.root} for {args.date}", file=sys.stderr)
        return 1
    model = JitterModel.build(frames)
    rows = limb_audit(model, n_frames=(args.n or None))
    print(f"{'i':>4} {'time':>9} {'limb_cx':>9} {'limb_cy':>9} {'r':>8} "
          f"{'limb-CRPIX x':>12} {'y':>7}")
    for r in rows:
        hh, rem = divmod(int(r["time"]), 3600)
        mm, _ = divmod(rem, 60)
        print(f"{r['index']:4d} {hh:02d}:{mm:02d}     {r['cx']:9.2f} {r['cy']:9.2f} "
              f"{r['r']:8.2f} {r['dx']:12.2f} {r['dy']:7.2f}")
    summary = audit_summary(rows)
    print("  residual scatter: std_x={std_dx:.3f} px, std_y={std_dy:.3f} px "
          "(1 px = {:.2f}\")".format(abs(frames[0].cdelt1), **summary))
    return 0


def cmd_apply(args) -> int:
    cfg = _config_from_args(args)
    cfg.output.save_qc = False
    pipeline = PrepPipeline(cfg)
    model = pipeline.model()
    outdir = resolve_outdir(cfg.output.outdir)
    start = max(int(cfg.start), 0)
    stop = len(model) if cfg.count <= 0 else min(start + int(cfg.count), len(model))
    for i in range(start, stop):
        rec = pipeline.process_frame(i, model, outdir)
        print(f"  [{i:3d}] {os.path.basename(rec['source'])[:28]} "
              f"shift=({rec['shift_px'][0]:+.2f},{rec['shift_px'][1]:+.2f}) px -> "
              f"{os.path.basename(rec['product'])}")
    print(f"{len(model)} products in {outdir}")
    return 0


def cmd_run(args) -> int:
    cfg = _config_from_args(args)
    manifest = PrepPipeline(cfg).run_day(do_audit=not args.no_audit)
    stats = manifest["stats"]
    print(f"[{manifest['date']}] {stats['n']} frames -> {manifest['outdir']}")
    print("  shift median {:.2f}\", p95 {:.2f}\"".format(stats["median_arcsec"],
                                                        stats["p95_arcsec"]))
    if manifest["audit"]["summary"]:
        s = manifest["audit"]["summary"]
        print("  limb audit: std_x={:.2f} px, std_y={:.2f} px ({} frames)".format(
            s["std_dx"], s["std_dy"], s["n"]))
    print("  manifest ->", manifest["manifest"])
    if manifest.get("qc"):
        print("  qc       ->", manifest["qc"])
    return 0


def _version() -> str:
    from . import __version__

    return __version__


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = dict(index=cmd_index, model=cmd_model, audit=cmd_audit,
                    apply=cmd_apply, run=cmd_run)
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
