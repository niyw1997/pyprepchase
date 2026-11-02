#!/usr/bin/env python
"""Preprocess one CHASE day and print the manifest summary.

Usage::

    python examples/prep_day.py /path/to/Chase/Lev1 2025-06-10 ./prep
"""
import sys

from pyprepchase import PrepConfig, PrepPipeline

root = sys.argv[1] if len(sys.argv) > 1 else "/path/to/Chase/Lev1"
date = sys.argv[2] if len(sys.argv) > 2 else "2025-06-10"
outdir = sys.argv[3] if len(sys.argv) > 3 else "./prep"

cfg = PrepConfig(root=root, date=date)
cfg.output.outdir = outdir
cfg.line_core.enabled = True
cfg.line_core.half_width = 2
cfg.start = 0
cfg.count = 25          # a segment; use count=0 for the whole day

manifest = PrepPipeline(cfg).run_day(do_audit=True)
print("frames :", manifest["stats"]["n"])
print("median :", round(manifest["stats"]["median_arcsec"], 2), "arcsec")
print("audit  :", manifest["audit"]["summary"])
print("manifest:", manifest["manifest"])
