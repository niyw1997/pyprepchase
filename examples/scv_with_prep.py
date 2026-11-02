#!/usr/bin/env python
"""Feed a jitter-corrected CHASE cube to solar-chromosphere-velocity.

Usage::

    python examples/scv_with_prep.py RSM20250610T010407_0000_HA.fits

Requires both packages: ``pip install -e .`` here and ``pip install -e .`` in scv.
"""
import sys

from pyprepchase.scv import describe, prepare

path = sys.argv[1]
print("model:", describe(path))
info = prepare(path, cache_dir="~/.cache/pyprepchase", verbose=True)
print("corrected cube:", info["path"], "cached:", info["cached"])

from solar_chromosphere_velocity import ChromosphereConfig, ChromosphereProcessor

cfg = ChromosphereConfig(input_dir="", output_dir="scv_out")
result = ChromosphereProcessor(cfg).process_file(info["path"])
print("scv keys:", sorted(result))
