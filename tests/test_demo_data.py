"""The bundled real CHASE excerpt must keep working as a regression fixture."""
from __future__ import annotations

import os

import numpy as np
import pytest

from pyprepchase import PrepConfig, PrepPipeline, discover

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data")

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(DATA, "2025-06-10")),
                                reason="bundled demo data missing")


def test_demo_frames_are_real():
    frames = discover(DATA, "2025-06-10")
    assert len(frames) == 5
    assert frames[0].naxis1 == 512 and frames[0].naxis2 == 512
    assert frames[0].line == "HA"
    # the jitter is preserved by the excerpt
    from pyprepchase import JitterModel

    model = JitterModel.build(frames)
    assert model.stats()["step_median_arcsec"] > 5.0


def test_demo_pipeline_runs(tmp_path):
    from astropy.io import fits

    cfg = PrepConfig(root=DATA, date="2025-06-10")
    cfg.output.outdir = str(tmp_path)
    cfg.channels = (0,)
    cfg.output.save_qc = False
    manifest = PrepPipeline(cfg).run_day(do_audit=False)
    assert len(manifest["products"]) == 5
    model = PrepPipeline(cfg).model()
    with fits.open(manifest["products"][1]["product"]) as hdul:
        assert hdul[0].header["CRPIX1"] == pytest.approx(model.ref1)
        assert hdul[0].data.shape == (512, 512)
    # products are finite and not constant
    with fits.open(manifest["products"][1]["product"]) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
    assert np.isfinite(data).all()
    assert data.std() > 0


def test_demo_correction_reduces_stack_difference():
    """The scientific point of the package, checked on the bundled real data."""
    from astropy.io import fits

    from pyprepchase import JitterModel, shift_image

    frames = discover(DATA, "2025-06-10")
    model = JitterModel.build(frames)
    raw = []
    for f in frames:
        with fits.open(f.path) as hdul:
            raw.append(np.asarray(hdul[1].data[0], dtype=np.float32))
    cor = [shift_image(img, *model.shift(i)) for i, img in enumerate(raw)]

    def stack_diff(stack):
        even = np.mean(np.stack(stack[0::2]), axis=0)
        odd = np.mean(np.stack(stack[1::2]), axis=0)
        return float(np.std(odd - even))

    before, after = stack_diff(raw), stack_diff(cor)
    assert after < 0.5 * before
