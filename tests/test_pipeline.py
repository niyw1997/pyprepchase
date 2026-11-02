import json
import os

import numpy as np
import pytest

from pyprepchase.config import PrepConfig
from pyprepchase.frames import discover
from pyprepchase.jitter import JitterModel
from pyprepchase.pipeline import PrepPipeline


def config_for(fake_day, tmp_path, **kwargs):
    cfg = PrepConfig(root=fake_day, date="2025-06-10")
    cfg.output.outdir = str(tmp_path / "out")
    cfg.channels = kwargs.pop("channels", (70,))
    cfg.output.save_qc = kwargs.pop("save_qc", False)
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def test_channels_from_line_core(fake_day, tmp_path):
    cfg = config_for(fake_day, tmp_path, channels=())
    cfg.line_core.enabled = True
    cfg.line_core.half_width = 2
    pipe = PrepPipeline(cfg)
    model = pipe.model()
    assert pipe.channels_for(model.frames[0]) == [69, 70, 71, 72, 73]


def test_explicit_channel_override(fake_day, tmp_path):
    cfg = config_for(fake_day, tmp_path, channels=(10, 70))
    pipe = PrepPipeline(cfg)
    model = pipe.model()
    assert pipe.channels_for(model.frames[0]) == [10, 70]


def test_run_day_manifest(fake_day, tmp_path):
    cfg = config_for(fake_day, tmp_path)
    manifest = PrepPipeline(cfg).run_day(do_audit=False)
    assert manifest["stats"]["n"] == 5
    assert manifest["date"] == "2025-06-10"
    assert len(manifest["products"]) == 5
    assert os.path.exists(manifest["manifest"])
    for product in manifest["products"]:
        assert os.path.exists(product["product"])
    with open(manifest["manifest"]) as fh:
        on_disk = json.load(fh)
    assert on_disk["package"] == "pyprepchase"
    assert len(on_disk["flags"]) == 5


def test_run_day_shifts_to_reference(fake_day, tmp_path):
    from astropy.io import fits

    cfg = config_for(fake_day, tmp_path)
    manifest = PrepPipeline(cfg).run_day(do_audit=False)
    model = PrepPipeline(cfg).model()
    for i, product in enumerate(manifest["products"][:2]):
        with fits.open(product["product"]) as hdul:
            hdr = hdul[0].header
        assert hdr["CRPIX1"] == pytest.approx(model.ref1)
        assert hdr["CRPIX2"] == pytest.approx(model.ref2)


def test_run_day_npz(fake_day, tmp_path):
    cfg = config_for(fake_day, tmp_path)
    cfg.output.format = "npz"
    manifest = PrepPipeline(cfg).run_day(do_audit=False)
    data = np.load(manifest["products"][0]["product"])
    assert data["data"].shape == (96, 96)
    assert data["channels"].tolist() == [70]


def test_run_day_crop(fake_day, tmp_path):
    cfg = config_for(fake_day, tmp_path)
    cfg.output.crop = (10, 20, 30, 40)
    manifest = PrepPipeline(cfg).run_day(do_audit=False)
    data = np.load(manifest["products"][0]["product"]) if False else None
    from astropy.io import fits
    with fits.open(manifest["products"][0]["product"]) as hdul:
        assert hdul[0].data.shape == (40, 30)
        assert hdul[0].header["CRPIX1"] == pytest.approx(
            PrepPipeline(cfg).model().ref1 - 10)


def test_flags_mark_repoint(fake_day, tmp_path):
    frames = discover(fake_day, date="2025-06-10")
    model = JitterModel.build(frames)
    cfg = config_for(fake_day, tmp_path)
    cfg.jitter.max_step_arcsec = 5.0
    pipe = PrepPipeline(cfg)
    flags = pipe.flags(model, [])
    kinds = [f["flags"] for f in flags]
    assert kinds[0] == ["first"]
    # the synthetic day alternates by ~10 arcsec, so every later frame is a jump
    assert "repoint" in kinds[1]
    assert "repoint" in kinds[2]
    # with a generous threshold nothing is a repoint
    cfg.jitter.max_step_arcsec = 30.0
    calm = PrepPipeline(cfg).flags(model, [])
    assert all("repoint" not in rec["flags"] for rec in calm)


def test_missing_data_raises(tmp_path):
    cfg = PrepConfig(root=str(tmp_path), date="2030-01-01")
    cfg.output.outdir = str(tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        PrepPipeline(cfg).run_day()
