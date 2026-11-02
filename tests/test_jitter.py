import os

import numpy as np
import pytest

from pyprepchase.frames import discover, read_header, read_plane
from pyprepchase.jitter import JitterModel


def test_discover_finds_day(fake_day):
    frames = discover(fake_day, date="2025-06-10")
    assert len(frames) == 5
    assert [f.index for f in frames] == list(range(5))
    assert frames[0].time == pytest.approx(3847)
    assert frames[1].time - frames[0].time == pytest.approx(72)
    assert frames[0].crpix1 == pytest.approx(64.0)
    assert frames[1].crpix1 == pytest.approx(72.0)
    assert frames[0].line == "HA"


def test_discover_flat_dir(fake_flat_day):
    frames = discover(fake_flat_day)
    assert len(frames) == 3


def test_discover_missing(tmp_path):
    assert discover(str(tmp_path), date="2030-01-01") == []


def test_read_plane_shape(fake_day):
    frames = discover(fake_day, date="2025-06-10")
    img = read_plane(frames[0].path, channel=70)
    assert img.shape == (96, 96)
    assert img.dtype == np.float32


def test_header_roundtrip(fake_day):
    frames = discover(fake_day, date="2025-06-10")
    hdr = read_header(frames[0].path)
    assert hdr["naxis1"] == 96
    assert hdr["cdelt1"] == pytest.approx(1.04)


def test_model_median_shifts(fake_day):
    frames = discover(fake_day, date="2025-06-10")
    model = JitterModel.build(frames, reference="median")
    # median CRPIX1 of [64, 72, 64.5, 72.2, 64.3] is 64.5
    assert model.ref1 == pytest.approx(64.5)
    dy, dx = model.shift(0)
    assert dx == pytest.approx(0.5)
    assert dy == pytest.approx(0.2)
    # frame 1 sits 8 px to the right, so it must be shifted back by 7.5
    assert model.shift(1)[1] == pytest.approx(-7.5)


def test_model_first_and_explicit(fake_day):
    frames = discover(fake_day, date="2025-06-10")
    m1 = JitterModel.build(frames, reference="first")
    assert m1.ref1 == pytest.approx(64.0)
    m2 = JitterModel.build(frames, reference="crpix:70,50")
    assert m2.ref1 == pytest.approx(70.0)
    assert m2.shift(0)[0] == pytest.approx(2.0)


def test_model_rejects_empty():
    with pytest.raises(ValueError):
        JitterModel.build([])


def test_model_stats(fake_day):
    frames = discover(fake_day, date="2025-06-10")
    stats = JitterModel.build(frames).stats()
    assert stats["n"] == 5
    assert stats["max_arcsec"] > 5
    assert stats["step_median_arcsec"] > 4


def test_model_save_load(tmp_path, fake_day):
    frames = discover(fake_day, date="2025-06-10")
    model = JitterModel.build(frames)
    path = model.save(str(tmp_path / "model.json"))
    back = JitterModel.load(path)
    assert back.ref1 == pytest.approx(model.ref1)
    assert back.shift(3) == pytest.approx(model.shift(3))
    assert len(back) == len(model)
