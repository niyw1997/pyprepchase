import os

import numpy as np
import pytest

from pyprepchase.scv import available, describe, prepare


pytestmark = pytest.mark.skipif(not available(), reason="astropy/scipy missing")


def test_describe(fake_flat_day):
    path = sorted(os.listdir(fake_flat_day))[1]
    info = describe(os.path.join(fake_flat_day, path))
    assert info["n_frames"] == 3
    assert len(info["shift_px"]) == 2


def test_prepare_writes_corrected_cube(fake_flat_day, tmp_path):
    from astropy.io import fits

    name = sorted(os.listdir(fake_flat_day))[1]
    src = os.path.join(fake_flat_day, name)
    with fits.open(src) as hdul:
        original_crpix = (hdul[1].header["CRPIX1"], hdul[1].header["CRPIX2"])
        original_shape = hdul[1].data.shape

    info = prepare(src, cache_dir=str(tmp_path), verbose=False)
    assert not info["cached"]
    assert os.path.exists(info["path"])
    with fits.open(info["path"]) as hdul:
        assert hdul[0].data.shape == original_shape
        new_crpix = (hdul[0].header["CRPIX1"], hdul[0].header["CRPIX2"])
    assert new_crpix != original_crpix
    # the source file is untouched
    with fits.open(src) as hdul:
        assert (hdul[1].header["CRPIX1"], hdul[1].header["CRPIX2"]) == original_crpix


def test_prepare_reuses_cache(fake_flat_day, tmp_path):
    name = sorted(os.listdir(fake_flat_day))[1]
    src = os.path.join(fake_flat_day, name)
    first = prepare(src, cache_dir=str(tmp_path))
    second = prepare(src, cache_dir=str(tmp_path))
    assert second["cached"] is True
    assert second["path"] == first["path"]


def test_prepare_channel_subset(fake_flat_day, tmp_path):
    from astropy.io import fits

    name = sorted(os.listdir(fake_flat_day))[1]
    src = os.path.join(fake_flat_day, name)
    info = prepare(src, cache_dir=str(tmp_path), channels=[70])
    with fits.open(info["path"]) as hdul:
        assert hdul[0].data.shape[0] == 1
