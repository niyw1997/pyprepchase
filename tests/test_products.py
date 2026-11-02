import os

import numpy as np
import pytest

from pyprepchase.frames import Frame, discover
from pyprepchase.products import (crop_array, line_core_channel, product_header,
                                  shift_cube, shift_image)


def gauss(shape=(80, 80), center=(40.0, 38.0), sigma=4.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * sigma ** 2))


def centroid(img):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    total = img.sum()
    return (xx * img).sum() / total, (yy * img).sum() / total


def test_shift_image_moves_centroid():
    img = gauss()
    moved = shift_image(img, dy=2.0, dx=-3.0)
    cx, cy = centroid(moved)
    assert cx == pytest.approx(37.0, abs=0.15)
    assert cy == pytest.approx(40.0, abs=0.15)


def test_shift_image_noop_is_identity():
    img = gauss()
    out = shift_image(img, 0.0, 0.0)
    assert out is img


def test_shift_cube_spatial_only():
    cube = np.stack([gauss(), 2 * gauss()])
    out = shift_cube(cube, 1.0, 2.0)
    assert out.shape == cube.shape
    cx, cy = centroid(out[0])
    assert cx == pytest.approx(42.0, abs=0.15)
    assert cy == pytest.approx(39.0, abs=0.15)
    # relative scaling between planes is preserved
    assert np.isclose(out[1].max() / out[0].max(), 2.0, rtol=0.05)


def test_crop_array():
    arr = np.arange(20 * 30).reshape(20, 30)
    out = crop_array(arr, (5, 2, 4, 3))
    assert out.shape == (3, 4)
    assert out[0, 0] == arr[2, 5]


def test_crop_array_none_is_noop():
    arr = np.zeros((4, 5))
    assert crop_array(arr, None) is arr


def test_line_core_channel():
    fr = Frame(index=0, time=0.0, path="x", crval3=6559.40, cdelt3=0.0484, naxis3=118)
    # (6562.83 - 6559.40) / 0.0484 = 70.87 -> nearest channel 71
    assert line_core_channel(fr, "HA") == 71
    with pytest.raises(ValueError):
        line_core_channel(Frame(index=0, time=0.0, path="x", crval3=np.nan,
                                cdelt3=np.nan, naxis3=118), "HA")


def test_product_header_crop():
    fr = Frame(index=0, time=0.0, path="x", cdelt1=1.04, cdelt2=1.04,
               date_obs="2025-06-10T01:04:07")
    hdr = product_header(fr, 100.0, 90.0, (10, 20, 30, 40), 70)
    assert hdr["CRPIX1"] == pytest.approx(90.0)
    assert hdr["CRPIX2"] == pytest.approx(70.0)
    assert hdr["CHANNEL"] == 70
    assert hdr["CDELT1"] == pytest.approx(1.04)


def test_discover_gives_frames_for_header(fake_day):
    frames = discover(fake_day, date="2025-06-10")
    hdr = product_header(frames[0], 64.0, 48.0, None, None)
    assert hdr["CRPIX1"] == 64.0
    assert "CHANNEL" not in hdr
