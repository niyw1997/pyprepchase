import os

import numpy as np
import pytest

from pyprepchase.frames import discover
from pyprepchase.jitter import JitterModel
from pyprepchase.limb import audit, audit_summary, fit_circle, limb_center


def synthetic_disk(shape=(300, 320), center=(190.0, 140.0), radius=110.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    rr = np.hypot(xx - center[0], yy - center[1])
    disk = 1.0 / (1.0 + np.exp((rr - radius) / 1.5))
    return 50.0 + 800.0 * disk


def test_fit_circle_exact():
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    x = 100 + 60 * np.cos(ang)
    y = 80 + 60 * np.sin(ang)
    fit = fit_circle(x, y)
    assert fit["cx"] == pytest.approx(100, abs=1e-6)
    assert fit["cy"] == pytest.approx(80, abs=1e-6)
    assert fit["r"] == pytest.approx(60, abs=1e-6)


def test_fit_circle_rejects_outliers():
    ang = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    x = 100 + 60 * np.cos(ang)
    y = 80 + 60 * np.sin(ang)
    x = np.r_[x, [0.0, 200.0]]  # two wild points
    y = np.r_[y, [0.0, 160.0]]
    fit = fit_circle(x, y)
    assert fit["cx"] == pytest.approx(100, abs=0.5)
    assert fit["r"] == pytest.approx(60, abs=0.5)


def test_limb_center_recovers_geometry():
    img = synthetic_disk()
    fit = limb_center(img, downsample=1)
    assert fit["cx"] == pytest.approx(190, abs=1.5)
    assert fit["cy"] == pytest.approx(140, abs=1.5)
    assert fit["r"] == pytest.approx(110, abs=2.5)


def test_limb_center_downsampled():
    img = synthetic_disk()
    fit = limb_center(img, downsample=2)
    assert fit["cx"] == pytest.approx(190, abs=2.0)
    # binning plus the 45% threshold biases the edge slightly inwards
    assert fit["r"] == pytest.approx(110, abs=6.0)


def test_audit_on_synthetic_day(fake_day):
    frames = discover(fake_day, date="2025-06-10")
    model = JitterModel.build(frames)
    rows = audit(model, n_frames=3, downsample=1)
    assert len(rows) == 3
    summary = audit_summary(rows)
    # The synthetic frames put the disk exactly at CRPIX, so the residual is ~0.
    assert summary["std_dx"] < 2.0
    assert summary["std_dy"] < 2.0


def test_audit_summary_empty():
    assert audit_summary([]) == {}
