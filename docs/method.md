# Method and validation

## 1. The problem in one paragraph

CHASE/HIS Level-1 `_HA.fits` cubes are delivered in detector coordinates. The solar disk does not
sit at a fixed place: from frame to frame it moves by up to about 10 arcsec, and the header records
that motion in `CRPIX1`/`CRPIX2`. An analysis that cuts a fixed pixel window therefore sees a
sawtooth of the same size. On 2025-06-10 the sawtooth period is two frames, which is why it was
first described as an odd/even effect. It is not: the odd/even phase flips during other days, so a
per-parity constant is not enough. Every frame needs its own shift.

## 2. Why the header is the right source

The header is an absolute record, and it can be checked. For every audit frame we fit a circle to
the solar limb (Otsu-like threshold between sky and the 99.5th percentile, hole filling, erosion,
boundary points, Kasa fit with 3-sigma clipping) and compare its centre with `CRPIX`:

| day | sampled frames | mean (x, y) | std (x, y) | range (x, y) |
|---|---|---|---|---|
| 2025-06-10 | 10 | (-1.41, +0.04) | (0.21, 0.16) | (0.68, 0.47) |
| 2025-06-11 | 7 | (-1.66, -0.01) | (0.21, 0.18) | (0.66, 0.56) |
| 2026-02-13 | 7 | (+0.13, -0.85) | (0.10, 0.14) | (0.30, 0.38) |

Units are pixels (1 pixel = 1.04"). The constant part is a threshold effect and differs by day; the
frame-to-frame scatter, 0.10 to 0.21 pixel, is the calibration accuracy.

Two cross-checks rule out the obvious alternatives:

* **not a scale change**: the limb radius differs by 0.13 pixel between consecutive frames, while a
  0.066% zoom would give 0.6 pixel; all nine tested wavelength channels give the same translation;
* **not a rotation**: after correction the residual displacement is the same in all four quadrants
  (spread 0.07 pixel). A rotation would flip the sign of the y residual between left and right.

## 3. The model

For a day, the reference grid is the median `(CRPIX1, CRPIX2)` of all frames. Frame `k` is shifted by

```
dx_k = ref_x - CRPIX1_k
dy_k = ref_y - CRPIX2_k
```

with a cubic spline (`scipy.ndimage.shift`). Integer and sub-pixel shifts use the same code path, so
a series does not change character when a shift happens to be a whole pixel. The reference can also
be the first frame or an explicit `crpix:x,y`.

## 4. Validation on real data

Same 25-frame segment, same 160 tracked points, only the correction toggled:

| metric | 2025-06-10 | 2026-02-13 |
|---|---|---|
| detrended residual before | 8.64" | 8.90" |
| detrended residual after | **0.38"** | **0.67"** |
| worst frame pair after | 0.43" | 0.85" |
| odd/even stack difference | 13.79 -> 2.70 DN | 8.02 -> 2.32 DN |
| quadrant spread | 0.070 px | 0.128 px |

The 2026-02-13 run contains one 68-minute data gap; that pair is excluded by the "frame-to-frame"
rule and reported separately by the flags.

## 5. Known limits

* No `CTYPE`/`CROTA` in the header: rotation is not modelled. It is below 0.1 pixel on the days
  tested, but high-latitude or long-baseline work should calibrate it with sunspots.
* Large repoints (88.7" on 2025-06-12) are detected as `repoint` flags, not corrected away.
* `RSUN_OBS` varies by 7.6" within a day while the measured limb is stable to 0.2 pixel; derive the
  plate scale from the limb, not from `RSUN_OBS`.
* The wavelength axis jitters too (about 5.3 km/s in `CRVAL3`); this package does not correct it.
