# Bundled CHASE excerpt

Five real CHASE/HIS Level-1 frames, reduced so the repository stays small while
keeping the effect the package is about.

| item | value |
|---|---|
| source | `solarmanage:/data2/Chase/Lev1/2025/6/10/` (read-only) |
| frames | `RSM20250610T010407` to `RSM20250610T010851`, 5 frames, 72 s cadence |
| channels | absolute 69, 70, 71 (H-alpha line centre at 6562.8 A) |
| crop | 512 x 512 pixels, origin (x0, y0) = (467, 848), recorded in the header |
| size | 1.6 MB per frame, 7.9 MB total |
| header | `CRPIX1/CRPIX2` shifted by the crop origin, `CRVAL3` moved to the first kept channel |

The pointing jitter is untouched: consecutive frames differ by about 9 and 7
pixels (8.3 and 5.9 arcsec), so the excerpt behaves exactly like the full cube
for the model, the audit and the resampling. `DEMO`, `CROPX0`, `CROPY0` and
`SRCFILE` record the provenance.

Run `python examples/demo.py` to reproduce `docs/demo.png`.

CHASE data are publicly available from the CHASE data centre; this excerpt is
redistributed only as a regression fixture and a demonstration. Please cite the
CHASE mission and the HIS instrument paper in any publication that uses it.
