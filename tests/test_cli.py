import json
import os

import pytest

from pyprepchase.cli import main


def test_index(fake_day, capsys):
    rc = main(["index", "--root", fake_day, "--date", "2025-06-10"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "RSM20250610T010407" in out
    assert "CRPIX1" in out


def test_index_json(fake_day, capsys):
    rc = main(["index", "--root", fake_day, "--date", "2025-06-10",
               "--json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(data) == 5
    assert data[0]["crpix1"] == pytest.approx(64.0)


def test_index_missing(tmp_path, capsys):
    rc = main(["index", "--root", str(tmp_path), "--date", "2030-01-01"])
    assert rc == 1


def test_model(fake_day, tmp_path, capsys):
    rc = main(["model", "--root", fake_day, "--date", "2025-06-10",
               "--out", str(tmp_path), "--no-qc"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "5 frames" in out
    assert (tmp_path / "jitcal_20250610.json").exists()


def test_audit(fake_day, capsys):
    rc = main(["audit", "--root", fake_day, "--date", "2025-06-10",
               "--n", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "residual scatter" in out


def test_run(fake_day, tmp_path, capsys):
    outdir = str(tmp_path / "products")
    rc = main(["run", "--root", fake_day, "--date", "2025-06-10",
               "--outdir", outdir, "--channels", "70", "--no-audit"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "5 frames" in out
    assert os.path.exists(os.path.join(outdir, "manifest.json"))


def test_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "0.1.0" in capsys.readouterr().out
