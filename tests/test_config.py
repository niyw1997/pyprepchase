import json

import pytest

from pyprepchase.config import (JitterOptions, LineCoreOptions, OutputOptions,
                                PrepConfig, resolve_outdir)


def test_defaults_and_roundtrip(tmp_path):
    cfg = PrepConfig(root="/tmp", date="2025-06-10")
    cfg.jitter.reference = "first"
    cfg.channels = (68, 69, 70)
    cfg.output.crop = (1, 2, 3, 4)
    data = cfg.to_dict()
    back = PrepConfig.from_dict(json.loads(json.dumps(data)))
    assert back.channels == (68, 69, 70)
    assert back.jitter.reference == "first"
    assert tuple(back.output.crop) == (1, 2, 3, 4)
    assert back.line_core == LineCoreOptions()


def test_save_load(tmp_path):
    cfg = PrepConfig(root="/tmp", date="2025-06-10")
    path = cfg.save(str(tmp_path / "cfg.json"))
    back = PrepConfig.load(path)
    assert back.root == "/tmp"
    assert back.date == "2025-06-10"


def test_from_dict_ignores_unknown_keys():
    cfg = PrepConfig.from_dict({"root": "/x", "nonsense": 1,
                                "jitter": {"enabled": False, "bogus": 2}})
    assert cfg.root == "/x"
    assert cfg.jitter.enabled is False


@pytest.mark.parametrize("bad", [
    dict(jitter=JitterOptions(mode="warp")),
    dict(jitter=JitterOptions(reference="middle")),
    dict(line_core=LineCoreOptions(line="HX")),
    dict(output=OutputOptions(format="jpg")),
])
def test_validate_rejects(bad):
    cfg = PrepConfig(**bad)
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_accepts_default():
    PrepConfig().validate()


def test_resolve_outdir(tmp_path):
    target = tmp_path / "a" / "b"
    out = resolve_outdir(str(target))
    assert out.endswith("a/b")
    assert target.is_dir()
