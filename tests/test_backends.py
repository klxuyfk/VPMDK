from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import vpmdk


def test_nequip_uses_compiled_model_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_path = tmp_path / "model.pth"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    def from_compiled_model(path, device=None):
        seen["path"] = path
        seen["device"] = device
        return "nequip-compiled"

    monkeypatch.setattr(
        vpmdk,
        "NequIPCalculator",
        SimpleNamespace(from_compiled_model=from_compiled_model),
    )

    calc = vpmdk._build_nequip_calculator({"MODEL": str(model_path), "DEVICE": "cuda"})

    assert calc == "nequip-compiled"
    assert seen == {"path": str(model_path), "device": "cuda"}


def test_matgl_load_model_path_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_dir = tmp_path / "M3GNet-MP"
    model_dir.mkdir()
    seen: dict[str, object] = {}

    def fake_load_model(path):
        seen["load_path"] = path
        return "potential"

    def fake_calc(*args, **kwargs):
        seen["calc_args"] = args
        seen["calc_kwargs"] = kwargs
        return "calc"

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", fake_calc)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", fake_load_model)
    monkeypatch.setattr(vpmdk, "LegacyM3GNetPotential", None)

    calc = vpmdk._build_m3gnet_calculator({"MODEL": str(model_dir)})

    assert calc == "calc"
    assert seen["load_path"] == str(model_dir)
    assert seen["calc_args"] == ("potential",)


def test_deepmd_head_is_forwarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    def fake_calc(*, model, **kwargs):
        seen["model"] = model
        seen.update(kwargs)
        return "deepmd"

    monkeypatch.setattr(vpmdk, "DeePMDCalculator", fake_calc)

    calc = vpmdk._build_deepmd_calculator(
        {"MODEL": str(model_path), "DEEPMD_HEAD": "myhead"}
    )

    assert calc == "deepmd"
    assert seen["model"] == str(model_path)
    assert seen["head"] == "myhead"


def test_deepmd_requires_model_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "DeePMDCalculator", lambda *a, **k: None)

    with pytest.raises(ValueError, match="requires MODEL"):
        vpmdk._build_deepmd_calculator({})


def test_deepmd_missing_model_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "DeePMDCalculator", lambda *a, **k: None)

    missing_path = tmp_path / "missing.pb"
    with pytest.raises(FileNotFoundError, match="not found"):
        vpmdk._build_deepmd_calculator({"MODEL": str(missing_path)})
