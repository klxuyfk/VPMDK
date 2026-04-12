from __future__ import annotations

from pathlib import Path

import pytest

import vpmdk
from tests.conftest import DummyCalculator


def test_public_single_point_returns_result_without_vasp_side_effects(
    tmp_path: Path,
    load_atoms,
):
    atoms = load_atoms()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        result = vpmdk.single_point(atoms, calculator=DummyCalculator())
    finally:
        monkeypatch.undo()

    assert isinstance(result, vpmdk.SinglePointResult)
    assert result.potential_energy == 0.5
    assert result.forces is not None
    assert not (tmp_path / "OUTCAR").exists()
    assert not (tmp_path / "OSZICAR").exists()
    assert not (tmp_path / "vasprun.xml").exists()
    assert not (tmp_path / "CONTCAR").exists()


def test_public_get_calculator_accepts_backend_kwargs(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_builder(tags):
        captured["tags"] = tags
        return "calc"

    monkeypatch.setattr(vpmdk, "_build_flashtp_calculator", fake_builder)

    calculator = vpmdk.get_calculator(
        mlp="FlashTP",
        model="7net-0",
        device="cuda:0",
        sevennet_enable_flash=True,
    )

    assert calculator == "calc"
    assert captured["tags"] == {
        "MLP": "FLASHTP",
        "MODEL": "7net-0",
        "DEVICE": "cuda:0",
        "SEVENNET_ENABLE_FLASH": "1",
    }


def test_get_backend_capabilities_reflects_matris_task():
    caps_e = vpmdk.get_backend_capabilities("MATRIS", matris_task="e")
    caps_efs = vpmdk.get_backend_capabilities("MATRIS", matris_task="efs")

    assert caps_e.energy is True
    assert caps_e.forces is False
    assert caps_e.stress is False
    assert caps_efs.forces is True
    assert caps_efs.stress is True


def test_list_backends_exposes_known_entries():
    names = {spec.name for spec in vpmdk.list_backends()}

    assert "CHGNET" in names
    assert "MACE" in names
    assert "FAIRCHEM" in names


def test_public_md_maps_thermostat_name(monkeypatch: pytest.MonkeyPatch, load_atoms):
    atoms = load_atoms()
    captured: dict[str, object] = {}

    class DummyDynamics:
        def run(self, n):
            assert n == 1

    def fake_selector(atoms_arg, mdalgo, timestep, initial_temperature, smass, params):
        captured.update(
            {
                "mdalgo": mdalgo,
                "timestep": timestep,
                "temperature": initial_temperature,
                "params": params,
            }
        )
        return DummyDynamics(), lambda temp: None

    monkeypatch.setattr(vpmdk, "_select_md_dynamics", fake_selector)
    monkeypatch.setattr(
        vpmdk.velocitydistribution,
        "MaxwellBoltzmannDistribution",
        lambda *a, **k: None,
    )

    result = vpmdk.md(
        atoms,
        calculator=DummyCalculator(),
        steps=1,
        temperature=300.0,
        timestep=2.0,
        thermostat="langevin",
        thermostat_kwargs={"LANGEVIN_GAMMA": 1.5},
    )

    assert isinstance(result, vpmdk.MDResult)
    assert captured["mdalgo"] == 3
    assert captured["timestep"] == 2.0
    assert captured["temperature"] == 300.0
    assert captured["params"] == {"LANGEVIN_GAMMA": 1.5}
