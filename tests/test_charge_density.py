from __future__ import annotations

from pathlib import Path
import importlib

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.vasp import VaspChargeDensity

import vpmdk
charge_density_module = importlib.import_module("vpmdk_core.charge_density")


def test_determine_vasp_fft_grid_matches_normal_reference():
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.75, 0.0]],
        cell=[
            [10.5475997925, 0.0, 0.0],
            [-5.2737998962, 9.1344893692, 0.0],
            [0.0, 0.0, 8.4589996338],
        ],
        pbc=True,
    )

    grid_shape = vpmdk.determine_vasp_fft_grid(atoms, {"PREC": "N", "ENCUT": "400"})

    assert grid_shape == (108, 108, 84)


def test_determine_vasp_fft_grid_matches_accurate_reference():
    atoms = Atoms(
        "Ti2O4",
        positions=np.zeros((6, 3)),
        cell=[
            [4.594, 0.0, 0.0],
            [0.0, 4.594, 0.0],
            [0.0, 0.0, 2.958],
        ],
        pbc=True,
    )

    grid_shape = vpmdk.determine_vasp_fft_grid(atoms, {"PREC": "A", "ENCUT": "350"})

    assert grid_shape == (56, 56, 36)


def test_determine_vasp_fft_grid_respects_explicit_fine_grid(load_atoms):
    atoms = load_atoms()

    grid_shape = vpmdk.determine_vasp_fft_grid(
        atoms,
        {"ENCUT": "520", "NGXF": "20", "NGYF": "24", "NGZF": "28"},
    )

    assert grid_shape == (20, 24, 28)


def test_write_chgcar_roundtrips_density(tmp_path: Path, load_atoms):
    atoms = load_atoms()
    density = np.arange(24, dtype=float).reshape(2, 3, 4) / 10.0
    path = tmp_path / "CHGCAR"

    vpmdk.write_chgcar(path, atoms, density)

    reread = VaspChargeDensity(filename=str(path))
    assert np.allclose(reread.chg[-1], density)


def test_public_predict_charge_density_uses_backend_runner(
    monkeypatch: pytest.MonkeyPatch,
):
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.75, 0.0]],
        cell=[
            [10.5475997925, 0.0, 0.0],
            [-5.2737998962, 9.1344893692, 0.0],
            [0.0, 0.0, 8.4589996338],
        ],
        pbc=True,
    )
    seen: dict[str, object] = {}

    def fake_runner(atoms_arg, **kwargs):
        seen["n_atoms"] = len(atoms_arg)
        seen.update(kwargs)
        return np.ones(kwargs["grid_shape"], dtype=np.float32)

    monkeypatch.setattr(charge_density_module, "_run_charge3net_backend", fake_runner)

    result = vpmdk.predict_charge_density(
        atoms,
        incar={"PREC": "N", "ENCUT": "400"},
        backend="ChargE3Net",
    )

    assert isinstance(result, vpmdk.ChargeDensityResult)
    assert result.backend == "CHARGE3NET"
    assert result.grid_shape == (108, 108, 84)
    assert result.density.shape == (108, 108, 84)
    assert seen["grid_shape"] == (108, 108, 84)
    assert seen["n_atoms"] == len(atoms)
