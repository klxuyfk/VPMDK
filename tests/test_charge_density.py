from __future__ import annotations

from pathlib import Path
import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.vasp import VaspChargeDensity

import vpmdk
charge_density_module = importlib.import_module("vpmdk_core.charge_density")
charge3net_runner_spec = importlib.util.spec_from_file_location(
    "vpmdk_core_charge3net_runner",
    Path(__file__).resolve().parents[1] / "src" / "vpmdk_core" / "charge3net_runner.py",
)
assert charge3net_runner_spec is not None and charge3net_runner_spec.loader is not None
charge3net_runner_module = importlib.util.module_from_spec(charge3net_runner_spec)
charge3net_runner_spec.loader.exec_module(charge3net_runner_module)


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

    assert grid_shape == (60, 60, 40)


def test_determine_vasp_fft_grid_respects_explicit_fine_grid(load_atoms):
    atoms = load_atoms()

    grid_shape = vpmdk.determine_vasp_fft_grid(
        atoms,
        {"ENCUT": "520", "NGXF": "20", "NGYF": "24", "NGZF": "28"},
    )

    assert grid_shape == (20, 24, 28)


def test_next_even_smooth_number_never_rounds_down():
    result = charge_density_module._next_even_smooth_number(10.1)

    assert result >= 10.1
    assert result % 2 == 0
    assert charge_density_module._largest_prime_factor(result) <= 7


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


def test_charge3net_backend_allows_missing_source_dir_and_omits_device_flag_when_unspecified(
    monkeypatch: pytest.MonkeyPatch,
):
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.75, 0.0]],
        cell=np.eye(3),
        pbc=True,
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_source_dir",
        lambda source_dir: None,
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_model_path",
        lambda model_path, source_dir: "/tmp/charge3net/models/model.pt",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_python",
        lambda python_executable: "/tmp/charge-env/bin/python",
    )
    monkeypatch.setattr(charge_density_module.np, "load", lambda path: np.ones((2, 2, 2), dtype=np.float32))
    monkeypatch.setattr(
        charge_density_module,
        "_root",
        lambda: SimpleNamespace(_resolve_device=lambda value: (_ for _ in ()).throw(AssertionError())),
    )

    def fake_run(command, **kwargs):
        seen["command"] = list(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(charge_density_module.subprocess, "run", fake_run)

    density = charge_density_module._run_charge3net_backend(atoms, grid_shape=(2, 2, 2))

    assert density.shape == (2, 2, 2)
    assert "--device" not in seen["command"]
    assert "--source-dir" not in seen["command"]


def test_charge3net_runner_auto_device_prefers_cuda_when_available():
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    assert charge3net_runner_module._resolve_device_argument(None, fake_torch) == "cuda"
    assert charge3net_runner_module._resolve_device_argument("cpu", fake_torch) == "cpu"


def test_charge3net_backend_passes_explicit_model_config_flags(
    monkeypatch: pytest.MonkeyPatch,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_source_dir",
        lambda source_dir: "/tmp/charge3net",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_model_path",
        lambda model_path, source_dir: "/tmp/charge3net/models/model.pt",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_python",
        lambda python_executable: "/tmp/charge-env/bin/python",
    )
    monkeypatch.setattr(charge_density_module.np, "load", lambda path: np.ones((2, 2, 2), dtype=np.float32))

    def fake_run(command, **kwargs):
        seen["command"] = list(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(charge_density_module.subprocess, "run", fake_run)

    charge_density_module._run_charge3net_backend(
        atoms,
        grid_shape=(2, 2, 2),
        num_interactions=4,
        num_neighbors=12.5,
        mul=384,
        lmax=5,
        basis="bessel",
        num_basis=16,
        spin=True,
    )

    command = seen["command"]
    assert "--num-interactions" in command
    assert "--num-neighbors" in command
    assert "--mul" in command
    assert "--lmax" in command
    assert "--basis" in command
    assert "--num-basis" in command
    assert "--spin" in command


def test_charge3net_runner_resolves_model_config_from_checkpoint_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        charge3net_runner_module,
        "_infer_model_config_from_state_dict",
        lambda state_dict, model_cls: {"lmax": 6, "spin": False},
    )
    checkpoint = {
        "model": {"dummy": np.zeros((1,))},
        "hyper_parameters": {
            "num_interactions": "4",
            "num_neighbors": "18",
            "basis": "gaussian",
            "num_basis": "14",
        },
        "config": {"model": {"mul": "320", "lmax": "5"}},
    }

    config = charge3net_runner_module._resolve_model_config(
        checkpoint,
        explicit_config={"basis": "bessel", "mul": 448, "cutoff": 5.5},
        model_cls=object,
    )

    assert config == {
        "num_interactions": 4,
        "num_neighbors": 18.0,
        "basis": "bessel",
        "num_basis": 14,
        "mul": 448,
        "lmax": 5,
        "cutoff": 5.5,
        "spin": False,
    }


def test_charge3net_runner_uses_inferred_config_for_metadata_free_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = {
        "model": {"dummy": np.zeros((1,))},
    }

    monkeypatch.setattr(
        charge3net_runner_module,
        "_infer_model_config_from_state_dict",
        lambda state_dict, model_cls: {
            "num_interactions": 2,
            "mul": 192,
            "lmax": 3,
            "num_basis": 12,
            "spin": False,
        },
    )

    config = charge3net_runner_module._resolve_model_config(
        checkpoint,
        explicit_config={"cutoff": 5.5},
        model_cls=object,
    )

    assert config == {
        "num_interactions": 2,
        "num_neighbors": 20.0,
        "mul": 192,
        "lmax": 3,
        "cutoff": 5.5,
        "basis": "gaussian",
        "num_basis": 12,
        "spin": False,
    }


def test_charge3net_runner_falls_back_to_defaults_when_inference_finds_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = {
        "model": {"dummy": np.zeros((1,))},
    }

    monkeypatch.setattr(
        charge3net_runner_module,
        "_infer_model_config_from_state_dict",
        lambda state_dict, model_cls: {},
    )

    config = charge3net_runner_module._resolve_model_config(
        checkpoint,
        explicit_config={"cutoff": 5.5},
        model_cls=object,
    )

    assert config == {
        "num_interactions": 3,
        "num_neighbors": 20.0,
        "mul": 500,
        "lmax": 4,
        "cutoff": 5.5,
        "basis": "gaussian",
        "num_basis": 20,
        "spin": False,
    }
