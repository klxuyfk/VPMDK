from __future__ import annotations

import sys
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
deepcdp_runner_spec = importlib.util.spec_from_file_location(
    "vpmdk_core_deepcdp_runner",
    Path(__file__).resolve().parents[1] / "src" / "vpmdk_core" / "deepcdp_runner.py",
)
assert deepcdp_runner_spec is not None and deepcdp_runner_spec.loader is not None
deepcdp_runner_module = importlib.util.module_from_spec(deepcdp_runner_spec)
deepcdp_runner_spec.loader.exec_module(deepcdp_runner_module)


class _FakeLoadedDensity:
    def __init__(self, density, spin_density=None):
        self.files = ["density"] if spin_density is None else ["density", "spin_density"]
        self._payload = {"density": density}
        if spin_density is not None:
            self._payload["spin_density"] = spin_density

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getitem__(self, key):
        return self._payload[key]


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


def test_determine_vasp_fft_grid_preserves_partial_explicit_coarse_override():
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

    grid_shape = vpmdk.determine_vasp_fft_grid(
        atoms,
        {"PREC": "N", "ENCUT": "400", "NGX": "80", "NGZ": "60"},
    )

    assert grid_shape == (160, 108, 120)


@pytest.mark.parametrize(
    ("grid_shape", "message"),
    [
        ((63.9, 64, 64), "Invalid grid shape"),
        ((64, 0.5, 64), "Invalid grid shape"),
    ],
)
def test_coerce_grid_shape_rejects_non_integral_values(grid_shape, message):
    with pytest.raises(ValueError, match=message):
        charge_density_module._coerce_grid_shape(grid_shape)


@pytest.mark.parametrize(
    "incar",
    [
        {"NGXF": "63.9", "NGYF": "64", "NGZF": "64"},
        {"ENCUT": "400", "PREC": "N", "NGX": "63.9"},
    ],
)
def test_determine_vasp_fft_grid_rejects_non_integral_explicit_values(incar):
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

    with pytest.raises(ValueError):
        vpmdk.determine_vasp_fft_grid(atoms, incar)


def test_next_even_smooth_number_never_rounds_down():
    result = charge_density_module._next_even_smooth_number(10.1)

    assert result >= 10.1
    assert result % 2 == 0
    assert charge_density_module._largest_prime_factor(result) <= 7


def test_charge_env_paths_resolve_relative_to_preserved_caller_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source_dir = caller_dir / "charge_src"
    source_dir.mkdir()
    model_path = caller_dir / "charge_model.pt"
    model_path.write_text("checkpoint")

    monkeypatch.chdir(run_dir)
    monkeypatch.setenv(charge_density_module._CHARGE_ENV_BASE_DIR_VAR, str(caller_dir))
    monkeypatch.setenv("VPMDK_CHARGE_PYTHON", "charge_env/bin/python")
    monkeypatch.setenv("VPMDK_CHARGE_SOURCE_DIR", "charge_src")
    monkeypatch.setenv("VPMDK_CHARGE_MODEL", "charge_model.pt")

    assert (
        charge_density_module._resolve_charge_python(None)
        == str((caller_dir / "charge_env" / "bin" / "python").resolve())
    )
    assert charge_density_module._resolve_charge_source_dir(None) == str(source_dir.resolve())
    assert charge_density_module._resolve_charge_model_path(None, None) == str(model_path.resolve())


def test_explicit_charge_model_path_expands_user_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    resolved = charge_density_module._resolve_charge_model_path("~/models/charge3net_mp.pt", None)

    assert resolved == str((home_dir / "models" / "charge3net_mp.pt").resolve())


def test_explicit_charge_python_path_expands_user_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    resolved = charge_density_module._resolve_charge_python("~/charge-env/bin/python")

    assert resolved == str((home_dir / "charge-env" / "bin" / "python").resolve())


def test_explicit_charge_paths_remain_relative_to_current_context(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(charge_density_module._CHARGE_ENV_BASE_DIR_VAR, "/tmp/caller")

    assert charge_density_module._resolve_charge_python("relative-python") == "relative-python"
    assert charge_density_module._resolve_charge_source_dir("relative-source") == "relative-source"
    assert (
        charge_density_module._resolve_charge_model_path("relative-model.pt", "relative-source")
        == "relative-model.pt"
    )


def test_backend_specific_charge_env_vars_override_generic_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.setenv(charge_density_module._CHARGE_ENV_BASE_DIR_VAR, str(caller_dir))
    monkeypatch.setenv("VPMDK_CHARGE_PYTHON", "generic-env/bin/python")
    monkeypatch.setenv("VPMDK_DEEPDFT_PYTHON", "deepdft-env/bin/python")
    monkeypatch.setenv("VPMDK_CHARGE_SOURCE_DIR", "generic-src")
    monkeypatch.setenv("VPMDK_DEEPDFT_SOURCE_DIR", "deepdft-src")
    monkeypatch.setenv("VPMDK_CHARGE_MODEL", "generic-model")
    monkeypatch.setenv("VPMDK_DEEPDFT_MODEL", "deepdft-model")

    assert charge_density_module._resolve_charge_python(None, backend="DEEPDFT") == str(
        (caller_dir / "deepdft-env" / "bin" / "python").resolve()
    )
    assert charge_density_module._resolve_charge_source_dir(None, backend="DEEPDFT") == str(
        (caller_dir / "deepdft-src").resolve()
    )
    assert charge_density_module._resolve_charge_model_path(
        None,
        None,
        backend="DEEPDFT",
    ) == str((caller_dir / "deepdft-model").resolve())


def test_charge3net_runner_adjusts_singleton_tail_slice():
    batch_start, batch_stop, keep_slice = charge3net_runner_module._adjust_singleton_probe_slice(
        2500,
        2501,
        2501,
    )

    assert (batch_start, batch_stop) == (2499, 2501)
    assert keep_slice.indices(2) == (1, 2, 1)


def test_charge3net_runner_preserves_only_probe_when_total_is_one():
    batch_start, batch_stop, keep_slice = charge3net_runner_module._adjust_singleton_probe_slice(
        0,
        1,
        1,
    )

    assert (batch_start, batch_stop) == (0, 1)
    assert keep_slice.indices(1) == (0, 1, 1)


def test_deepcdp_runner_normalizes_layers_prefix_in_checkpoint_keys():
    normalized = deepcdp_runner_module._normalize_state_dict_keys(
        {
            "layers.0.weight": np.zeros((3, 2), dtype=np.float32),
            "layers.0.bias": np.zeros(3, dtype=np.float32),
        }
    )

    assert set(normalized) == {"0.weight", "0.bias"}


def test_deepcdp_runner_requires_explicit_activation():
    args = SimpleNamespace(activation=None)

    with pytest.raises(ValueError, match="DeepCDP activation must be provided"):
        deepcdp_runner_module._resolve_activation(args, {})


def test_deepcdp_runner_accepts_activation_from_metadata():
    args = SimpleNamespace(activation=None)

    assert deepcdp_runner_module._resolve_activation(args, {"activation": "silu"}) == "silu"


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
        return np.ones(kwargs["grid_shape"], dtype=np.float32), None

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


def test_public_predict_charge_density_uses_deepdft_backend_runner(
    monkeypatch: pytest.MonkeyPatch,
):
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.75, 0.0]],
        cell=np.eye(3),
        pbc=True,
    )
    seen: dict[str, object] = {}

    def fake_runner(atoms_arg, **kwargs):
        seen["n_atoms"] = len(atoms_arg)
        seen.update(kwargs)
        return np.ones(kwargs["grid_shape"], dtype=np.float32), None

    monkeypatch.setattr(charge_density_module, "_run_deepdft_backend", fake_runner)

    result = vpmdk.predict_charge_density(
        atoms,
        grid_shape=(4, 4, 4),
        backend="DeepDFT",
    )

    assert isinstance(result, vpmdk.ChargeDensityResult)
    assert result.backend == "DEEPDFT"
    assert result.grid_shape == (4, 4, 4)
    assert result.density.shape == (4, 4, 4)
    assert result.spin_density is None
    assert seen["grid_shape"] == (4, 4, 4)
    assert seen["n_atoms"] == len(atoms)


def test_public_predict_charge_density_uses_deepcdp_backend_runner(
    monkeypatch: pytest.MonkeyPatch,
):
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.75, 0.0]],
        cell=np.eye(3),
        pbc=True,
    )
    seen: dict[str, object] = {}

    def fake_runner(atoms_arg, **kwargs):
        seen["n_atoms"] = len(atoms_arg)
        seen.update(kwargs)
        return np.ones(kwargs["grid_shape"], dtype=np.float32), None

    monkeypatch.setattr(charge_density_module, "_run_deepcdp_backend", fake_runner)

    result = vpmdk.predict_charge_density(
        atoms,
        grid_shape=(4, 4, 4),
        backend="DeepCDP",
    )

    assert isinstance(result, vpmdk.ChargeDensityResult)
    assert result.backend == "DEEPCDP"
    assert result.grid_shape == (4, 4, 4)
    assert result.density.shape == (4, 4, 4)
    assert result.spin_density is None
    assert seen["grid_shape"] == (4, 4, 4)
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
        lambda source_dir, backend=None: None,
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_model_path",
        lambda model_path, source_dir, backend=None: "/tmp/charge3net/models/model.pt",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_python",
        lambda python_executable, backend=None: "/tmp/charge-env/bin/python",
    )
    monkeypatch.setattr(
        charge_density_module.np,
        "load",
        lambda path: _FakeLoadedDensity(np.ones((2, 2, 2), dtype=np.float32)),
    )
    monkeypatch.setattr(
        charge_density_module,
        "_root",
        lambda: SimpleNamespace(_resolve_device=lambda value: (_ for _ in ()).throw(AssertionError())),
    )

    def fake_run(command, **kwargs):
        seen["command"] = list(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(charge_density_module.subprocess, "run", fake_run)

    density, spin_density = charge_density_module._run_charge3net_backend(atoms, grid_shape=(2, 2, 2))

    assert density.shape == (2, 2, 2)
    assert spin_density is None
    assert "--device" not in seen["command"]
    assert "--source-dir" not in seen["command"]
    assert "--cutoff" not in seen["command"]


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
        lambda source_dir, backend=None: "/tmp/charge3net",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_model_path",
        lambda model_path, source_dir, backend=None: "/tmp/charge3net/models/model.pt",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_python",
        lambda python_executable, backend=None: "/tmp/charge-env/bin/python",
    )
    monkeypatch.setattr(
        charge_density_module.np,
        "load",
        lambda path: _FakeLoadedDensity(np.ones((2, 2, 2), dtype=np.float32)),
    )

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
    assert "--cutoff" not in command
    assert "--num-interactions" in command
    assert "--num-neighbors" in command
    assert "--mul" in command
    assert "--lmax" in command
    assert "--basis" in command
    assert "--num-basis" in command
    assert "--spin" in command


@pytest.mark.parametrize("raw_value", ["0", "-2", "0.5", "1.5"])
def test_charge_density_options_reject_non_positive_max_probes_per_batch(raw_value: str):
    with pytest.raises(ValueError, match="CHARGE_MAX_PROBES_PER_BATCH"):
        charge_density_module._charge_density_options_from_bcar(
            {"CHARGE_MAX_PROBES_PER_BATCH": raw_value}
        )


def test_charge_density_options_parse_model_config_overrides():
    options = charge_density_module._charge_density_options_from_bcar(
        {
            "CHARGE_NUM_INTERACTIONS": "4",
            "CHARGE_NUM_NEIGHBORS": "12.5",
            "CHARGE_MUL": "384",
            "CHARGE_LMAX": "5",
            "CHARGE_BASIS": "bessel",
            "CHARGE_NUM_BASIS": "16",
            "CHARGE_SPIN": "1",
        }
    )

    assert options["num_interactions"] == 4
    assert options["num_neighbors"] == 12.5
    assert options["mul"] == 384
    assert options["lmax"] == 5
    assert options["basis"] == "bessel"
    assert options["num_basis"] == 16
    assert options["spin"] is True


@pytest.mark.parametrize(
    ("tag_name", "raw_value"),
    [
        ("CHARGE_NUM_INTERACTIONS", "3.9"),
        ("CHARGE_MUL", "384.2"),
        ("CHARGE_LMAX", "4.7"),
        ("CHARGE_NUM_BASIS", "16.5"),
    ],
)
def test_charge_density_options_reject_fractional_integer_model_config_values(
    tag_name: str,
    raw_value: str,
):
    with pytest.raises(ValueError, match=tag_name):
        charge_density_module._charge_density_options_from_bcar({tag_name: raw_value})


@pytest.mark.parametrize("raw_value", [0, -2, 0.5, 1.5])
def test_public_predict_charge_density_rejects_invalid_max_probes_per_batch(raw_value: object):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)

    with pytest.raises(ValueError, match="CHARGE_MAX_PROBES_PER_BATCH"):
        vpmdk.predict_charge_density(
            atoms,
            grid_shape=(2, 2, 2),
            max_probes_per_batch=raw_value,
        )


def test_public_predict_charge_density_metadata_uses_resolved_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    source_dir = tmp_path / "charge_src"
    model_path = source_dir / "models" / "charge3net_mp.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("checkpoint")

    monkeypatch.setattr(
        charge_density_module,
        "_run_charge3net_backend",
        lambda atoms_arg, **kwargs: (
            np.ones((2, 2, 2), dtype=np.float32),
            None,
        ),
    )

    result = vpmdk.predict_charge_density(
        atoms,
        grid_shape=(2, 2, 2),
        source_dir=str(source_dir),
    )

    assert result.metadata["model_path"] == str(model_path)
    assert result.metadata["source_dir"] == str(source_dir)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_interactions": 3.9}, "num_interactions"),
        ({"mul": 384.2}, "mul"),
        ({"lmax": 4.7}, "lmax"),
        ({"num_basis": 16.5}, "num_basis"),
    ],
)
def test_public_predict_charge_density_rejects_fractional_integer_model_config_values(
    kwargs: dict[str, object],
    message: str,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)

    with pytest.raises(ValueError, match=message):
        vpmdk.predict_charge_density(
            atoms,
            grid_shape=(2, 2, 2),
            **kwargs,
        )


def test_charge3net_runner_split_probe_output_handles_spin_channels():
    predictions = [
        SimpleNamespace(
            detach=lambda: SimpleNamespace(
                cpu=lambda: SimpleNamespace(
                    numpy=lambda: np.array(
                        [
                            [1.0, 0.25],
                            [0.5, -0.5],
                        ],
                        dtype=np.float32,
                    )
                )
            )
        )
    ]

    density, spin_density = charge3net_runner_module._split_probe_output(predictions, spin=True)

    assert np.allclose(density, np.array([1.25, 0.0], dtype=np.float32))
    assert np.allclose(spin_density, np.array([0.75, 1.0], dtype=np.float32))


def test_charge3net_runner_load_checkpoint_falls_back_without_weights_only():
    calls: list[dict[str, object]] = []

    class FakeTorch:
        @staticmethod
        def load(path, map_location=None, **kwargs):
            calls.append({"path": path, "map_location": map_location, "kwargs": dict(kwargs)})
            if "weights_only" in kwargs:
                raise TypeError("unexpected keyword argument 'weights_only'")
            return {"model": {}}

    checkpoint = charge3net_runner_module._torch_load_checkpoint(
        FakeTorch,
        "/tmp/model.pt",
        map_location="cpu",
    )

    assert checkpoint == {"model": {}}
    assert calls == [
        {
            "path": "/tmp/model.pt",
            "map_location": "cpu",
            "kwargs": {"weights_only": False},
        },
        {
            "path": "/tmp/model.pt",
            "map_location": "cpu",
            "kwargs": {},
        },
    ]


def test_public_predict_charge_density_preserves_spin_density(monkeypatch: pytest.MonkeyPatch):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)

    monkeypatch.setattr(
        charge_density_module,
        "_run_charge3net_backend",
        lambda atoms_arg, **kwargs: (
            np.ones((2, 2, 2), dtype=np.float32),
            np.full((2, 2, 2), 0.5, dtype=np.float32),
        ),
    )

    result = vpmdk.predict_charge_density(atoms, grid_shape=(2, 2, 2), spin=True)

    assert result.spin_density is not None
    assert result.spin_density.shape == (2, 2, 2)
    assert result.metadata["model_config"]["spin_output"] is True


def test_charge3net_backend_passes_explicit_cutoff_override(
    monkeypatch: pytest.MonkeyPatch,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_source_dir",
        lambda source_dir, backend=None: None,
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_model_path",
        lambda model_path, source_dir, backend=None: "/tmp/charge3net/models/model.pt",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_python",
        lambda python_executable, backend=None: "/tmp/charge-env/bin/python",
    )
    monkeypatch.setattr(
        charge_density_module.np,
        "load",
        lambda path: _FakeLoadedDensity(np.ones((2, 2, 2), dtype=np.float32)),
    )

    def fake_run(command, **kwargs):
        seen["command"] = list(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(charge_density_module.subprocess, "run", fake_run)

    charge_density_module._run_charge3net_backend(atoms, grid_shape=(2, 2, 2), cutoff=5.5)

    command = seen["command"]
    cutoff_index = command.index("--cutoff")
    assert command[cutoff_index + 1] == "5.5"


def test_deepdft_backend_passes_model_dir_and_probe_count(
    monkeypatch: pytest.MonkeyPatch,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_source_dir",
        lambda source_dir, backend=None: "/tmp/deepdft",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_deepdft_model_dir",
        lambda model_path, source_dir: "/tmp/deepdft/model",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_python",
        lambda python_executable, backend=None: "/tmp/deepdft-env/bin/python",
    )
    monkeypatch.setattr(
        charge_density_module.np,
        "load",
        lambda path: _FakeLoadedDensity(np.ones((2, 2, 2), dtype=np.float32)),
    )

    def fake_run(command, **kwargs):
        seen["command"] = list(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(charge_density_module.subprocess, "run", fake_run)

    density, spin_density = charge_density_module._run_deepdft_backend(
        atoms,
        grid_shape=(2, 2, 2),
        device="cuda:0",
        max_probes_per_batch=4096,
    )

    assert density.shape == (2, 2, 2)
    assert spin_density is None
    command = seen["command"]
    assert command[:2] == [
        "/tmp/deepdft-env/bin/python",
        str(Path(charge_density_module.__file__).with_name("deepdft_runner.py")),
    ]
    assert "--model-dir" in command
    assert command[command.index("--model-dir") + 1] == "/tmp/deepdft/model"
    assert "--probe-count" in command
    assert command[command.index("--probe-count") + 1] == "4096"
    assert "--source-dir" in command
    assert command[command.index("--source-dir") + 1] == "/tmp/deepdft"
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cuda:0"


def test_deepcdp_backend_passes_checkpoint_and_soap_overrides(
    monkeypatch: pytest.MonkeyPatch,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_source_dir",
        lambda source_dir, backend=None: "/tmp/deepcdp",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_deepcdp_checkpoint_path",
        lambda model_path, source_dir: "/tmp/deepcdp/model.pt",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_metadata_path",
        lambda metadata_path, model_path, backend=None: "/tmp/deepcdp/config.json",
    )
    monkeypatch.setattr(
        charge_density_module,
        "_resolve_charge_python",
        lambda python_executable, backend=None: "/tmp/deepcdp-env/bin/python",
    )
    monkeypatch.setattr(
        charge_density_module.np,
        "load",
        lambda path: _FakeLoadedDensity(np.ones((2, 2, 2), dtype=np.float32)),
    )

    def fake_run(command, **kwargs):
        seen["command"] = list(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(charge_density_module.subprocess, "run", fake_run)

    density, spin_density = charge_density_module._run_deepcdp_backend(
        atoms,
        grid_shape=(2, 2, 2),
        device="cuda:0",
        max_probes_per_batch=4096,
        charge_species=["O", "H"],
        soap_rcut=5.0,
        soap_nmax=4,
        soap_lmax=4,
        soap_sigma=0.5,
        soap_periodic=True,
        activation="relu",
        weighting={"function": "poly", "r0": 1.5, "c": 2, "m": 2},
    )

    assert density.shape == (2, 2, 2)
    assert spin_density is None
    command = seen["command"]
    assert command[:2] == [
        "/tmp/deepcdp-env/bin/python",
        str(Path(charge_density_module.__file__).with_name("deepcdp_runner.py")),
    ]
    assert "--model-path" in command
    assert command[command.index("--model-path") + 1] == "/tmp/deepcdp/model.pt"
    assert "--metadata-path" in command
    assert command[command.index("--metadata-path") + 1] == "/tmp/deepcdp/config.json"
    assert "--species" in command
    assert command[command.index("--species") + 1] == "O,H"
    assert "--soap-rcut" in command
    assert command[command.index("--soap-rcut") + 1] == "5.0"
    assert "--soap-nmax" in command
    assert command[command.index("--soap-nmax") + 1] == "4"
    assert "--soap-lmax" in command
    assert command[command.index("--soap-lmax") + 1] == "4"
    assert "--soap-sigma" in command
    assert command[command.index("--soap-sigma") + 1] == "0.5"
    assert "--soap-periodic" in command
    assert command[command.index("--soap-periodic") + 1] == "1"
    assert "--activation" in command
    assert command[command.index("--activation") + 1] == "relu"
    assert "--weighting-function" in command
    assert command[command.index("--weighting-function") + 1] == "poly"
    assert "--weighting-r0" in command
    assert command[command.index("--weighting-r0") + 1] == "1.5"
    assert "--weighting-c" in command
    assert command[command.index("--weighting-c") + 1] == "2"
    assert "--weighting-m" in command
    assert command[command.index("--weighting-m") + 1] == "2"


def test_charge3net_runner_safe_globals_is_compatible_with_older_torch():
    fake_torch = SimpleNamespace()

    charge3net_runner_module._ensure_torch_safe_globals(fake_torch)

    fake_serialization = SimpleNamespace()
    fake_torch_with_serialization = SimpleNamespace(serialization=fake_serialization)
    charge3net_runner_module._ensure_torch_safe_globals(fake_torch_with_serialization)

    seen: dict[str, object] = {}
    fake_torch_modern = SimpleNamespace(
        serialization=SimpleNamespace(
            add_safe_globals=lambda values: seen.setdefault("values", list(values))
        )
    )

    charge3net_runner_module._ensure_torch_safe_globals(fake_torch_modern)

    assert seen["values"] == [slice]


def test_charge_density_options_supports_charge_mlp_alias():
    options = charge_density_module._charge_density_options_from_bcar(
        {
            "CHARGE_MLP": "DeepDFT",
            "CHARGE_MODEL": "/tmp/deepdft/model",
            "CHARGE_MAX_PROBES_PER_BATCH": "4096",
        }
    )

    assert options["backend"] == "DeepDFT"
    assert options["model_path"] == "/tmp/deepdft/model"
    assert options["max_probes_per_batch"] == 4096


def test_charge_density_options_parse_deepcdp_overrides():
    options = charge_density_module._charge_density_options_from_bcar(
        {
            "CHARGE_MLP": "DeepCDP",
            "CHARGE_MODEL": "/tmp/deepcdp/model.pt",
            "CHARGE_DEEPCDP_METADATA": "/tmp/deepcdp/config.json",
            "CHARGE_DEEPCDP_SPECIES": "O,H",
            "CHARGE_DEEPCDP_RCUT": "5.0",
            "CHARGE_DEEPCDP_NMAX": "4",
            "CHARGE_DEEPCDP_LMAX": "4",
            "CHARGE_DEEPCDP_SIGMA": "0.5",
            "CHARGE_DEEPCDP_PERIODIC": "1",
            "CHARGE_DEEPCDP_ACTIVATION": "relu",
            "CHARGE_DEEPCDP_WEIGHTING_FUNCTION": "poly",
            "CHARGE_DEEPCDP_WEIGHTING_R0": "1.5",
            "CHARGE_DEEPCDP_WEIGHTING_C": "2",
            "CHARGE_DEEPCDP_WEIGHTING_M": "2",
        }
    )

    assert options["backend"] == "DeepCDP"
    assert options["model_path"] == "/tmp/deepcdp/model.pt"
    assert options["metadata_path"] == "/tmp/deepcdp/config.json"
    assert options["charge_species"] == ["O", "H"]
    assert options["soap_rcut"] == 5.0
    assert options["soap_nmax"] == 4
    assert options["soap_lmax"] == 4
    assert options["soap_sigma"] == 0.5
    assert options["soap_periodic"] is True
    assert options["activation"] == "relu"
    assert options["weighting"] == {
        "function": "poly",
        "r0": 1.5,
        "c": 2.0,
        "m": 2.0,
    }


def test_resolve_deepdft_model_dir_accepts_best_model_checkpoint_path(tmp_path: Path):
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_text("placeholder")

    resolved = charge_density_module._resolve_deepdft_model_dir(str(checkpoint), None)

    assert resolved == str(tmp_path)


def test_resolve_deepcdp_checkpoint_path_accepts_directory_with_single_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("placeholder")

    resolved = charge_density_module._resolve_deepcdp_checkpoint_path(str(tmp_path), None)

    assert resolved == str(checkpoint)


def test_charge3net_runner_loads_checkout_modules_without_importing_data_init(
    tmp_path: Path,
):
    checkout = tmp_path / "charge3net_checkout"
    package_root = checkout / "src" / "charge3net"
    models_dir = package_root / "models"
    data_dir = package_root / "data"
    models_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    (checkout / "src" / "__init__.py").write_text("")
    (package_root / "__init__.py").write_text("")
    (models_dir / "__init__.py").write_text("")
    (data_dir / "__init__.py").write_text("raise RuntimeError('training-only dependency import')")
    (models_dir / "e3.py").write_text("class E3DensityModel:\n    pass\n")
    (data_dir / "graph_construction.py").write_text("class KdTreeGraphConstructor:\n    pass\n")
    (data_dir / "collate.py").write_text(
        "def collate_list_of_dicts(*args, **kwargs):\n    return args, kwargs\n"
    )

    original_modules = {
        name: sys.modules.get(name)
        for name in (
            "src",
            "src.charge3net",
            "src.charge3net.models",
            "src.charge3net.models.e3",
            "src.charge3net.data",
            "src.charge3net.data.graph_construction",
            "src.charge3net.data.collate",
        )
    }
    try:
        E3DensityModel, KdTreeGraphConstructor, collate_list_of_dicts = (
            charge3net_runner_module._load_charge3net_modules(str(checkout))
        )
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    assert E3DensityModel.__name__ == "E3DensityModel"
    assert KdTreeGraphConstructor.__name__ == "KdTreeGraphConstructor"
    assert callable(collate_list_of_dicts)


def test_charge3net_runner_loads_installed_modules_without_importing_data_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    package_root = tmp_path / "site-packages" / "charge3net"
    models_dir = package_root / "models"
    data_dir = package_root / "data"
    models_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    (package_root / "__init__.py").write_text("")
    (models_dir / "__init__.py").write_text("")
    (data_dir / "__init__.py").write_text("raise RuntimeError('training-only dependency import')")
    (data_dir / "layer.py").write_text("SENTINEL = 7\n")
    (models_dir / "e3.py").write_text(
        "from src.charge3net.data.layer import SENTINEL\n"
        "class E3DensityModel:\n"
        "    sentinel = SENTINEL\n"
    )
    (data_dir / "graph_construction.py").write_text("class KdTreeGraphConstructor:\n    pass\n")
    (data_dir / "collate.py").write_text(
        "def collate_list_of_dicts(*args, **kwargs):\n    return args, kwargs\n"
    )

    monkeypatch.setattr(
        charge3net_runner_module.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(submodule_search_locations=[str(package_root)])
            if name == "charge3net"
            else None
        ),
    )

    original_modules = {
        name: sys.modules.get(name)
        for name in (
            "src",
            "src.charge3net",
            "src.charge3net.models",
            "src.charge3net.data",
            "src.charge3net.data.layer",
            "charge3net",
            "charge3net.models",
            "charge3net.models.e3",
            "charge3net.data",
            "charge3net.data.graph_construction",
            "charge3net.data.collate",
        )
    }
    try:
        E3DensityModel, KdTreeGraphConstructor, collate_list_of_dicts = (
            charge3net_runner_module._load_charge3net_modules(None)
        )
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    assert E3DensityModel.__name__ == "E3DensityModel"
    assert E3DensityModel.sentinel == 7
    assert KdTreeGraphConstructor.__name__ == "KdTreeGraphConstructor"
    assert callable(collate_list_of_dicts)


def test_charge3net_runner_coalesces_empty_atom_edges():
    edges, displacements, count = charge3net_runner_module._coalesce_edge_groups(
        [],
        [],
    )

    assert edges.shape == (0, 2)
    assert displacements.shape == (0, 3)
    assert count == 0


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
