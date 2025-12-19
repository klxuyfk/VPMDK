from __future__ import annotations

import pytest

import vpmdk
from tests.conftest import DummyCalculator


def test_run_md_executes_multiple_steps(tmp_path, load_atoms):
    atoms = load_atoms()

    class DummyDynamics:
        def __init__(self):
            self.steps: list[int] = []

        def run(self, n):
            self.steps.append(n)
            atoms.positions += 0.01

    written: list[str] = []
    xdat_steps: list[int] = []
    updates: list[float] = []
    captured: dict[str, DummyDynamics] = {}

    def fake_selector(atoms_arg, mdalgo, timestep, initial_temperature, smass, params):
        dyn = DummyDynamics()
        captured["dyn"] = dyn

        def updater(temp: float) -> None:
            updates.append(temp)

        return dyn, updater

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_select_md_dynamics", fake_selector)
    monkeypatch.setattr(
        vpmdk.velocitydistribution,
        "MaxwellBoltzmannDistribution",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(vpmdk, "_write_xdatcar_step", lambda filename, atoms, step: xdat_steps.append(step))
    monkeypatch.setattr(
        vpmdk,
        "write",
        lambda filename, atoms, direct=True: written.append(filename),
    )
    try:
        energy = vpmdk.run_md(
            atoms,
            DummyCalculator(),
            steps=3,
            temperature=450,
            timestep=1.0,
            mdalgo=0,
            teend=600,
        )
    finally:
        monkeypatch.undo()

    assert isinstance(energy, float)
    assert xdat_steps == [0, 1, 2]
    assert "CONTCAR" in written
    assert captured["dyn"].steps == [1, 1, 1]
    assert updates == [525.0, 600.0]


def test_get_lammps_interval_rejects_nonpositive():
    with pytest.raises(ValueError, match="at least 1"):
        vpmdk._get_lammps_trajectory_interval({"LAMMPS_TRAJ_INTERVAL": "0"})


def test_run_md_writes_lammps_dump_on_interval(tmp_path, load_atoms):
    atoms = load_atoms()

    class DummyDynamics:
        def __init__(self):
            self.steps: list[int] = []

        def run(self, n):
            self.steps.append(n)
            atoms.positions += 0.01

    lammps_steps: list[int] = []

    def fake_selector(atoms_arg, mdalgo, timestep, initial_temperature, smass, params):
        dyn = DummyDynamics()

        def updater(temp: float) -> None:
            return None

        return dyn, updater

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_select_md_dynamics", fake_selector)
    monkeypatch.setattr(
        vpmdk.velocitydistribution,
        "MaxwellBoltzmannDistribution",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(vpmdk, "_write_xdatcar_step", lambda filename, atoms, step: None)
    monkeypatch.setattr(
        vpmdk, "_write_lammps_trajectory_step", lambda path, atoms, step: lammps_steps.append(step)
    )
    monkeypatch.setattr(vpmdk, "write", lambda filename, atoms, direct=True: None)

    try:
        vpmdk.run_md(
            atoms,
            DummyCalculator(),
            steps=4,
            temperature=300,
            timestep=1.0,
            mdalgo=0,
            write_lammps_traj=True,
            lammps_traj_interval=2,
        )
    finally:
        monkeypatch.undo()

    assert lammps_steps == [0, 2]


def test_select_md_dynamics_andersen_uses_probability(load_atoms, monkeypatch):
    atoms = load_atoms()
    created: dict[str, object] = {}

    class DummyAndersen:
        def __init__(self, atoms, timestep, temperature_K, andersen_prob, logfile=None):
            created.update(
                {
                    "timestep": timestep,
                    "temperature": temperature_K,
                    "prob": andersen_prob,
                    "logfile": logfile,
                }
            )

        def set_temperature(self, value):
            created.setdefault("updates", []).append(value)

    rescaled: list[float] = []

    monkeypatch.setattr(vpmdk, "Andersen", DummyAndersen)
    monkeypatch.setattr(vpmdk, "_rescale_velocities", lambda atoms, temp: rescaled.append(temp))

    dyn, updater = vpmdk._select_md_dynamics(
        atoms,
        mdalgo=1,
        timestep=1.5,
        initial_temperature=350.0,
        smass=None,
        thermostat_params={"ANDERSEN_PROB": 0.2},
    )

    assert isinstance(dyn, DummyAndersen)
    assert created["prob"] == 0.2

    updater(360.0)
    assert created["updates"] == [360.0]
    assert rescaled == [360.0]


def test_select_md_dynamics_andersen_missing_dependency(load_atoms, monkeypatch):
    atoms = load_atoms()
    monkeypatch.setattr(vpmdk, "Andersen", None)

    with pytest.raises(RuntimeError, match="Andersen thermostat requested"):
        vpmdk._select_md_dynamics(
            atoms,
            mdalgo=1,
            timestep=1.0,
            initial_temperature=300.0,
            smass=None,
            thermostat_params={},
        )


def test_select_md_dynamics_langevin_converts_gamma(load_atoms, monkeypatch):
    atoms = load_atoms()
    captured: dict[str, object] = {}

    class DummyLangevin:
        def __init__(
            self,
            atoms,
            timestep,
            temperature_K=None,
            friction=None,
            logfile=None,
        ):
            captured.update(
                {
                    "timestep": timestep,
                    "temperature": temperature_K,
                    "friction": friction,
                    "logfile": logfile,
                }
            )

        def set_temperature(self, value):
            captured.setdefault("updates", []).append(value)

    monkeypatch.setattr(vpmdk, "Langevin", DummyLangevin)

    dyn, updater = vpmdk._select_md_dynamics(
        atoms,
        mdalgo=3,
        timestep=1.0,
        initial_temperature=300.0,
        smass=-2.5,
        thermostat_params={"LANGEVIN_GAMMA": 15.0},
    )

    assert isinstance(dyn, DummyLangevin)
    expected = 15.0 / 1000.0 / vpmdk.units.fs
    assert pytest.approx(captured["friction"], rel=1e-12) == expected

    updater(325.0)
    assert captured["updates"] == [325.0]


def test_select_md_dynamics_langevin_missing_dependency(load_atoms, monkeypatch):
    atoms = load_atoms()
    monkeypatch.setattr(vpmdk, "Langevin", None)

    with pytest.raises(RuntimeError, match="Langevin thermostat requested"):
        vpmdk._select_md_dynamics(
            atoms,
            mdalgo=3,
            timestep=1.0,
            initial_temperature=300.0,
            smass=None,
            thermostat_params={},
        )


def test_select_md_dynamics_nose_hoover_missing_dependency(load_atoms, monkeypatch):
    atoms = load_atoms()
    monkeypatch.setattr(vpmdk, "NoseHooverChainNVT", None)

    with pytest.raises(RuntimeError, match="Nose-Hoover thermostat requested"):
        vpmdk._select_md_dynamics(
            atoms,
            mdalgo=2,
            timestep=1.0,
            initial_temperature=300.0,
            smass=None,
            thermostat_params={},
        )


def test_select_md_dynamics_csvr_missing_dependency(load_atoms, monkeypatch):
    atoms = load_atoms()
    monkeypatch.setattr(vpmdk, "Bussi", None)

    with pytest.raises(RuntimeError, match="CSVR thermostat"):
        vpmdk._select_md_dynamics(
            atoms,
            mdalgo=5,
            timestep=1.0,
            initial_temperature=300.0,
            smass=None,
            thermostat_params={},
        )
