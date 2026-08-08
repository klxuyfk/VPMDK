from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
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
    outcar = (tmp_path / "OUTCAR").read_text()
    assert "direct lattice vectors                 reciprocal lattice vectors" in outcar
    assert "k-points in reciprocal lattice and weights" in outcar
    assert "FORCES: max atom, RMS" in outcar
    assert "total drift:" in outcar
    assert "energy  without entropy=" in outcar
    assert "General timing and accounting informations for this job" in outcar
    assert "Voluntary context switches" in outcar
    assert (tmp_path / "OSZICAR").exists()
    assert (tmp_path / "vasprun.xml").exists()


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


def test_run_md_uses_local_incar_pseudo_scf_settings_when_enabled(tmp_path, load_atoms):
    atoms = load_atoms()
    (tmp_path / "INCAR").write_text("NELM = 39\nNELMIN = 5\nNELMDL = -1\nEDIFF = 2E-06\n")

    class DummyDynamics:
        def run(self, n):
            assert n == 1
            atoms.positions += 0.01

    def fake_selector(atoms_arg, mdalgo, timestep, initial_temperature, smass, params):
        return DummyDynamics(), lambda temp: None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_select_md_dynamics", fake_selector)
    monkeypatch.setattr(
        vpmdk.velocitydistribution,
        "MaxwellBoltzmannDistribution",
        lambda *a, **k: None,
    )
    try:
        vpmdk.run_md(
            atoms,
            DummyCalculator(),
            steps=1,
            temperature=300,
            timestep=1.0,
            mdalgo=0,
            oszicar_pseudo_scf=True,
        )
    finally:
        monkeypatch.undo()

    outcar = (tmp_path / "OUTCAR").read_text()
    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    electronic = root.find("./parameters/separator[@name='electronic']")
    assert "NELM   =     39;" in outcar
    assert electronic is not None
    assert electronic.find("./i[@name='NELM']").text.strip() == "39"
    assert electronic.find("./i[@name='NELMIN']").text.strip() == "5"
    assert electronic.find("./i[@name='NELMDL']").text.strip() == "-1"
    assert root.find("./incar/i[@name='EDIFF']").text.strip() == "2.00000000E-06"


def test_lammps_dump_writes_box_bounds_from_prism(tmp_path, load_atoms):
    atoms = load_atoms()
    path = tmp_path / "traj.lammpstrj"

    vpmdk._write_lammps_trajectory_step(str(path), atoms, 0)

    prism = vpmdk.Prism(atoms.get_cell().array, atoms.get_pbc())
    lx, ly, lz, xy, xz, yz = prism.get_lammps_prism()
    expected_xlo = min(0.0, xy, xz, xy + xz)
    expected_xhi = lx + max(0.0, xy, xz, xy + xz)
    expected_ylo = min(0.0, yz)
    expected_yhi = ly + max(0.0, yz)

    lines = path.read_text().splitlines()
    assert lines[0] == "ITEM: TIMESTEP"
    assert lines[8].startswith("ITEM: ATOMS id type xs ys zs ix iy iz")
    assert lines[4].split()[:2] == ["ITEM:", "BOX"]

    xlo_line, xhi_line, xy_line = lines[5].split()
    assert float(xlo_line) == expected_xlo
    assert float(xhi_line) == expected_xhi
    assert float(xy_line) == xy

    ylo_line, yhi_line, xz_line = lines[6].split()
    assert float(ylo_line) == expected_ylo
    assert float(yhi_line) == expected_yhi
    assert float(xz_line) == xz

    zlo_line, zhi_line, yz_line = lines[7].split()
    assert float(zlo_line) == 0.0
    assert float(zhi_line) == lz
    assert float(yz_line) == yz

    # The load-bearing check: a standard reader must recover the cell we wrote.
    # ase.io.lammpsrun.construct_cell computes (xhi-xlo) - |xy| - |xz|, so the old
    # bounds gave a volume off by a factor of ~3 for this cell.
    from ase.io import read as ase_read

    recovered = ase_read(str(path), format="lammps-dump-text")
    assert recovered.get_volume() == pytest.approx(atoms.get_volume(), rel=1e-12)


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
    assert rescaled == []


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


def test_select_md_dynamics_nose_hoover_chain_updates_ramp_target(load_atoms, monkeypatch):
    atoms = load_atoms()
    atoms.set_velocities([[0.02, 0.0, 0.0] for _ in range(len(atoms))])
    captured: dict[str, object] = {}

    class DummyNoseHooverChain:
        def __init__(
            self,
            atoms,
            timestep,
            temperature_K,
            tdamp,
            tchain,
            logfile=None,
        ):
            self.atoms = atoms
            self._p = atoms.get_momenta()
            self._thermostat = type("Thermostat", (), {})()
            self._thermostat._num_atoms_global = len(atoms)
            self._thermostat._tdamp = tdamp
            self._thermostat._kT = vpmdk.units.kB * temperature_K
            self._thermostat._Q = np.zeros(tchain)
            self._thermostat._Q[0] = (
                3.0 * len(atoms) * self._thermostat._kT * tdamp**2
            )
            if tchain > 1:
                self._thermostat._Q[1:] = self._thermostat._kT * tdamp**2
            captured.update(
                {
                    "timestep": timestep,
                    "temperature": temperature_K,
                    "tdamp": tdamp,
                    "tchain": tchain,
                    "logfile": logfile,
                }
            )

    monkeypatch.setattr(vpmdk, "NoseHooverChainNVT", DummyNoseHooverChain)

    dyn, updater = vpmdk._select_md_dynamics(
        atoms,
        mdalgo=4,
        timestep=1.5,
        initial_temperature=300.0,
        smass=None,
        thermostat_params={"NHC_PERIOD": 20.0, "NHC_NCHAINS": 4},
    )

    assert isinstance(dyn, DummyNoseHooverChain)
    assert captured["tchain"] == 4
    assert captured["tdamp"] == pytest.approx(30.0 * vpmdk.units.fs)

    updater(450.0)

    kT = vpmdk.units.kB * 450.0
    tdamp = captured["tdamp"]
    assert dyn._thermostat._kT == pytest.approx(kT)
    assert dyn._thermostat._Q[0] == pytest.approx(3.0 * len(atoms) * kT * tdamp**2)
    assert dyn._thermostat._Q[1] == pytest.approx(kT * tdamp**2)
    assert np.allclose(dyn._p, atoms.get_momenta())


def test_select_md_dynamics_nose_hoover_chain_update_errors_when_ase_incompatible(
    load_atoms,
    monkeypatch,
):
    atoms = load_atoms()
    atoms.set_velocities([[0.02, 0.0, 0.0] for _ in range(len(atoms))])

    class IncompatibleNoseHooverChain:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(vpmdk, "NoseHooverChainNVT", IncompatibleNoseHooverChain)

    _, updater = vpmdk._select_md_dynamics(
        atoms,
        mdalgo=4,
        timestep=1.0,
        initial_temperature=300.0,
        smass=None,
        thermostat_params={},
    )

    with pytest.raises(RuntimeError, match="Temperature ramping with Nose-Hoover chain"):
        updater(350.0)


def test_select_md_dynamics_nose_hoover_chain_rejects_disabled_chain(
    load_atoms,
    monkeypatch,
):
    atoms = load_atoms()
    monkeypatch.setattr(
        vpmdk,
        "NoseHooverChainNVT",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should reject before creating dynamics")
        ),
    )

    with pytest.raises(RuntimeError, match="NHC_NCHAINS must be at least 1"):
        vpmdk._select_md_dynamics(
            atoms,
            mdalgo=4,
            timestep=1.0,
            initial_temperature=300.0,
            smass=None,
            thermostat_params={"NHC_NCHAINS": 0},
        )


def test_select_md_dynamics_nose_hoover_chain_rejects_nonpositive_initial_temperature(
    load_atoms,
    monkeypatch,
):
    atoms = load_atoms()
    monkeypatch.setattr(
        vpmdk,
        "NoseHooverChainNVT",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should reject before creating dynamics")
        ),
    )

    with pytest.raises(RuntimeError, match="positive initial"):
        vpmdk._select_md_dynamics(
            atoms,
            mdalgo=4,
            timestep=1.0,
            initial_temperature=0.0,
            smass=None,
            thermostat_params={},
        )


def test_public_md_nose_hoover_chain_rejects_nonpositive_ramp_target(load_atoms):
    atoms = load_atoms()

    with pytest.raises(RuntimeError, match="positive TEBEG and TEEND"):
        vpmdk.md(
            atoms,
            calculator=DummyCalculator(),
            steps=2,
            temperature=300.0,
            thermostat="nose_hoover_chain",
            temperature_end=0.0,
        )


def test_estimate_tdamp_uses_vasp_style_nhc_period():
    assert vpmdk._estimate_tdamp(
        None,
        2.0,
        {"NHC_PERIOD": 40.0},
    ) == pytest.approx(80.0)


def test_estimate_tdamp_rejects_disabled_nhc_period():
    with pytest.raises(RuntimeError, match="NHC_PERIOD must be positive"):
        vpmdk._estimate_tdamp(None, 2.0, {"NHC_PERIOD": 0.0})


def test_semantic_incar_md_checks_raise_workdir_input_error(load_atoms, monkeypatch):
    # Semantic INCAR-input failures raised by the RUNTIME path (not the wrapped
    # parse phase) must be WorkdirInputError, so server mode classifies them as
    # input_error (exit 1) rather than a retryable calculation_error (exit 2).
    # WorkdirInputError subclasses RuntimeError, so the existing RuntimeError-match
    # tests still hold; these assert the more specific type the classifier keys on.
    assert issubclass(vpmdk.WorkdirInputError, RuntimeError)

    # NHC_PERIOD <= 0
    with pytest.raises(vpmdk.WorkdirInputError, match="NHC_PERIOD must be positive"):
        vpmdk._estimate_tdamp(None, 2.0, {"NHC_PERIOD": 0.0})

    atoms = load_atoms()
    monkeypatch.setattr(
        vpmdk,
        "NoseHooverChainNVT",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should reject before creating dynamics")
        ),
    )
    # TEBEG=0 (non-positive initial temperature)
    with pytest.raises(vpmdk.WorkdirInputError, match="positive initial"):
        vpmdk._select_md_dynamics(
            atoms, mdalgo=4, timestep=1.0, initial_temperature=0.0,
            smass=None, thermostat_params={},
        )
    # NHC_NCHAINS=0
    with pytest.raises(vpmdk.WorkdirInputError, match="NHC_NCHAINS must be at least 1"):
        vpmdk._select_md_dynamics(
            atoms, mdalgo=4, timestep=1.0, initial_temperature=300.0,
            smass=None, thermostat_params={"NHC_NCHAINS": 0},
        )
    with pytest.raises(vpmdk.WorkdirInputError, match="exceeds the supported maximum"):
        vpmdk._select_md_dynamics(
            atoms, mdalgo=4, timestep=1.0, initial_temperature=300.0,
            smass=None, thermostat_params={"NHC_NCHAINS": 100_000_000},
        )


def test_andersen_default_probability_is_disclosed(load_atoms, monkeypatch, capsys):
    class RecordingAndersen:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(vpmdk, "Andersen", RecordingAndersen)
    dyn, _ = vpmdk._select_md_dynamics(
        load_atoms(), mdalgo=1, timestep=1.0, initial_temperature=300.0,
        smass=None, thermostat_params={},
    )
    out = capsys.readouterr().out
    assert "ANDERSEN_PROB" in out and "0.1" in out and "ANDERSEN_PROB = 0.0" in out
    assert dyn.kwargs["andersen_prob"] == 0.1

    # An explicit value -- including 0.0 -- stays silent.
    for explicit in (0.0, 0.2):
        vpmdk._select_md_dynamics(
            load_atoms(), mdalgo=1, timestep=1.0, initial_temperature=300.0,
            smass=None, thermostat_params={"ANDERSEN_PROB": explicit},
        )
        assert "ANDERSEN_PROB" not in capsys.readouterr().out


def test_md_warns_when_cell_dynamics_are_requested(capsys):
    vpmdk._warn_md_is_fixed_cell(isif=3, pstress=None)
    out = capsys.readouterr().out
    assert "FIXED-CELL" in out and "ISIF=3" in out

    vpmdk._warn_md_is_fixed_cell(isif=0, pstress=10.0)
    out = capsys.readouterr().out
    assert "FIXED-CELL" in out and "PSTRESS=10" in out

    # Plain NVT/NVE input stays silent.
    vpmdk._warn_md_is_fixed_cell(isif=2, pstress=None)
    vpmdk._warn_md_is_fixed_cell(isif=None, pstress=0.0)
    assert capsys.readouterr().out == ""


def test_nhc_nchains_bound_does_not_clip_legitimate_chains(load_atoms, monkeypatch):
    # Companion to the huge-NHC_NCHAINS rejection above: a long-but-physical
    # chain must still construct (the monkeypatched thermostat records the
    # tchain it would have been built with, without writing an OUTCAR logfile).
    captured: dict[str, object] = {}

    class RecordingNHC:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(vpmdk, "NoseHooverChainNVT", RecordingNHC)
    dyn, _ = vpmdk._select_md_dynamics(
        load_atoms(), mdalgo=4, timestep=1.0, initial_temperature=300.0,
        smass=None, thermostat_params={"NHC_NCHAINS": 10},
    )
    assert isinstance(dyn, RecordingNHC)
    assert captured["tchain"] == 10


def test_public_md_nhc_nonpositive_ramp_is_workdir_input_error(load_atoms):
    # The up-front TEBEG/TEEND<=0 guard in the public md() path is also input, exit 1.
    with pytest.raises(vpmdk.WorkdirInputError, match="positive TEBEG and TEEND"):
        vpmdk.md(
            load_atoms(),
            calculator=DummyCalculator(),
            steps=2,
            temperature=300.0,
            thermostat="nose_hoover_chain",
            temperature_end=0.0,
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


@pytest.mark.parametrize(
    "mdalgo,temperature,rejected",
    [
        # Measured in ase 3.29: a negative absolute temperature makes ASE take the
        # square root of a negative number with NO exception, so the trajectory
        # becomes nan and only surfaces later as a raw error from the energy
        # formatter -- calculation_error, i.e. exit 2, documented as RETRYABLE for a
        # permanently invalid INCAR.
        (1, -300.0, True),   # Andersen: T<0 -> nan
        (3, -300.0, True),   # Langevin: T<0 -> nan
        (5, -300.0, True),   # Bussi: cannot rescale a zero/negative kinetic energy
        (5, 0.0, True),      # Bussi: "Initial kinetic energy is zero"
        (1, 0.0, False),     # 0 K is the legal limit for Andersen ...
        (3, 0.0, False),     # ... and for Langevin; both complete today
    ],
)
def test_md_temperature_guard_matches_what_ase_actually_breaks_on(
    load_atoms, mdalgo: int, temperature: float, rejected: bool
):
    from ase.calculators.emt import EMT
    from ase.build import bulk
    from vpmdk_core.compat import vasp as vasp_compat_models
    from vpmdk_core import execution as execution_module

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    config = vpmdk.MDConfig(
        steps=2,
        timestep_fs=1.0,
        temperature=temperature,
        compat=vasp_compat_models.VaspMDConfig(mdalgo=mdalgo),
    )

    if rejected:
        with pytest.raises(vpmdk.WorkdirInputError):
            execution_module.execute_md(atoms, EMT(), config=config)
    else:
        result = execution_module.execute_md(atoms, EMT(), config=config)
        assert np.isfinite(result.potential_energy)


def test_negative_langevin_gamma_is_an_input_error(load_atoms):
    atoms = load_atoms()

    with pytest.raises(vpmdk.WorkdirInputError, match="LANGEVIN_GAMMA must not be"):
        vpmdk._select_md_dynamics(
            atoms,
            mdalgo=3,
            timestep=1.0,
            initial_temperature=300.0,
            smass=None,
            thermostat_params={"LANGEVIN_GAMMA": -10.0},
        )

    dyn, _ = vpmdk._select_md_dynamics(
        atoms,
        mdalgo=3,
        timestep=1.0,
        initial_temperature=300.0,
        smass=None,
        thermostat_params={"LANGEVIN_GAMMA": 0.0},
    )
    assert dyn is not None


def test_non_finite_energy_reports_the_divergence_not_an_unpack_error():
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="calculation diverged"):
            vpmdk._format_energy_value(value)

    assert vpmdk._format_energy_value(0.0) == "+.00000000E+00"
    assert vpmdk._format_energy_value(-16.328932) == "-.16328932E+02"


def test_thermostat_energy_is_reported_so_the_total_is_conserved():
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from vpmdk_core.compat import vasp as vasp_compat_models
    from vpmdk_core import execution as execution_module

    for mdalgo, thermostat_kwargs in ((2, {"NHC_PERIOD": 20.0}), (5, {"CSVR_PERIOD": 40.0})):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1)
        config = vpmdk.MDConfig(
            steps=30,
            timestep_fs=2.0,
            temperature=600.0,
            thermostat_kwargs=thermostat_kwargs,
            compat=vasp_compat_models.VaspMDConfig(mdalgo=mdalgo),
        )
        result = execution_module.execute_md(atoms, EMT(), config=config)

        totals = np.array([step.total_energy for step in result.steps])
        bare = np.array(
            [step.potential_energy + step.kinetic_energy for step in result.steps]
        )
        thermostat = np.array(
            [step.thermostat_potential + step.thermostat_kinetic for step in result.steps]
        )
        assert np.any(thermostat != 0.0), f"MDALGO={mdalgo} reported a zero thermostat"
        # The thermostat term is what makes the reported total the conserved one.
        total_drift = float(totals.max() - totals.min())
        bare_drift = float(bare.max() - bare.min())
        assert total_drift < bare_drift / 10.0, f"MDALGO={mdalgo} total not conserved"

    # The Nose-Hoover split must reproduce ASE's own public sum exactly.
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1)
    atoms.calc = EMT()
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.0)
    dyn, _ = vpmdk._select_md_dynamics(
        atoms,
        mdalgo=2,
        timestep=2.0,
        initial_temperature=600.0,
        smass=None,
        thermostat_params={"NHC_PERIOD": 20.0},
    )
    dyn.run(10)
    potential, kinetic = vpmdk._thermostat_energy_terms(dyn)
    assert potential + kinetic == pytest.approx(
        dyn._thermostat.get_thermostat_energy(), abs=1e-12
    )
    assert kinetic >= 0.0  # the chain's kinetic term is a sum of squares


@pytest.mark.parametrize(
    "mdalgo,potim,rejected",
    [
        # Measured in ase 3.29: POTIM=0 collapses every thermostat mass to zero
        # (nan for MDALGO 1/2/3/4, ZeroDivisionError for 5); a negative POTIM only
        # breaks MDALGO=3. MDALGO=0 completes in every case and must keep doing so.
        (1, 0.0, True),
        (2, 0.0, True),
        (3, 0.0, True),
        (4, 0.0, True),
        (5, 0.0, True),
        (0, 0.0, False),
        (3, -1.0, True),
        (2, -1.0, False),
        (0, -1.0, False),
    ],
)
def test_md_potim_guard_matches_what_ase_actually_breaks_on(
    mdalgo: int, potim: float, rejected: bool
):
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = EMT()
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)

    def build():
        return vpmdk._select_md_dynamics(
            atoms,
            mdalgo=mdalgo,
            timestep=potim,
            initial_temperature=300.0,
            smass=None,
            thermostat_params={},
        )

    if rejected:
        with pytest.raises(vpmdk.WorkdirInputError, match="POTIM"):
            build()
    else:
        dyn, _ = build()
        assert dyn is not None


@pytest.mark.parametrize(
    "incar,expected,warns",
    [
        # Every algorithm _select_md_dynamics implements passes through silently.
        ({"MDALGO": 0}, 0, False),
        ({"MDALGO": 1}, 1, False),
        ({"MDALGO": 2}, 2, False),
        ({"MDALGO": 3}, 3, False),
        ({"MDALGO": 4}, 4, False),
        ({"MDALGO": 5}, 5, False),
        # Anything else fell through to bare velocity-Verlet with no warning.
        ({"MDALGO": 13}, 0, True),
        ({"MDALGO": 6}, 0, True),
        ({"MDALGO": -1}, 0, True),
        # SMASS still selects the thermostat when MDALGO is omitted or 0...
        ({"SMASS": 1.0}, 2, False),
        ({"SMASS": -3.0}, 3, False),
        ({"MDALGO": 0, "SMASS": 1.0}, 2, False),
        # ...but must NOT promote an explicit out-of-range MDALGO, or the
        # warning would announce an NVE run that does not happen.
        ({"MDALGO": 13, "SMASS": 1.0}, 0, True),
    ],
)
def test_unsupported_mdalgo_falls_back_to_nve_out_loud(
    capsys, incar: dict, expected: int, warns: bool
):
    settings = vpmdk._load_incar_settings({"IBRION": 0, **incar})

    assert settings.mdalgo == expected
    warned = "MDALGO" in capsys.readouterr().out
    assert warned is warns


@pytest.mark.parametrize(
    "smass,nhc_period,potim,expected_tdamp,warns",
    [
        # SMASS is read as an absolute damping time in fs, so ordinary VASP
        # Nose-mass values give a thermostat far stiffer than VPMDK's own
        # 100*POTIM default -- measured: it pins the temperature and, depending
        # on the cell/seed, makes the ASE chain integrator diverge to NaN.
        (0.5, None, 2.0, 0.5, True),
        (1.0, None, 2.0, 1.0, True),
        (25.0, None, 2.0, 25.0, False),
        (None, None, 2.0, 200.0, False),
        # NHC_PERIOD is in MD steps; the same stiffness applies to the product.
        (None, 5.0, 2.0, 10.0, True),
        (None, 100.0, 2.0, 200.0, False),
    ],
)
def test_strong_thermostat_coupling_is_reported(
    capsys, smass, nhc_period, potim, expected_tdamp, warns
):
    params = {} if nhc_period is None else {"NHC_PERIOD": nhc_period}

    tdamp = vpmdk._estimate_tdamp(smass, potim, params)

    assert tdamp == pytest.approx(expected_tdamp)
    out = capsys.readouterr().out
    assert ("Nose-Hoover damping time" in out) is warns
    if warns:
        assert "not as VASP's Nose mass" in out


def test_temperature_ramp_keeps_the_requested_ensemble():
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from vpmdk_core.compat import vasp as vasp_compat_models
    from vpmdk_core import execution as execution_module

    def run(mdalgo, temperature_end, thermostat_kwargs):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1)
        atoms.calc = EMT()
        np.random.seed(11)
        config = vpmdk.MDConfig(
            steps=25,
            timestep_fs=2.0,
            temperature=300.0,
            temperature_end=temperature_end,
            thermostat_kwargs=thermostat_kwargs,
            compat=vasp_compat_models.VaspMDConfig(mdalgo=mdalgo),
        )
        result = execution_module.execute_md(atoms, EMT(), config=config)
        return np.array([step.temperature for step in result.steps]), np.array(
            [step.total_energy for step in result.steps]
        )

    for mdalgo, kwargs in ((2, {"NHC_PERIOD": 20.0}), (5, {"CSVR_PERIOD": 40.0})):
        flat_t, flat_e = run(mdalgo, 300.0, kwargs)
        # An infinitesimal ramp must not change the physics at all: this is the
        # discontinuity the rescale introduced.
        ramped_t, ramped_e = run(mdalgo, 300.0001, kwargs)
        # Tolerances are loose on purpose: a 0.0001 K ramp DOES retarget the
        # thermostat every step, which moves the energy by ~1e-7 eV. What must
        # not happen is the discontinuity the rescale caused -- 73 K of lost
        # temperature spread and 2.37 eV of drift.
        assert ramped_t == pytest.approx(flat_t, abs=1e-3), f"MDALGO={mdalgo}"
        assert ramped_e == pytest.approx(flat_e, abs=1e-4), f"MDALGO={mdalgo}"

        # A real ramp keeps canonical fluctuations (an isokinetic trajectory
        # collapses them) and does not blow up the conserved quantity.
        real_t, real_e = run(mdalgo, 600.0, kwargs)
        assert float(real_t.std()) > float(flat_t.std()) / 3.0, f"MDALGO={mdalgo}"
        assert float(real_e.max() - real_e.min()) < 1.0, f"MDALGO={mdalgo}"


def test_temperature_ramp_updaters_do_not_rescale_velocities(load_atoms, monkeypatch):
    # The unit-level statement of the same rule, per integrator. Only MDALGO=0
    # keeps rescaling: plain NVE has no thermostat to retarget, so rescaling IS
    # the ramp there.
    rescaled: list[float] = []
    monkeypatch.setattr(
        vpmdk, "_rescale_velocities", lambda atoms, temp: rescaled.append(temp)
    )

    class DummyThermostat:
        def __init__(self, *args, **kwargs):
            self.updates: list[float] = []

        def set_temperature(self, temperature_K=None, **kwargs):
            self.updates.append(temperature_K)

    for mdalgo, name, params in (
        (1, "Andersen", {"ANDERSEN_PROB": 0.2}),
        (3, "Langevin", {"LANGEVIN_GAMMA": 0.1}),
        (5, "Bussi", {"CSVR_PERIOD": 40.0}),
    ):
        atoms = load_atoms()
        monkeypatch.setattr(vpmdk, name, DummyThermostat)
        rescaled.clear()
        dyn, updater = vpmdk._select_md_dynamics(
            atoms,
            mdalgo=mdalgo,
            timestep=1.5,
            initial_temperature=350.0,
            smass=None,
            thermostat_params=params,
        )
        updater(360.0)
        assert dyn.updates == [360.0], name
        assert rescaled == [], name

    # MDALGO=0 (NVE) still ramps by rescaling.
    atoms = load_atoms()
    rescaled.clear()
    _, updater = vpmdk._select_md_dynamics(
        atoms,
        mdalgo=0,
        timestep=1.5,
        initial_temperature=350.0,
        smass=None,
        thermostat_params={},
    )
    updater(360.0)
    assert rescaled == [360.0]


def test_nose_hoover_chain_ramp_retargets_without_touching_momenta(load_atoms):
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = EMT()
    np.random.seed(5)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)
    dyn, updater = vpmdk._select_md_dynamics(
        atoms,
        mdalgo=2,
        timestep=2.0,
        initial_temperature=300.0,
        smass=None,
        thermostat_params={"NHC_PERIOD": 20.0},
    )
    dyn.run(3)
    before = atoms.get_momenta().copy()

    updater(600.0)

    assert dyn._thermostat._kT == pytest.approx(vpmdk.units.kB * 600.0)
    assert np.allclose(atoms.get_momenta(), before)


def test_nose_hoover_chain_rejects_constrained_atoms(load_atoms, monkeypatch):
    from ase.constraints import FixAtoms

    class DummyNoseHooverChain:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "the integrator must not be constructed for constrained atoms"
            )

    monkeypatch.setattr(vpmdk, "NoseHooverChainNVT", DummyNoseHooverChain)

    atoms = load_atoms()
    atoms.set_velocities([[0.02, 0.0, 0.0] for _ in range(len(atoms))])
    atoms.set_constraint(FixAtoms(indices=[0]))

    for mdalgo in (2, 4):
        with pytest.raises(vpmdk.UnsupportedInputError, match="constrained atoms"):
            vpmdk._select_md_dynamics(
                atoms,
                mdalgo=mdalgo,
                timestep=1.5,
                initial_temperature=300.0,
                smass=None,
                thermostat_params={},
            )

    # The same constrained atoms stay legal for a thermostat that handles
    # constraints correctly (Langevin writes momenta back through set_momenta
    # every step) -- the rejection must not widen beyond MDALGO=2/4.
    dyn, _ = vpmdk._select_md_dynamics(
        atoms,
        mdalgo=3,
        timestep=1.5,
        initial_temperature=300.0,
        smass=None,
        thermostat_params={"LANGEVIN_GAMMA": 10.0},
    )
    assert dyn is not None


def test_lammps_dump_velocities_are_metal_units(tmp_path, load_atoms):
    import ase.io
    from ase import units as ase_units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from vpmdk_core.io import trajectories as trajectories_module

    atoms = load_atoms()
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.0)
    source_velocities = atoms.get_velocities()
    assert float(np.abs(source_velocities).max()) > 0.0

    path = tmp_path / "lammps.lammpstrj"
    trajectories_module._write_lammps_trajectory_step(str(path), atoms, 0)

    frame = ase.io.read(
        str(path), format="lammps-dump-text", specorder=["Si"]
    )
    readback = frame.get_velocities()
    assert np.allclose(readback, source_velocities, rtol=1e-9, atol=1e-15)
    # The raw column is the metal-units value (Angstrom/ps).
    raw_line = [
        line for line in path.read_text().splitlines() if len(line.split()) == 12
    ][0]
    raw_vx = float(raw_line.split()[-4])
    assert raw_vx == pytest.approx(
        source_velocities[0, 0] * (1000.0 * ase_units.fs), rel=1e-9
    )


def test_andersen_com_freeze_temperature_convention_is_disclosed(
    load_atoms, monkeypatch, capsys
):
    class DummyAndersen:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(vpmdk, "Andersen", DummyAndersen)
    monkeypatch.setattr(vpmdk, "_rescale_velocities", lambda atoms, temp: None)

    atoms = load_atoms()
    vpmdk._select_md_dynamics(
        atoms,
        mdalgo=1,
        timestep=1.5,
        initial_temperature=300.0,
        smass=None,
        thermostat_params={"ANDERSEN_PROB": 0.2},
    )
    out = capsys.readouterr().out
    assert "freezes the center of mass" in out
    assert "3N-3" in out


def test_single_atom_langevin_is_an_input_error(load_atoms, monkeypatch):
    from ase import Atoms

    class DummyLangevin:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(vpmdk, "Langevin", DummyLangevin)
    monkeypatch.setattr(vpmdk, "_rescale_velocities", lambda atoms, temp: None)

    one = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 5.4, pbc=True)
    with pytest.raises(vpmdk.WorkdirInputError, match="at least two atoms"):
        vpmdk._select_md_dynamics(
            one,
            mdalgo=3,
            timestep=1.0,
            initial_temperature=300.0,
            smass=None,
            thermostat_params={},
        )

    # Two atoms keep constructing.
    atoms = load_atoms()
    atoms.set_velocities([[0.02, 0.0, 0.0] for _ in range(len(atoms))])
    dyn, _ = vpmdk._select_md_dynamics(
        atoms,
        mdalgo=3,
        timestep=1.0,
        initial_temperature=300.0,
        smass=None,
        thermostat_params={},
    )
    assert dyn is not None


def test_lammps_dump_records_species_via_element_column(tmp_path):
    import ase.io
    from ase import Atoms
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from vpmdk_core.io import trajectories as trajectories_module

    atoms = Atoms(
        "LiFeOAl",
        positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 2]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)

    path = tmp_path / "lammps.lammpstrj"
    trajectories_module._write_lammps_trajectory_step(str(path), atoms, 0)

    header = [
        line for line in path.read_text().splitlines()
        if line.startswith("ITEM: ATOMS")
    ][0]
    assert header.split()[-1] == "element"

    frame = ase.io.read(str(path), format="lammps-dump-text")
    assert frame.get_chemical_symbols() == ["Li", "Fe", "O", "Al"]
    assert frame.get_kinetic_energy() == pytest.approx(
        atoms.get_kinetic_energy(), rel=1e-9
    )


def test_md_divergence_guard_stops_before_the_force_call(load_atoms, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    atoms = load_atoms()
    calculator = DummyCalculator()

    # Deterministic divergence: MaxwellBoltzmann velocities are random and,
    # for a small cell, a 1e6 fs step does not ALWAYS push the per-step
    # relative excursion past the volume bound (atoms.wrap() between steps
    # keeps it from accumulating). One atom at 0.1 A/ASE-time covers
    # ~9.8e3 A per axis in a single 1e6 fs step.
    def fake_maxwell(target, temperature_K=None, **kwargs):
        velocities = np.zeros((len(target), 3))
        velocities[0] = [0.1, 0.1, 0.1]
        target.set_velocities(velocities)

    monkeypatch.setattr(
        vpmdk.velocitydistribution, "MaxwellBoltzmannDistribution", fake_maxwell
    )

    with pytest.raises(RuntimeError, match="diverged"):
        vpmdk.run_md(
            atoms,
            calculator,
            steps=3,
            temperature=300,
            timestep=1.0e6,
            mdalgo=0,
        )

    # The guard proxy must not leak past the dynamics run: _build_result
    # publishes atoms.calc, and a resident server reuses the calculator.
    assert atoms.calc is calculator


def test_md_divergence_guard_bounds_each_axis(tmp_path, monkeypatch):
    from ase import Atoms

    monkeypatch.chdir(tmp_path)
    # Genuinely collinear: every atom shares y = z = 0, so the y/z spans are
    # EXACTLY zero and the unfloored product stayed zero however far the
    # atoms separated. (A bulk cell has nonzero transverse spread, which
    # would let even the unfloored product fire.)
    atoms = Atoms(
        "Si2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0], pbc=True,
    )
    calculator = DummyCalculator()

    def fake_maxwell(target, temperature_K=None, **kwargs):
        velocities = np.zeros((len(target), 3))
        velocities[0] = [2.0e4, 0.0, 0.0]  # x only: ~2e9 A in one 1e6 fs step
        target.set_velocities(velocities)

    monkeypatch.setattr(
        vpmdk.velocitydistribution, "MaxwellBoltzmannDistribution", fake_maxwell
    )

    with pytest.raises(RuntimeError, match="diverged"):
        vpmdk.run_md(
            atoms,
            calculator,
            steps=3,
            temperature=300,
            timestep=1.0e6,
            mdalgo=0,
        )


def test_md_divergence_guard_respects_the_cell_bounding_box():
    # The input-time cap admits cells whose bounding box exceeds 1e9 A^3 (one
    # long axis, sub-Angstrom floors on the others), and wrapped positions
    # legitimately span that box today -- the per-axis floors must not make
    # the run-time guard reject what the cell itself allows.
    from ase import Atoms

    from vpmdk_core import execution as execution_module

    guard = execution_module._MDDivergenceGuardCalculator(DummyCalculator())

    diverged = Atoms(
        "H2", positions=[[0.0, 0.0, 0.0], [2.0e9, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0], pbc=True,
    )
    with pytest.raises(RuntimeError, match="diverged"):
        guard._vpmdk_check_positions(diverged)

    inside_long_cell = Atoms(
        "H2", positions=[[0.0, 0.0, 0.0], [1.9e9, 0.0, 0.0]],
        cell=[[2.0e9, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], pbc=True,
    )
    guard._vpmdk_check_positions(inside_long_cell)


def test_md_divergence_guard_bounds_a_single_axis_divergence(tmp_path, monkeypatch):
    from ase import Atoms

    monkeypatch.chdir(tmp_path)
    atoms = Atoms(
        "Si2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0], pbc=True,
    )
    calculator = DummyCalculator()

    def fake_maxwell(target, temperature_K=None, **kwargs):
        velocities = np.zeros((len(target), 3))
        # ~2e8 A in one 1e6 fs step on ONE axis: far above the 1e7 A axis
        # cap, far below the 1e9 A^3 floored product.
        velocities[0] = [2.0e3, 0.0, 0.0]
        target.set_velocities(velocities)

    monkeypatch.setattr(
        vpmdk.velocitydistribution, "MaxwellBoltzmannDistribution", fake_maxwell
    )

    with pytest.raises(RuntimeError, match="per.?axis"):
        vpmdk.run_md(
            atoms,
            calculator,
            steps=3,
            temperature=300,
            timestep=1.0e6,
            mdalgo=0,
        )
