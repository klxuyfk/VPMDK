from __future__ import annotations

import re
from pathlib import Path

import pytest
import numpy as np
import xml.etree.ElementTree as ET

import vpmdk
from tests.conftest import DummyCalculator


def test_relaxation_isif2_moves_ions_without_changing_cell(
    tmp_path: Path, load_atoms, arrays_close
):
    atoms = load_atoms()
    initial_positions = atoms.get_positions().copy()
    initial_cell = atoms.cell.array.copy()

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            target.positions += 0.05

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(atoms, DummyCalculator(), steps=2, fmax=0.01, isif=2)
    finally:
        monkeypatch.undo()

    assert not arrays_close(atoms.get_positions(), initial_positions)
    assert arrays_close(atoms.cell.array, initial_cell)
    outcar = (tmp_path / "OUTCAR").read_text()
    assert "direct lattice vectors                 reciprocal lattice vectors" in outcar
    assert "k-points in reciprocal lattice and weights" in outcar
    assert "FORCES: max atom, RMS" in outcar
    assert "total drift:" in outcar
    assert "energy  without entropy=" in outcar
    assert "General timing and accounting informations for this job" in outcar
    assert (tmp_path / "OSZICAR").exists()
    assert (tmp_path / "vasprun.xml").exists()


def test_relaxation_neb_mode_writes_projection_line(tmp_path: Path, load_atoms):
    atoms = load_atoms()

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            target.positions += 0.01

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            DummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=2,
            neb_mode=True,
        )
    finally:
        monkeypatch.undo()

    outcar = (tmp_path / "OUTCAR").read_text()
    assert "NEB: projections on to tangent" in outcar
    assert "tangential force (eV/A)" in outcar
    assert "CHAIN + TOTAL  (eV/Angst)" in outcar


def test_estimate_neb_chain_approximation_uses_neighbor_displacements():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    prev = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    nxt = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    forces = np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)

    approx = vpmdk._estimate_neb_chain_approximation(
        positions=positions,
        forces=forces,
        prev_positions=prev,
        next_positions=nxt,
    )

    assert approx is not None
    assert pytest.approx(approx.tangential_force, rel=1e-12) == 4.242640687119286
    assert np.allclose(
        approx.chain_force_vectors,
        np.array([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        atol=1e-12,
    )
    assert np.allclose(approx.chain_plus_total, np.array([12.0, 0.0, 0.0], dtype=float), atol=1e-12)


def test_relaxation_neb_chain_block_uses_neighbor_approximation(tmp_path: Path, load_atoms):
    atoms = load_atoms()

    class ForceDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            count = len(atoms) if atoms is not None else 0
            self.results["forces"] = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=float), (count, 1))

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj
            self._callbacks = []

        def attach(self, callback, *args, **kwargs):
            self._callbacks.append(callback)

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            target.positions += 0.01
            for callback in self._callbacks:
                callback()

    neighbor_delta = np.array([0.2, 0.0, 0.0], dtype=float)
    prev_positions = atoms.get_positions() - neighbor_delta
    next_positions = atoms.get_positions() + neighbor_delta

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            ForceDummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=2,
            neb_mode=True,
            neb_prev_positions=prev_positions,
            neb_next_positions=next_positions,
        )
    finally:
        monkeypatch.undo()

    outcar = (tmp_path / "OUTCAR").read_text()
    match = re.search(r"tangential force \(eV/A\)\s+([-+0-9.]+)", outcar)
    assert match is not None
    assert abs(float(match.group(1))) > 1.0e-6
    assert " 4.00000" in outcar


def test_relaxation_oszicar_pseudo_scf_is_off_by_default(tmp_path: Path, load_atoms):
    atoms = load_atoms()

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj
            self._callbacks = []

        def attach(self, callback, *args, **kwargs):
            self._callbacks.append(callback)

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            target.positions += 0.01
            for callback in self._callbacks:
                callback()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            DummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=2,
        )
    finally:
        monkeypatch.undo()

    oszicar = (tmp_path / "OSZICAR").read_text()
    assert "DAV:" not in oszicar
    assert "N       E" not in oszicar


def test_relaxation_oszicar_pseudo_scf_is_written_when_enabled(tmp_path: Path, load_atoms):
    atoms = load_atoms()

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj
            self._callbacks = []

        def attach(self, callback, *args, **kwargs):
            self._callbacks.append(callback)

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            target.positions += 0.01
            for callback in self._callbacks:
                callback()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            DummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=2,
            oszicar_pseudo_scf=True,
        )
    finally:
        monkeypatch.undo()

    oszicar = (tmp_path / "OSZICAR").read_text()
    assert "DAV:" in oszicar
    assert "N       E" in oszicar


def test_relaxation_writes_stress_block_when_isif_allows(tmp_path: Path, load_atoms):
    atoms = load_atoms()
    class StressDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            self.results["stress"] = np.zeros(6, dtype=float)

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            target.positions += 0.01

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            StressDummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=2,
            stress_isif=2,
        )
    finally:
        monkeypatch.undo()

    outcar = (tmp_path / "OUTCAR").read_text()
    assert "FORCE on cell =-STRESS in cart. coord." in outcar
    assert "external pressure" in outcar


def test_relaxation_omits_stress_block_when_isif_zero(tmp_path: Path, load_atoms):
    atoms = load_atoms()
    class StressDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            self.results["stress"] = np.zeros(6, dtype=float)

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            target.positions += 0.01

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            StressDummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=2,
            stress_isif=0,
        )
    finally:
        monkeypatch.undo()

    outcar = (tmp_path / "OUTCAR").read_text()
    assert "FORCE on cell =-STRESS in cart. coord." not in outcar


def test_relaxation_vasprun_includes_kpoints_and_timing(tmp_path: Path, load_atoms):
    atoms = load_atoms()

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            target.positions += 0.01

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            DummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=2,
            stress_isif=2,
        )
    finally:
        monkeypatch.undo()

    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    assert root.find("kpoints") is not None
    assert root.find("./structure[@name='primitive_cell']") is not None
    assert root.find("./varray[@name='primitive_index']") is not None
    first_calc = root.find("calculation")
    assert first_calc is not None
    assert first_calc.find("./time[@name='totalsc']") is not None


def test_relaxation_isif3_moves_ions_and_cell(tmp_path: Path, load_atoms, arrays_close):
    atoms = load_atoms()
    initial_positions = atoms.get_positions().copy()
    initial_cell = atoms.cell.array.copy()

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            target.positions += 0.05
            new_cell = target.cell.array * 1.01
            target.set_cell(new_cell, scale_atoms=False)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(atoms, DummyCalculator(), steps=2, fmax=0.01, isif=3)
    finally:
        monkeypatch.undo()

    assert not arrays_close(atoms.get_positions(), initial_positions)
    assert not arrays_close(atoms.cell.array, initial_cell)


def test_relaxation_isif3_applies_pstress(tmp_path: Path, load_atoms):
    atoms = load_atoms()

    class DummyUnitCellFilter:
        def __init__(self, atoms, scalar_pressure=0.0):
            self.atoms = atoms
            self.scalar_pressure = scalar_pressure

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            target.positions += 0.01

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    captured: dict[str, float] = {}

    def capture_ucf(atoms, scalar_pressure=0.0):
        captured["scalar_pressure"] = scalar_pressure
        return DummyUnitCellFilter(atoms, scalar_pressure=scalar_pressure)

    monkeypatch.setattr(vpmdk, "UnitCellFilter", capture_ucf)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            DummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=3,
            pstress=12.5,
        )
    finally:
        monkeypatch.undo()

    expected = 12.5 * vpmdk.KBAR_TO_EV_PER_A3
    assert "scalar_pressure" in captured
    assert pytest.approx(captured["scalar_pressure"], rel=1e-12) == expected


def test_relaxation_isif4_uses_constant_volume_filter(tmp_path: Path, load_atoms):
    atoms = load_atoms()
    initial_constraints = list(atoms.constraints)
    captured_kwargs: dict[str, object] = {}
    seen_constraints: list[list[object]] = []

    class DummyUnitCellFilter:
        def __init__(self, atoms, **kwargs):
            self.atoms = atoms
            captured_kwargs.update(kwargs)

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            seen_constraints.append(list(target.constraints))
            target.positions += 0.01

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "UnitCellFilter", lambda atoms, **kw: DummyUnitCellFilter(atoms, **kw))
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            DummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=4,
            pstress=7.5,
        )
    finally:
        monkeypatch.undo()

    assert captured_kwargs.get("constant_volume") is True
    expected_pressure = 7.5 * vpmdk.KBAR_TO_EV_PER_A3
    assert pytest.approx(captured_kwargs.get("scalar_pressure", 0.0), rel=1e-12) == expected_pressure
    assert captured_kwargs.get("hydrostatic_strain") in (None, False)
    assert seen_constraints and seen_constraints[0] == initial_constraints
    assert atoms.constraints == initial_constraints


def test_relaxation_isif5_freezes_ions_constant_volume(tmp_path: Path, load_atoms):
    atoms = load_atoms()
    initial_constraints = list(atoms.constraints)
    captured_kwargs: dict[str, object] = {}
    seen_constraints: list[list[object]] = []

    class DummyUnitCellFilter:
        def __init__(self, atoms, **kwargs):
            self.atoms = atoms
            captured_kwargs.update(kwargs)

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            seen_constraints.append(list(target.constraints))
            new_cell = target.cell.array * 1.01
            target.set_cell(new_cell, scale_atoms=True)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "UnitCellFilter", lambda atoms, **kw: DummyUnitCellFilter(atoms, **kw))
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            DummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=5,
        )
    finally:
        monkeypatch.undo()

    assert captured_kwargs.get("constant_volume") is True
    assert pytest.approx(captured_kwargs.get("scalar_pressure", 0.0), rel=1e-12) == 0.0
    assert captured_kwargs.get("hydrostatic_strain") in (None, False)
    assert seen_constraints
    assert any(isinstance(constraint, vpmdk.FixAtoms) for constraint in seen_constraints[0])
    assert atoms.constraints == initial_constraints


def test_relaxation_isif6_scales_cell_preserving_fractional_positions(
    tmp_path: Path, load_atoms, arrays_close
):
    atoms = load_atoms()
    initial_positions = atoms.get_positions().copy()
    initial_scaled_positions = atoms.get_scaled_positions().copy()
    initial_cell = atoms.cell.array.copy()

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            new_cell = target.cell.array * 1.02
            target.set_cell(new_cell, scale_atoms=True)

    class DummyStrainFilter:
        def __init__(self, atoms):
            self.atoms = atoms

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "StrainFilter", DummyStrainFilter)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(atoms, DummyCalculator(), steps=2, fmax=0.01, isif=6)
    finally:
        monkeypatch.undo()

    assert not arrays_close(atoms.get_positions(), initial_positions)
    assert arrays_close(atoms.get_scaled_positions(), initial_scaled_positions)
    assert not arrays_close(atoms.cell.array, initial_cell)


def test_relaxation_isif7_freezes_ions_with_isotropic_cell_changes(
    tmp_path: Path, load_atoms
):
    atoms = load_atoms()
    initial_constraints = list(atoms.constraints)
    captured_kwargs: dict[str, object] = {}
    seen_constraints: list[list[object]] = []

    class DummyUnitCellFilter:
        def __init__(self, atoms, **kwargs):
            self.atoms = atoms
            captured_kwargs.update(kwargs)

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            seen_constraints.append(list(target.constraints))
            new_cell = target.cell.array * 1.02
            target.set_cell(new_cell, scale_atoms=True)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "UnitCellFilter", lambda atoms, **kw: DummyUnitCellFilter(atoms, **kw))
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            DummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=7,
        )
    finally:
        monkeypatch.undo()

    assert captured_kwargs.get("mask") == [1, 1, 1, 0, 0, 0]
    assert captured_kwargs.get("hydrostatic_strain") is True
    assert seen_constraints
    assert any(isinstance(constraint, vpmdk.FixAtoms) for constraint in seen_constraints[0])
    assert atoms.constraints == initial_constraints


def test_relaxation_isif8_relaxes_ions_with_isotropic_volume(
    tmp_path: Path, load_atoms, arrays_close
):
    atoms = load_atoms()
    initial_positions = atoms.get_positions().copy()
    initial_constraints = list(atoms.constraints)
    captured_kwargs: dict[str, object] = {}
    seen_constraints: list[list[object]] = []

    class DummyUnitCellFilter:
        def __init__(self, atoms, **kwargs):
            self.atoms = atoms
            captured_kwargs.update(kwargs)

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            seen_constraints.append(list(target.constraints))
            new_cell = target.cell.array * 1.02
            target.set_cell(new_cell, scale_atoms=True)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "UnitCellFilter", lambda atoms, **kw: DummyUnitCellFilter(atoms, **kw))
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(
            atoms,
            DummyCalculator(),
            steps=1,
            fmax=0.01,
            isif=8,
        )
    finally:
        monkeypatch.undo()

    assert captured_kwargs.get("mask") == [1, 1, 1, 0, 0, 0]
    assert captured_kwargs.get("hydrostatic_strain") is True
    assert seen_constraints and seen_constraints[0] == initial_constraints
    assert atoms.constraints == initial_constraints
    assert not arrays_close(atoms.get_positions(), initial_positions)


def test_relaxation_stops_when_energy_change_below_tolerance(
    tmp_path: Path, load_atoms
):
    atoms = load_atoms()

    class DummyBFGS:
        last_instance = None

        def __init__(self, obj, logfile=None):
            self.obj = obj
            self.logfile = logfile
            self.observers: list[object] = []
            self.nsteps = 0
            self.fmax = None
            DummyBFGS.last_instance = self

        def attach(self, func):
            self.observers.append(func)

        def irun(self, steps):
            yield False
            while self.nsteps < steps:
                self.nsteps += 1
                target = getattr(self.obj, "atoms", self.obj)
                target.positions += 0.01
                for func in list(self.observers):
                    func()
                yield False

        def run(self, *args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("Energy convergence should use irun")

    energy_values = [1.0, 0.8, 0.7, 0.69, 0.68]
    index = {"value": 0}

    def fake_energy():
        idx = index["value"]
        if idx >= len(energy_values):
            return energy_values[-1]
        value = energy_values[idx]
        index["value"] += 1
        return value

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    monkeypatch.setattr(atoms, "get_potential_energy", fake_energy)
    try:
        vpmdk.run_relaxation(
            atoms,
            DummyCalculator(),
            steps=10,
            fmax=-0.01,
            energy_tolerance=0.015,
        )
    finally:
        instance = DummyBFGS.last_instance
        monkeypatch.undo()

    assert instance is not None
    assert instance.nsteps == 3
