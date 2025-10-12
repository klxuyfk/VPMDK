from __future__ import annotations

from pathlib import Path

import pytest

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
