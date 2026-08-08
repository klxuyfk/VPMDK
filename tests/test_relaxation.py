from __future__ import annotations

import re
from pathlib import Path

import pytest
import numpy as np
import ase.units
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
    assert "Voluntary context switches" in outcar
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


def test_estimate_neb_chain_approximation_uses_minimum_image_tangent():
    # A migrating atom crossing a periodic face: POSCAR fractional coordinates
    # wrap into [0, 1), so the raw next-prev difference spans nearly the whole
    # cell while the physical image separation is a fraction of an angstrom.
    # ASE's NEB engine takes the minimum-image tangent (Spring._find_mic ->
    # ase.geometry.find_mic); the OUTCAR TANGENT block must describe the same
    # band the optimizer relaxed.
    a = 5.43
    cell = np.eye(3) * a
    frac_prev = np.array([[0.20, 0.0, 0.95], [0.5, 0.5, 0.5]])
    frac_cur = np.array([[0.26, 0.0, 0.99], [0.5, 0.5, 0.5]])
    frac_next = np.array([[0.32, 0.0, 0.03], [0.5, 0.5, 0.5]])
    prev, cur, nxt = (f @ cell for f in (frac_prev, frac_cur, frac_next))
    forces = np.array([[0.1, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=float)

    approx = vpmdk._estimate_neb_chain_approximation(
        positions=cur,
        forces=forces,
        prev_positions=prev,
        next_positions=nxt,
        cell=cell,
        pbc=np.array([True, True, True]),
    )

    assert approx is not None
    # Minimum-image displacement is (0.6516, 0, 0.4344) A -> unit tangent
    # (0.8321, 0, 0.5547). The raw subtraction gave (0.1293, 0, -0.9916):
    # 116 degrees off with the z sign inverted.
    assert np.allclose(
        approx.tangent_vectors[0], [0.832050, 0.0, 0.554700], atol=1e-6
    )
    assert approx.tangent_vectors[0][2] > 0.0


def test_estimate_neb_chain_approximation_without_cell_keeps_raw_difference():
    # Callers with bare position arrays (no cell/pbc) must keep the raw
    # difference: there is nothing to wrap against.
    prev = np.array([[0.0, 0.0, 5.1585], [2.715, 2.715, 2.715]])
    cur = np.array([[0.3258, 0.0, 5.3757], [2.715, 2.715, 2.715]])
    nxt = np.array([[0.6516, 0.0, 0.1629], [2.715, 2.715, 2.715]])
    forces = np.zeros_like(cur)
    forces[0] = [0.1, 0.0, 0.5]

    approx = vpmdk._estimate_neb_chain_approximation(
        positions=cur, forces=forces, prev_positions=prev, next_positions=nxt
    )

    assert approx is not None
    raw = (nxt - prev).ravel()
    expected = raw / np.linalg.norm(raw)
    assert np.allclose(approx.tangent_vectors.ravel(), expected, atol=1e-12)


def test_estimate_neb_chain_approximation_zero_cell_keeps_raw_difference():
    # A zero cell (ASE's default for molecules) must not be treated as
    # periodic even if pbc flags are accidentally truthy.
    prev = np.array([[0.0, 0.0, 0.0]])
    cur = np.array([[1.0, 0.0, 0.0]])
    nxt = np.array([[2.0, 0.0, 0.0]])
    forces = np.array([[1.0, 0.0, 0.0]])

    approx = vpmdk._estimate_neb_chain_approximation(
        positions=cur,
        forces=forces,
        prev_positions=prev,
        next_positions=nxt,
        cell=np.zeros((3, 3)),
        pbc=np.array([True, True, True]),
    )

    assert approx is not None
    assert np.allclose(approx.tangent_vectors, [[1.0, 0.0, 0.0]], atol=1e-12)


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
    outcar = (tmp_path / "OUTCAR").read_text()
    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    electronic = root.find("./parameters/separator[@name='electronic']")
    assert "DAV:" not in oszicar
    assert "N       E" not in oszicar
    assert "NELM   =" not in outcar
    assert "Iteration      1(   1)" in outcar
    assert "Voluntary context switches" in outcar
    assert electronic is not None
    assert root.find("./parameters/separator[@name='electronic convergence']") is None
    assert root.find("./incar/i[@name='NELM']") is None
    assert electronic.find("./i[@name='NELM']") is not None
    assert electronic.find("./i[@name='NELMIN']") is None
    assert electronic.find("./i[@name='EDIFF']") is None
    assert electronic.find("./i[@name='NBANDS']") is None
    assert root.find("./incar/i[@name='NELMIN']") is None
    assert root.find(".//scstep/energy") is not None
    assert root.find("./calculation/time[@name='totalsc']") is None


def test_relaxation_oszicar_pseudo_scf_is_written_when_enabled(tmp_path: Path, load_atoms):
    atoms = load_atoms()
    (tmp_path / "INCAR").write_text("NELM = 37\nNELMIN = 4\nNELMDL = -3\nEDIFF = 5E-07\n")

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
    outcar = (tmp_path / "OUTCAR").read_text()
    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    electronic = root.find("./parameters/separator[@name='electronic']")
    assert "DAV:" in oszicar
    assert "N       E" in oszicar
    assert "NELM   =     37;" in outcar
    assert "Iteration      1(   1)" in outcar
    assert electronic is not None
    assert root.find("./parameters/separator[@name='electronic convergence']") is None
    assert electronic.find("./i[@name='NBANDS']") is not None
    assert electronic.find("./i[@name='NELM']").text.strip() == "37"
    assert electronic.find("./i[@name='NELMIN']").text.strip() == "4"
    assert electronic.find("./i[@name='NELMDL']").text.strip() == "-3"
    assert electronic.find("./i[@name='EDIFF']").text.strip() == "5.00000000E-07"
    assert root.find("./incar/i[@name='NELM']").text.strip() == "37"
    assert root.find("./incar/i[@name='NELMIN']").text.strip() == "4"
    assert root.find("./incar/i[@name='NELMDL']").text.strip() == "-3"
    assert root.find("./incar/i[@name='EDIFF']").text.strip() == "5.00000000E-07"
    assert root.find(".//scstep") is not None
    assert root.find(".//i[@name='NELM']") is not None
    assert root.find("./calculation/time[@name='totalsc']") is not None


def test_single_point_oszicar_pseudo_scf_reads_local_incar_when_enabled(
    tmp_path: Path, load_atoms
):
    atoms = load_atoms()
    (tmp_path / "INCAR").write_text("NELM = 41\nNELMIN = 3\nNELMDL = -2\nEDIFF = 1E-06\n")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        vpmdk.run_single_point(
            atoms,
            DummyCalculator(),
            oszicar_pseudo_scf=True,
        )
    finally:
        monkeypatch.undo()

    outcar = (tmp_path / "OUTCAR").read_text()
    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    electronic = root.find("./parameters/separator[@name='electronic']")
    assert "NELM   =     41;" in outcar
    assert electronic is not None
    assert electronic.find("./i[@name='NELM']").text.strip() == "41"
    assert electronic.find("./i[@name='NELMIN']").text.strip() == "3"
    assert electronic.find("./i[@name='NELMDL']").text.strip() == "-2"
    assert root.find("./incar/i[@name='EDIFF']").text.strip() == "1.00000000E-06"


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


def test_relaxation_vasprun_includes_kpoints_and_omits_pseudo_scf_timing_by_default(
    tmp_path: Path, load_atoms
):
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
    assert first_calc.find("./time[@name='totalsc']") is None
    # The pseudo-SCF TIMING stays off by default; the scstep energy block itself
    # is what every VASP XML reader requires (see the default-configuration test
    # above), so it is written for every ionic step.
    scstep_energy = first_calc.find("./scstep/energy")
    assert scstep_energy is not None
    assert scstep_energy.find("./i[@name='e_0_energy']") is not None


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
    assert "scalar_pressure" not in captured_kwargs
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


def test_stress_output_uses_vasps_sign_and_units(tmp_path: Path, load_atoms):
    atoms = load_atoms()
    sigma_xx = -0.05  # eV/A^3, ASE sign: negative = compression
    voigt = np.array([sigma_xx, sigma_xx, sigma_xx, 0.0, 0.0, 0.0], dtype=float)

    class StressDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            self.results["stress"] = voigt

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_single_point(atoms, StressDummyCalculator(), isif=3)
    finally:
        monkeypatch.undo()

    expected_kbar = -sigma_xx / vpmdk.KBAR_TO_EV_PER_A3
    expected_total = -sigma_xx * atoms.get_volume()
    assert expected_kbar > 0.0  # compression -> positive in VASP's convention

    outcar = (tmp_path / "OUTCAR").read_text()
    kbar_line = next(line for line in outcar.splitlines() if line.startswith("  in kB"))
    assert [float(value) for value in kbar_line.split()[2:5]] == pytest.approx(
        [expected_kbar] * 3, rel=1e-4
    )
    pressure_line = next(
        line for line in outcar.splitlines() if "external pressure" in line
    )
    assert float(pressure_line.split("=")[1].split()[0]) == pytest.approx(
        expected_kbar, rel=1e-4
    )
    total_line = next(line for line in outcar.splitlines() if line.startswith("  Total"))
    assert [float(value) for value in total_line.split()[1:4]] == pytest.approx(
        [expected_total] * 3, rel=1e-4
    )

    tree = ET.parse(tmp_path / "vasprun.xml")
    rows = tree.find(".//varray[@name='stress']")
    assert rows is not None
    written = np.array([[float(v) for v in row.text.split()] for row in rows])
    assert written[0, 0] == pytest.approx(expected_kbar, rel=1e-4)
    # ASE's documented conversion must recover the calculator's own stress.
    assert (written * -0.1 * ase.units.GPa)[0, 0] == pytest.approx(sigma_xx, rel=1e-6)
    # VPMDK reads its own NEB image output back, so the inverse must close.
    _, _, parsed = vpmdk._read_last_vasprun_step(str(tmp_path / "vasprun.xml"))
    assert parsed[0, 0] == pytest.approx(sigma_xx, rel=1e-6)


def test_energy_convergence_does_not_leak_ases_default_force_limit(load_atoms):
    from ase.calculators.emt import EMT

    seen: dict[str, float] = {}

    class ProbeBFGS(vpmdk.BFGS):
        def irun(self, *args, **kwargs):
            generator = super().irun(*args, **kwargs)
            seen["fmax"] = self.fmax
            return generator

    from ase.build import bulk

    # EMT has no Si parameters; Cu is the standard cheap, deterministic choice.
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.positions[0] += (0.3, 0.1, 0.0)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "BFGS", ProbeBFGS)
    try:
        result = vpmdk.relax(
            atoms,
            calculator=EMT(),
            steps=60,
            fmax=-1e-12,
            energy_tolerance=1e-12,
        )
    finally:
        monkeypatch.undo()

    assert seen["fmax"] == -1e-12, "ASE's 0.05 default overwrote the requested fmax"
    # A negative fmax can never force-converge, so only the energy criterion fires.
    delta = abs(result.steps[-1].potential_energy - result.steps[-2].potential_energy)
    assert delta <= 1e-12
    assert float(np.abs(result.forces).max()) < 1e-4


def test_isif6_applies_pstress_that_strainfilter_cannot_take(load_atoms):
    builder, freeze_required = vpmdk._make_relaxation_builder(6, None, 0.0)
    assert builder is vpmdk.StrainFilter
    assert freeze_required is False

    # An explicit PSTRESS=0 must keep the documented StrainFilter mapping so runs
    # that were already correct stay bit-for-bit unchanged.
    builder_zero, _ = vpmdk._make_relaxation_builder(6, 0.0, 0.0)
    assert builder_zero is vpmdk.StrainFilter

    pressure = 500.0 * vpmdk.KBAR_TO_EV_PER_A3
    builder_pressure, freeze_pressure = vpmdk._make_relaxation_builder(
        6, pressure, pressure
    )
    assert freeze_pressure is True, "ISIF=6 must not move ions"
    relax_object = builder_pressure(load_atoms())
    assert isinstance(relax_object, vpmdk.UnitCellFilter)
    assert relax_object.scalar_pressure == pytest.approx(pressure)


@pytest.mark.parametrize("isif,pstress", [(6, None), (6, 500.0), (5, 500.0), (7, 500.0)])
def test_cell_only_relaxations_report_physical_forces(
    tmp_path: Path, isif: int, pstress: float | None
):
    from ase.build import bulk
    from ase.calculators.emt import EMT

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.positions[0] += (0.05, 0.02, 0.0)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        vpmdk.run_relaxation(
            atoms,
            EMT(),
            steps=3,
            fmax=0.01,
            isif=isif,
            stress_isif=isif,
            pstress=pstress,
            ibrion=2,
        )
    finally:
        monkeypatch.undo()

    outcar = (tmp_path / "OUTCAR").read_text()
    force_lines = [line for line in outcar.splitlines() if "FORCES: max atom" in line]
    assert force_lines
    reported = float(force_lines[-1].split()[-2])
    assert reported > 1e-3, "cell-only relaxation reported zeroed forces"

    tree = ET.parse(tmp_path / "vasprun.xml")
    rows = tree.findall(".//varray[@name='forces']/v")
    assert rows
    values = np.array([[float(v) for v in row.text.split()] for row in rows])
    assert np.abs(values).max() > 1e-3


def test_internal_isif_freeze_is_dropped_but_user_constraints_are_kept(tmp_path: Path):
    # The recorder must remove ONLY VPMDK's own freeze: a user's selective dynamics
    # is a real constraint and must keep zeroing the forces it fixes.
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms

    from vpmdk_core.io import vasp_compat
    from vpmdk_core.runtime import relax as relax_module

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.positions[0] += (0.05, 0.02, 0.0)
    atoms.calc = EMT()
    user_constraint = FixAtoms(indices=[1])
    atoms.set_constraint(user_constraint)

    with relax_module._temporarily_freeze_atoms(atoms, True):
        with vasp_compat._without_internal_isif_freeze(atoms):
            forces = atoms.get_forces(apply_constraint=True)
        # The internal freeze is restored for the optimiser itself.
        assert len(atoms.constraints) == 2

    assert np.abs(forces[0]).max() > 1e-3, "internal freeze still zeroed the report"
    assert np.allclose(forces[1], 0.0), "user selective dynamics was dropped"
    assert list(atoms.constraints) == [user_constraint]


@pytest.mark.parametrize("isif", [4, 5])
def test_constant_volume_modes_ignore_pstress(tmp_path: Path, isif: int, capsys):
    from ase.build import bulk
    from ase.calculators.emt import EMT

    volumes = []
    for pstress in (0.0, 2000.0):
        atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
        atoms.rattle(0.10, seed=1)
        initial_volume = atoms.get_volume()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.chdir(tmp_path)
        try:
            vpmdk.run_relaxation(
                atoms,
                EMT(),
                steps=40,
                fmax=0.02,
                isif=isif,
                stress_isif=isif,
                pstress=pstress,
                ibrion=2,
            )
        finally:
            monkeypatch.undo()
        volumes.append(atoms.get_volume() / initial_volume)

    # The pressure must not change the trajectory at all.
    assert volumes[1] == pytest.approx(volumes[0], rel=1e-9)
    assert abs(volumes[0] - 1.0) < 0.02, "constant volume was not preserved"
    assert "PSTRESS is ignored" in capsys.readouterr().out


def test_pstress_output_reports_corrected_pressure_and_pullay_stress(
    tmp_path: Path, load_atoms
):
    atoms = load_atoms()
    pstress_kbar = 500.0
    sigma_xx = -pstress_kbar * vpmdk.KBAR_TO_EV_PER_A3  # exactly balances
    voigt = np.array([sigma_xx, sigma_xx, sigma_xx, 0.0, 0.0, 0.0], dtype=float)

    class StressDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            self.results["stress"] = voigt

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, callback, *args, **kwargs):
            self._callback = callback

        def run(self, *args, **kwargs):
            self._callback()
            return True

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
            isif=3,
            pstress=pstress_kbar,
        )
    finally:
        monkeypatch.undo()

    outcar = (tmp_path / "OUTCAR").read_text()
    pressure_line = next(
        line for line in outcar.splitlines() if "external pressure" in line
    )
    # Internal stress balances the applied PSTRESS: corrected pressure ~ 0,
    # Pullay field echoes the applied 500.
    assert float(pressure_line.split()[3]) == pytest.approx(0.0, abs=0.01)
    assert float(pressure_line.split()[-2]) == pytest.approx(500.0, abs=0.01)
    kbar_line = next(line for line in outcar.splitlines() if line.startswith("  in kB"))
    assert [float(value) for value in kbar_line.split()[2:5]] == pytest.approx(
        [0.0] * 3, abs=0.01
    )

    # The vasprun stress varray is the same corrected tensor.
    import xml.etree.ElementTree as ET

    tree = ET.parse(tmp_path / "vasprun.xml")
    stress_rows = tree.findall(".//varray[@name='stress']/v")
    assert stress_rows
    diagonal = [float(row.text.split()[index]) for index, row in enumerate(stress_rows[-3:])]
    assert diagonal == pytest.approx([0.0] * 3, abs=0.01)


def test_vasprun_echoes_the_requested_nsw_not_the_step_count(
    tmp_path: Path, load_atoms
):
    import xml.etree.ElementTree as ET

    atoms = load_atoms()

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, callback, *args, **kwargs):
            self._callback = callback

        def run(self, *args, **kwargs):
            for _ in range(3):  # converges after 3 recorded steps
                self._callback()
            return True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(atoms, DummyCalculator(), steps=8, fmax=0.01, isif=2)
    finally:
        monkeypatch.undo()

    tree = ET.parse(tmp_path / "vasprun.xml")
    echoed = [
        node.text
        for node in tree.iter("i")
        if node.get("name") == "NSW"
    ]
    assert echoed == ["8", "8"], echoed
    # The recorded step count is untouched (pinned legacy behavior).
    assert len(tree.findall(".//calculation")) == 3


def test_pstress_vasprun_energy_carries_pv_term_and_declares_pstress(
    tmp_path: Path, load_atoms
):
    import xml.etree.ElementTree as ET

    import ase.io

    atoms = load_atoms()
    pstress_kbar = 500.0
    energy_ev = -1.25
    sigma = -pstress_kbar * vpmdk.KBAR_TO_EV_PER_A3
    voigt = np.array([sigma] * 3 + [0.0] * 3, dtype=float)

    class StressDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            self.results["energy"] = energy_ev
            self.results["stress"] = voigt

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj

        def attach(self, callback, *args, **kwargs):
            self._callback = callback

        def run(self, *args, **kwargs):
            self._callback()
            return True

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
            isif=3,
            pstress=pstress_kbar,
        )
    finally:
        monkeypatch.undo()

    pv_term = pstress_kbar * vpmdk.KBAR_TO_EV_PER_A3 * atoms.get_volume()
    tree = ET.parse(tmp_path / "vasprun.xml")
    calc_e = float(tree.find(".//calculation/energy/i[@name='e_fr_energy']").text)
    scstep_e = float(
        tree.find(".//calculation/scstep/energy/i[@name='e_fr_energy']").text
    )
    assert scstep_e == pytest.approx(energy_ev, abs=1e-8)
    assert calc_e == pytest.approx(energy_ev + pv_term, abs=1e-6)
    # PSTRESS is declared in BOTH blocks (ASE reads parameters, lowercased).
    assert tree.find(".//incar/i[@name='PSTRESS']") is not None
    assert tree.find(".//parameters//i[@name='PSTRESS']") is not None
    # And the documented ASE consumer round-trips back to the plain energy.
    restored = ase.io.read(str(tmp_path / "vasprun.xml"))
    assert restored.get_potential_energy() == pytest.approx(energy_ev, abs=1e-5)
    # OSZICAR keeps the plain E, like real VASP.
    oszicar = (tmp_path / "OSZICAR").read_text()
    assert "E0=  -.12500000E+01" in oszicar


def test_neb_vasprun_readback_undoes_pstress_transformations(tmp_path: Path):
    import vpmdk_core.runtime.neb as neb_module

    pstress_kbar = 100.0
    volume = 5.0 ** 3
    energy_ev = -10.0
    pv = pstress_kbar * vpmdk.KBAR_TO_EV_PER_A3 * volume
    raw_sigma = -0.02  # ASE-signed eV/A^3
    file_kbar = -raw_sigma / vpmdk.KBAR_TO_EV_PER_A3 - pstress_kbar

    path = tmp_path / "vasprun.xml"
    path.write_text(
        "<modeling>"
        '<incar><i name="PSTRESS" type="float">100.00000000</i></incar>'
        "<calculation>"
        '<structure><crystal><varray name="basis">'
        "<v>5.0 0.0 0.0</v><v>0.0 5.0 0.0</v><v>0.0 0.0 5.0</v>"
        "</varray></crystal></structure>"
        f'<energy><i name="e_fr_energy">{energy_ev + pv:.8f}</i></energy>'
        '<varray name="forces"><v>0.1 0.0 0.0</v></varray>'
        '<varray name="stress">'
        f"<v>{file_kbar:.8f} 0.0 0.0</v>"
        f"<v>0.0 {file_kbar:.8f} 0.0</v>"
        f"<v>0.0 0.0 {file_kbar:.8f}</v>"
        "</varray>"
        "</calculation></modeling>"
    )

    energy, forces, stress = neb_module._read_last_vasprun_step(str(path))
    assert energy == pytest.approx(energy_ev, abs=1e-6)
    assert np.allclose(stress, np.eye(3) * raw_sigma, atol=1e-10)

    # Without PSTRESS the reader keeps its previous behavior.
    plain = tmp_path / "plain.xml"
    plain.write_text(
        "<modeling><calculation>"
        '<structure><crystal><varray name="basis">'
        "<v>5.0 0.0 0.0</v><v>0.0 5.0 0.0</v><v>0.0 0.0 5.0</v>"
        "</varray></crystal></structure>"
        f'<energy><i name="e_fr_energy">{energy_ev:.8f}</i></energy>'
        '<varray name="stress">'
        f"<v>{-raw_sigma / vpmdk.KBAR_TO_EV_PER_A3:.8f} 0.0 0.0</v>"
        f"<v>0.0 {-raw_sigma / vpmdk.KBAR_TO_EV_PER_A3:.8f} 0.0</v>"
        f"<v>0.0 0.0 {-raw_sigma / vpmdk.KBAR_TO_EV_PER_A3:.8f}</v>"
        "</varray>"
        "</calculation></modeling>"
    )
    energy2, _, stress2 = neb_module._read_last_vasprun_step(str(plain))
    assert energy2 == pytest.approx(energy_ev, abs=1e-8)
    assert np.allclose(stress2, np.eye(3) * raw_sigma, atol=1e-10)


def test_pstress_correction_applies_to_single_point_and_md_too(
    tmp_path: Path, load_atoms
):
    pstress_kbar = 500.0
    sigma_xx = -pstress_kbar * vpmdk.KBAR_TO_EV_PER_A3
    voigt = np.array([sigma_xx, sigma_xx, sigma_xx, 0.0, 0.0, 0.0], dtype=float)

    class StressDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            self.results["stress"] = voigt

    def pressure_line(workdir: Path) -> list[str]:
        outcar = (workdir / "OUTCAR").read_text()
        return next(
            line for line in outcar.splitlines() if "external pressure" in line
        ).split()

    # Single point.
    single_dir = tmp_path / "single"
    single_dir.mkdir()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(single_dir)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_single_point(
            load_atoms(), StressDummyCalculator(), isif=3, pstress=pstress_kbar
        )
    finally:
        monkeypatch.undo()
    line = pressure_line(single_dir)
    assert float(line[3]) == pytest.approx(0.0, abs=0.01)
    assert float(line[-2]) == pytest.approx(500.0, abs=0.01)

    # MD (one velocity-Verlet step).
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(md_dir)
    monkeypatch.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_md(
            load_atoms(),
            StressDummyCalculator(),
            1,
            300.0,
            1.0,
            mdalgo=0,
            isif=3,
            pstress=pstress_kbar,
        )
    finally:
        monkeypatch.undo()
    line = pressure_line(md_dir)
    assert float(line[3]) == pytest.approx(0.0, abs=0.01)
    assert float(line[-2]) == pytest.approx(500.0, abs=0.01)


def test_write_vasp_structure_rejects_directory_target_as_isadirectoryerror(tmp_path: Path, load_atoms):
    target = tmp_path / "CONTCAR"
    target.mkdir()

    with pytest.raises(IsADirectoryError):
        vpmdk._write_vasp_structure(str(target), load_atoms())

    # The directory is left untouched.
    assert target.is_dir()


def test_default_vasprun_round_trips_through_ase_and_reports_the_run_energy(
    tmp_path: Path, load_atoms
):
    ase_io = pytest.importorskip("ase.io")
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
        vpmdk.run_relaxation(atoms, DummyCalculator(), steps=1, fmax=0.01, isif=2)
    finally:
        monkeypatch.undo()

    images = ase_io.read(str(tmp_path / "vasprun.xml"), index=":")

    assert images
    assert images[-1].get_potential_energy() == pytest.approx(0.5)
    assert len(images[-1]) == len(atoms)


def test_contcar_keeps_per_axis_selective_dynamics(tmp_path: Path):
    from ase import Atoms
    from ase.constraints import FixAtoms, FixCartesian

    atoms = Atoms(
        "Cu4",
        scaled_positions=[
            (0.02, 0.01, 0.03),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ],
        cell=[3.6, 3.6, 3.6],
        pbc=True,
    )
    atoms.set_constraint(
        [
            FixCartesian(0, mask=(False, False, True)),  # T T F
            FixAtoms(indices=[3]),  # F F F
        ]
    )

    path = tmp_path / "CONTCAR"
    vpmdk._write_vasp_structure(str(path), atoms)

    text = path.read_text()
    assert "Selective dynamics" in text
    rows = [line.split() for line in text.splitlines()[9:13]]
    flags = [row[3:6] for row in rows]
    assert flags == [
        ["T", "T", "F"],
        ["T", "T", "T"],
        ["T", "T", "T"],
        ["F", "F", "F"],
    ], text

    # And the file reads back with the same constraints, which is what makes the
    # `cp CONTCAR POSCAR` continuation keep the frozen axis frozen.
    import ase.io

    restored = ase.io.read(str(path))
    restored_flags = ase.io.vasp._handle_ase_constraints(restored)
    assert restored_flags.tolist() == [
        [False, False, True],
        [False, False, False],
        [False, False, False],
        [True, True, True],
    ]


def test_write_vasp_structure_leaves_other_constraints_and_atoms_alone(tmp_path: Path):
    from ase import Atoms
    from ase.constraints import FixAtoms, FixCartesian

    atoms = Atoms("Cu2", scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
                  cell=[3.6, 3.6, 3.6], pbc=True)
    original = [FixCartesian(0, mask=(True, False, False)), FixAtoms(indices=[1])]
    atoms.set_constraint(original)

    vpmdk._write_vasp_structure(str(tmp_path / "CONTCAR"), atoms)

    # The translation happens on a throwaway copy: the running atoms keep the
    # constraint objects that were computing the physics.
    assert [type(c).__name__ for c in atoms.constraints] == ["FixCartesian", "FixAtoms"]
    assert atoms.constraints[0] is original[0]

    # Atoms without any FixCartesian are handed to the writer untouched.
    plain = Atoms("Cu1", scaled_positions=[(0.0, 0.0, 0.0)], cell=[3.6, 3.6, 3.6], pbc=True)
    plain.set_constraint(FixAtoms(indices=[0]))
    assert vpmdk._atoms_with_writable_selective_dynamics(plain) is plain
