from __future__ import annotations

from pathlib import Path
import sys
import types

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised when numpy missing
    np = None
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

if "pymatgen" not in sys.modules:
    pymatgen_module = types.ModuleType("pymatgen")
    sys.modules["pymatgen"] = pymatgen_module
    io_module = types.ModuleType("pymatgen.io")
    sys.modules["pymatgen.io"] = io_module
    vasp_module = types.ModuleType("pymatgen.io.vasp")
    sys.modules["pymatgen.io.vasp"] = vasp_module
    ase_module = types.ModuleType("pymatgen.io.ase")
    sys.modules["pymatgen.io.ase"] = ase_module

    def _coerce_array(values, *, scale: float | None = None):
        if np is not None:
            arr = np.array(values, dtype=float)
            if scale is not None:
                arr = arr * scale
            return arr

        def _convert(val):
            if isinstance(val, (list, tuple)):
                return [_convert(item) for item in val]
            return float(val)

        def _apply_scale(val):
            if isinstance(val, list):
                return [_apply_scale(item) for item in val]
            return val * scale if scale is not None else val

        converted = _convert(values)
        return _apply_scale(converted)

    class _Structure:
        def __init__(self, lattice, frac_coords, species):
            self.lattice = _coerce_array(lattice)
            self.frac_coords = _coerce_array(frac_coords)
            self.species = list(species)

    class Poscar:
        def __init__(self, structure, site_symbols):
            self._structure = structure
            self._site_symbols = list(site_symbols)

        @property
        def structure(self):
            return self._structure

        @property
        def site_symbols(self):
            return self._site_symbols

        @site_symbols.setter
        def site_symbols(self, symbols):
            self._site_symbols = list(symbols)

        @classmethod
        def from_file(cls, path):
            with open(path) as f:
                raw_lines = [line.strip() for line in f if line.strip()]
            scale = float(raw_lines[1])
            lattice = [list(map(float, raw_lines[i].split())) for i in range(2, 5)]
            lattice = _coerce_array(lattice, scale=scale)
            species_names = raw_lines[5].split()
            counts = list(map(int, raw_lines[6].split()))
            coord_start = 8
            frac_coords = []
            species = []
            for name, count in zip(species_names, counts):
                for _ in range(count):
                    values = list(map(float, raw_lines[coord_start].split()[:3]))
                    coord_start += 1
                    frac_coords.append(values)
                    species.append(name)
            structure = _Structure(lattice, frac_coords, species)
            poscar = cls(structure, species_names)
            return poscar

        @classmethod
        def from_str(cls, content):
            tmp = Path("poscar.tmp")
            tmp.write_text(content)
            try:
                return cls.from_file(tmp)
            finally:
                tmp.unlink()

    class Incar(dict):
        @classmethod
        def from_file(cls, path):
            data = cls()
            with open(path) as f:
                for line in f:
                    for comment in ("#", "!"):
                        if comment in line:
                            line = line.split(comment, 1)[0]
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    data[key.strip().upper()] = value.strip().split()[0]
            return data

    class Potcar:
        def __init__(self, symbols):
            self.symbols = symbols

        @classmethod
        def from_file(cls, path):
            with open(path) as f:
                lines = [line.strip() for line in f if line.strip()]
            symbols = [line for line in lines if line.isalpha()]
            return cls(symbols)

    class AseAtomsAdaptor:
        @staticmethod
        def get_atoms(structure):
            return Atoms(
                symbols=structure.species,
                cell=structure.lattice,
                scaled_positions=structure.frac_coords,
                pbc=True,
            )

    vasp_module.Poscar = Poscar
    vasp_module.Incar = Incar
    vasp_module.Potcar = Potcar
    ase_module.AseAtomsAdaptor = AseAtomsAdaptor

from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vpmdk  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent


class DummyCalculator(Calculator):
    """Lightweight calculator returning constant energy."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self):
        super().__init__()
        self.called = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.called += 1
        if atoms is not None:
            forces = atoms.get_positions() * 0.0
        else:
            forces = []
        self.results = {
            "energy": 0.5,
            "forces": forces,
            "stress": [0.0] * 6,
        }


def test_incar_parsing_handles_case_whitespace_and_comments(tmp_path: Path):
    incar_content = """
    ! leading comment
    nsw = 5   ! ionic steps
      IBrIoN = 2 # relaxation mode
    """
    path = tmp_path / "INCAR"
    path.write_text(incar_content)

    incar = Incar.from_file(path)

    assert "NSW" in incar
    assert str(incar.get("NSW")) == "5"
    assert str(incar.get("IBRION")) == "2"


def test_bcar_parsing_handles_case_whitespace_and_comments(tmp_path: Path):
    bcar_content = """
    # initial comment
      nnp = mace   # inline comment
    Model = /path/to/model.nn  ! trailing comment
    WRITE_energy_csv = On
    """
    path = tmp_path / "BCAR"
    path.write_text(bcar_content)

    tags = vpmdk.parse_key_value_file(str(path))

    assert tags["NNP"] == "mace"
    assert tags["MODEL"] == "/path/to/model.nn"
    assert tags["WRITE_ENERGY_CSV"] == "On"


def prepare_inputs(
    tmp_path: Path,
    *,
    potential: str = "CHGNET",
    incar_overrides: dict[str, str] | None = None,
    extra_bcar: dict[str, str] | None = None,
) -> None:
    """Copy canonical POSCAR/INCAR/BCAR into ``tmp_path`` with overrides."""

    (tmp_path / "POSCAR").write_text((DATA_DIR / "POSCAR").read_text())

    incar_lines = (DATA_DIR / "INCAR").read_text().splitlines()
    if incar_overrides:
        for key, value in incar_overrides.items():
            incar_lines.append(f"{key} = {value}")
    (tmp_path / "INCAR").write_text("\n".join(incar_lines) + "\n")

    bcar_lines = (DATA_DIR / "BCAR").read_text().splitlines()
    replaced = False
    for idx, line in enumerate(bcar_lines):
        if line.startswith("NNP="):
            bcar_lines[idx] = f"NNP={potential.upper()}"
            replaced = True
    if not replaced:
        bcar_lines.append(f"NNP={potential.upper()}")
    if extra_bcar:
        for key, value in extra_bcar.items():
            bcar_lines.append(f"{key}={value}")
    (tmp_path / "BCAR").write_text("\n".join(bcar_lines) + "\n")


@pytest.mark.parametrize("potential", ["CHGNET", "MATGL", "MACE", "MATTERSIM"])
def test_single_point_energy_for_all_potentials(tmp_path: Path, potential: str):
    prepare_inputs(tmp_path, potential=potential, incar_overrides={"NSW": "0"})

    created: list[tuple[str, DummyCalculator]] = []

    def factory(name: str):
        calc = DummyCalculator()
        created.append((name, calc))
        return calc

    mp = pytest.MonkeyPatch()
    mp.setattr(vpmdk, "CHGNetCalculator", lambda *a, **k: factory("CHGNET"))
    mp.setattr(vpmdk, "M3GNetCalculator", lambda *a, **k: factory("MATGL"))
    mp.setattr(vpmdk, "MACECalculator", lambda *a, **k: factory("MACE"))
    mp.setattr(vpmdk, "MatterSimCalculator", lambda *a, **k: factory("MATTERSIM"))
    mp.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        mp.undo()

    assert created and created[-1][0] == potential
    assert created[-1][1].called == 1


def load_atoms():
    structure = Poscar.from_file(DATA_DIR / "POSCAR").structure
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.wrap()
    return atoms


def arrays_close(a, b, tol: float = 1e-8) -> bool:
    if np is not None:
        return float(((a - b) ** 2).sum()) <= tol

    def _flatten(seq):
        if isinstance(seq, (list, tuple)):
            for item in seq:
                yield from _flatten(item)
        else:
            yield float(seq)

    flat_a = list(_flatten(a))
    flat_b = list(_flatten(b))
    if len(flat_a) != len(flat_b):
        return False
    diff = sum((x - y) ** 2 for x, y in zip(flat_a, flat_b))
    return diff <= tol


def test_relaxation_isif2_moves_ions_without_changing_cell(tmp_path: Path):
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

    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path)
    mp.setattr(vpmdk, "BFGS", DummyBFGS)
    mp.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(atoms, DummyCalculator(), steps=2, fmax=0.01, isif=2)
    finally:
        mp.undo()

    assert not arrays_close(atoms.get_positions(), initial_positions)
    assert arrays_close(atoms.cell.array, initial_cell)


def test_relaxation_isif3_moves_ions_and_cell(tmp_path: Path):
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

    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path)
    mp.setattr(vpmdk, "BFGS", DummyBFGS)
    mp.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(atoms, DummyCalculator(), steps=2, fmax=0.01, isif=3)
    finally:
        mp.undo()

    assert not arrays_close(atoms.get_positions(), initial_positions)
    assert not arrays_close(atoms.cell.array, initial_cell)


def test_relaxation_isif6_scales_cell_preserving_fractional_positions(tmp_path: Path):
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

    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path)
    mp.setattr(vpmdk, "BFGS", DummyBFGS)
    mp.setattr(vpmdk, "StrainFilter", DummyStrainFilter)
    mp.setattr(vpmdk, "write", lambda *a, **k: None)
    try:
        vpmdk.run_relaxation(atoms, DummyCalculator(), steps=2, fmax=0.01, isif=6)
    finally:
        mp.undo()

    assert not arrays_close(atoms.get_positions(), initial_positions)
    assert arrays_close(atoms.get_scaled_positions(), initial_scaled_positions)
    assert not arrays_close(atoms.cell.array, initial_cell)


def test_run_md_executes_multiple_steps(tmp_path: Path):
    atoms = load_atoms()

    class DummyVerlet:
        def __init__(self, atoms, timestep, logfile=None):
            self.atoms = atoms
            self.timestep = timestep
            self.steps = []

        def run(self, n):
            self.steps.append(n)
            self.atoms.positions += 0.01

    written = []

    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path)
    mp.setattr(vpmdk, "VelocityVerlet", DummyVerlet)
    mp.setattr(vpmdk.velocitydistribution, "MaxwellBoltzmannDistribution", lambda *a, **k: None)
    mp.setattr(vpmdk, "write", lambda filename, atoms, direct=True, append=False: written.append((filename, append)))
    try:
        energy = vpmdk.run_md(atoms, DummyCalculator(), steps=3, temperature=300, timestep=1.0)
    finally:
        mp.undo()

    assert isinstance(energy, float)
    assert written.count(("XDATCAR", False)) == 1
    assert written.count(("XDATCAR", True)) == 2
    assert ("CONTCAR", False) in written


@pytest.mark.parametrize(
    "isif, expected, warning_fragment",
    [
        (0, 2, None),
        (1, 2, None),
        (2, 2, None),
        (3, 3, None),
        (4, 3, "ISIF=4"),
        (5, 3, "ISIF=5"),
        (6, 6, None),
        (7, 6, "ISIF=7"),
        (8, 2, "ISIF=8"),
    ],
)
def test_main_relaxation_respects_isif(
    tmp_path: Path, isif: int, expected: int, warning_fragment: str | None
):
    prepare_inputs(tmp_path, potential="CHGNET", incar_overrides={"NSW": "2", "ISIF": str(isif)})

    seen = {}

    def fake_run_relaxation(atoms, calculator, steps, fmax, write_energy_csv=False, isif=2):
        seen["isif"] = isif
        return 0.0

    mp = pytest.MonkeyPatch()
    mp.setattr(vpmdk, "get_calculator", lambda *_: DummyCalculator())
    mp.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    mp.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    messages: list[str] = []

    def fake_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        messages.append(sep.join(str(a) for a in args) + end)

    mp.setattr("builtins.print", fake_print)
    try:
        vpmdk.main()
    finally:
        mp.undo()

    assert seen["isif"] == expected
    if warning_fragment is None:
        assert not any("Warning: ISIF=" in message for message in messages)
    else:
        assert any(warning_fragment in message for message in messages)
