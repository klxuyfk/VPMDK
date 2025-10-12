from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Callable

import pytest

try:  # pragma: no cover - numpy optional in test env
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


def _install_pymatgen_stubs() -> None:
    """Install lightweight pymatgen stand-ins for the test suite."""

    if "pymatgen" in sys.modules:
        return

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

        converted = _convert(values)

        if scale is None:
            return converted

        def _apply_scale(val):
            if isinstance(val, list):
                return [_apply_scale(item) for item in val]
            return val * scale

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
        def structure(self):  # pragma: no cover - tiny accessor
            return self._structure

        @property
        def site_symbols(self):  # pragma: no cover - tiny accessor
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
            return cls(structure, species_names)

        @classmethod
        def from_str(cls, content):  # pragma: no cover - convenience
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
                    key = key.strip().upper()
                    cleaned = value.strip()
                    if key == "MAGMOM":
                        data[key] = cleaned
                    else:
                        parts = cleaned.split()
                        data[key] = parts[0] if parts else ""
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


_install_pymatgen_stubs()


def pytest_configure() -> None:
    """Ensure the project root is importable across all tests."""

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def arrays_close() -> Callable[[object, object, float], bool]:
    def _arrays_close(a, b, tol: float = 1e-8) -> bool:
        if np is not None:
            arr_a = np.array(a, dtype=float)
            arr_b = np.array(b, dtype=float)
            if arr_a.shape != arr_b.shape:
                return False
            return float(((arr_a - arr_b) ** 2).sum()) <= tol

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

    return _arrays_close


class DummyCalculator(Calculator):
    """Lightweight calculator returning constant energy."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self):
        super().__init__()
        self.called = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.called += 1
        forces = atoms.get_positions() * 0.0 if atoms is not None else []
        self.results = {
            "energy": 0.5,
            "forces": forces,
            "stress": [0.0] * 6,
        }


@pytest.fixture(scope="session")
def dummy_calculator_cls() -> type[DummyCalculator]:
    return DummyCalculator


@pytest.fixture
def load_atoms(data_dir: Path) -> Callable[[], Atoms]:
    from pymatgen.io.vasp import Poscar
    from pymatgen.io.ase import AseAtomsAdaptor

    def _loader() -> Atoms:
        structure = Poscar.from_file(data_dir / "POSCAR").structure
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.wrap()
        return atoms

    return _loader


@pytest.fixture
def prepare_inputs(data_dir: Path):
    def _prepare(
        target: Path,
        *,
        potential: str = "CHGNET",
        incar_overrides: dict[str, str] | None = None,
        extra_bcar: dict[str, str] | None = None,
    ) -> None:
        (target / "POSCAR").write_text((data_dir / "POSCAR").read_text())

        incar_lines = (data_dir / "INCAR").read_text().splitlines()
        if incar_overrides:
            for key, value in incar_overrides.items():
                incar_lines.append(f"{key} = {value}")
        (target / "INCAR").write_text("\n".join(incar_lines) + "\n")

        bcar_lines = (data_dir / "BCAR").read_text().splitlines()
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
        (target / "BCAR").write_text("\n".join(bcar_lines) + "\n")

    return _prepare
