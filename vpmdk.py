"""vpmdk: Run machine-learning potentials using VASP style inputs.

The utility consumes VASP-style inputs (POSCAR, INCAR, POTCAR, BCAR) and
executes single-point, relaxation, or molecular dynamics runs with the selected
neural-network potential.  Multiple ASE calculators are supported (CHGNet,
M3GNet/MatGL, MACE, MatterSim, Matlantis) and the expected VASP outputs such as
CONTCAR and OUTCAR-style energy logs are produced.
"""

import argparse
import csv
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List

from pymatgen.io.vasp import Incar, Poscar, Potcar
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from chgnet.model import CHGNetCalculator
except Exception:  # pragma: no cover - optional dependency
    CHGNetCalculator = None  # type: ignore

try:
    from matgl.ext.ase import M3GNetCalculator
except Exception:  # pragma: no cover - optional dependency
    M3GNetCalculator = None  # type: ignore

try:
    from mace.calculators import MACECalculator
except Exception:  # pragma: no cover - optional dependency
    MACECalculator = None  # type: ignore

try:
    from mattersim.forcefield import MatterSimCalculator
except Exception:  # pragma: no cover - optional dependency
    MatterSimCalculator = None  # type: ignore

try:
    from pfp_api_client.pfp.estimator import Estimator as MatlantisEstimator
    from pfp_api_client.pfp.estimator import EstimatorCalcMode
    from pfp_api_client.pfp.calculators.ase_calculator import (
        ASECalculator as MatlantisASECalculator,
    )
except Exception:  # pragma: no cover - optional dependency
    MatlantisEstimator = None  # type: ignore
    MatlantisASECalculator = None  # type: ignore
    EstimatorCalcMode = None  # type: ignore

from ase import units
from ase.io import write
from ase.io.vasp import write_vasp_xdatcar
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter, StrainFilter, FixAtoms
from ase.md.verlet import VelocityVerlet
from ase.md import velocitydistribution

try:  # pragma: no cover - optional thermostat dependency
    from ase.md.andersen import Andersen
except Exception:  # pragma: no cover - handled dynamically
    Andersen = None  # type: ignore

try:  # pragma: no cover - optional thermostat dependency
    from ase.md.langevin import Langevin
except Exception:  # pragma: no cover - handled dynamically
    Langevin = None  # type: ignore

try:  # pragma: no cover - optional thermostat dependency
    from ase.md.bussi import Bussi
except Exception:  # pragma: no cover - handled dynamically
    Bussi = None  # type: ignore

try:  # pragma: no cover - optional thermostat dependency
    from ase.md.nose_hoover_chain import NoseHooverChainNVT
except Exception:  # pragma: no cover - handled dynamically
    NoseHooverChainNVT = None  # type: ignore


def parse_key_value_file(path: str) -> Dict[str, str]:
    """Parse simple key=value style file."""
    data: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            for comment in ("#", "!"):
                if comment in line:
                    line = line.split(comment, 1)[0]
            line = line.strip()
            if not line or '=' not in line:
                continue
            k, v = line.split('=', 1)
            data[k.strip().upper()] = v.strip()
    return data


def _flatten(values: Iterable[object]) -> List[float]:
    """Return flattened list of floats from nested sequences."""

    flattened: List[float] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            flattened.extend(_flatten(item))
        else:
            try:
                flattened.append(float(item))
            except (TypeError, ValueError):
                continue
    return flattened


def _parse_magmom_values(value) -> List[float]:
    """Parse VASP-style MAGMOM definition into a list of floats."""

    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return _flatten(value)

    text = str(value).strip()
    if not text:
        return []

    tokens = text.replace(",", " ").split()
    result: List[float] = []
    for token in tokens:
        if not token:
            continue
        if "*" in token:
            count_str, moment_str = token.split("*", 1)
            try:
                count = int(float(count_str))
            except (TypeError, ValueError):
                continue
            nested = _parse_magmom_values(moment_str)
            if not nested:
                try:
                    nested = [float(moment_str)]
                except (TypeError, ValueError):
                    continue
            if len(nested) == 1:
                result.extend(nested * count)
            else:
                for _ in range(count):
                    result.extend(nested)
            continue
        try:
            result.append(float(token))
        except (TypeError, ValueError):
            continue
    return result


def _normalize_species_labels(symbols: Iterable[object]) -> List[str]:
    """Return species labels with POTCAR-style suffixes removed."""

    normalized: List[str] = []
    for symbol in symbols:
        if not isinstance(symbol, str):
            continue
        text = symbol.strip()
        if not text:
            continue
        base = text.split("_", 1)[0].strip()
        normalized.append(base or text)
    return normalized


def _expand_magmom_to_atoms(magmoms: List[float], atoms) -> List[float] | None:
    """Expand species MAGMOM values to per-atom list when necessary."""

    if not magmoms:
        return None

    num_atoms = len(atoms)
    if len(magmoms) == num_atoms:
        return magmoms

    symbols = atoms.get_chemical_symbols()
    species_counts: List[int] = []
    previous_symbol: str | None = None
    for symbol in symbols:
        if symbol == previous_symbol:
            species_counts[-1] += 1
        else:
            species_counts.append(1)
            previous_symbol = symbol

    if len(magmoms) == len(species_counts):
        expanded: List[float] = []
        for moment, count in zip(magmoms, species_counts):
            expanded.extend([moment] * count)
        return expanded

    return None


def _apply_initial_magnetization(atoms, incar) -> None:
    """Populate initial magnetic moments from INCAR when available."""

    if not hasattr(incar, "get"):
        return
    if "MAGMOM" not in incar:
        return

    raw = incar.get("MAGMOM")
    magmoms = _parse_magmom_values(raw)
    if not magmoms:
        return
    expanded = _expand_magmom_to_atoms(magmoms, atoms)
    if expanded is None or len(expanded) != len(atoms):
        print(
            "Warning: Unable to reconcile MAGMOM values with number of atoms; "
            "initial magnetic moments will not be set."
        )
        return
    atoms.set_initial_magnetic_moments(expanded)


def _append_xdatcar_configuration(path: str, atoms, frame_number: int) -> None:
    """Append a single XDATCAR configuration block for ``atoms``."""

    scaled_positions = atoms.get_scaled_positions()
    float_string = "{:11.8f}"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"Direct configuration={frame_number:6d}\n")
        for row in scaled_positions:
            handle.write(" ")
            handle.write(" ".join(float_string.format(value) for value in row))
            handle.write("\n")


def _rewrite_first_xdatcar_frame(path: str, atoms) -> None:
    """Ensure the first XDATCAR frame uses direct (fractional) coordinates."""

    scaled_positions = atoms.get_scaled_positions()
    float_string = "{:11.8f}"
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    configuration_index: int | None = None
    for index, line in enumerate(lines):
        if "configuration=" in line.lower():
            configuration_index = index
            break

    if configuration_index is None:
        return

    lines[configuration_index] = f"Direct configuration={1:6d}\n"
    start = configuration_index + 1
    formatted_positions = [
        " " + " ".join(float_string.format(value) for value in row) + "\n"
        for row in scaled_positions
    ]
    end = start + len(formatted_positions)
    lines[start:end] = formatted_positions

    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def _write_xdatcar_step(path: str, atoms, step_index: int) -> None:
    """Write or append an XDATCAR snapshot for the given MD ``step_index``."""

    frame_number = step_index + 1
    if step_index == 0:
        with open(path, "w", encoding="utf-8") as handle:
            write_vasp_xdatcar(handle, [atoms])
        _rewrite_first_xdatcar_frame(path, atoms)
        return

    _append_xdatcar_configuration(path, atoms, frame_number)


def read_structure(poscar_path: str, potcar_path: str | None = None):
    """Read POSCAR and reconcile species with POTCAR if necessary."""
    poscar = Poscar.from_file(poscar_path)
    structure = poscar.structure
    if potcar_path and os.path.exists(potcar_path):
        try:
            potcar = Potcar.from_file(potcar_path)
            potcar_symbols = getattr(potcar, "symbols", [])
        except Exception:
            potcar_symbols = []
        normalized_potcar_symbols = _normalize_species_labels(potcar_symbols)
        if normalized_potcar_symbols:
            # check consistency
            if poscar.site_symbols and len(poscar.site_symbols) == len(normalized_potcar_symbols):
                normalized_poscar_symbols = _normalize_species_labels(poscar.site_symbols)
                if normalized_poscar_symbols != normalized_potcar_symbols:
                    print(
                        "Warning: species in POSCAR and POTCAR differ. "
                        f"Using POTCAR order: {normalized_potcar_symbols}"
                    )
                    poscar.site_symbols = normalized_potcar_symbols
                    structure = poscar.structure
                elif list(poscar.site_symbols) != normalized_potcar_symbols:
                    poscar.site_symbols = normalized_potcar_symbols
                    structure = poscar.structure
            elif not poscar.site_symbols:
                poscar.site_symbols = normalized_potcar_symbols
                structure = poscar.structure
    else:
        if not poscar.site_symbols:
            print("Warning: POSCAR has no species names and no POTCAR provided.")
    return structure


def _coerce_int_tag(value: str, tag_name: str) -> int:
    """Parse integer BCAR tag values with a descriptive error message."""

    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {tag_name} value: {value!r}") from None


def _list_matlantis_calc_modes() -> str:
    """Return comma-separated list of available Matlantis calc modes."""

    if EstimatorCalcMode is None:
        return ""
    members = getattr(EstimatorCalcMode, "__members__", None)
    if isinstance(members, dict) and members:
        return ", ".join(sorted(members))
    candidates = [name for name in dir(EstimatorCalcMode) if name.isupper()]
    if candidates:
        return ", ".join(sorted(candidates))
    return ""


def _resolve_matlantis_calc_mode(name):
    """Return ``EstimatorCalcMode`` or passthrough string for Matlantis calc mode."""

    if EstimatorCalcMode is None:
        raise RuntimeError(
            "Matlantis EstimatorCalcMode not available. Install pfp-api-client."
        )

    if isinstance(name, EstimatorCalcMode):
        return name

    if name is None:
        raise ValueError("MATLANTIS_CALC_MODE must not be None")

    text = str(name)
    normalized = text.upper()

    candidate = getattr(EstimatorCalcMode, normalized, None)
    if candidate is not None:
        return candidate

    members = getattr(EstimatorCalcMode, "__members__", None)
    if isinstance(members, dict) and normalized in members:
        return members[normalized]

    try:
        return EstimatorCalcMode[normalized]  # type: ignore[index]
    except Exception:
        pass

    try:
        return EstimatorCalcMode(normalized)  # type: ignore[call-arg]
    except Exception:
        pass

    return text


def _build_matlantis_calculator(bcar_tags: Dict[str, str]):
    """Create the Matlantis ASE calculator configured from BCAR tags."""

    if MatlantisEstimator is None or MatlantisASECalculator is None or EstimatorCalcMode is None:
        raise RuntimeError(
            "Matlantis calculator not available. Install pfp-api-client and dependencies."
        )

    model_version = (
        bcar_tags.get("MATLANTIS_MODEL_VERSION")
        or bcar_tags.get("MODEL_VERSION")
        or bcar_tags.get("MODEL")
        or "v8.0.0"
    )
    priority_raw = bcar_tags.get("MATLANTIS_PRIORITY") or bcar_tags.get("PRIORITY")
    priority = 50 if priority_raw is None else _coerce_int_tag(priority_raw, "MATLANTIS_PRIORITY")

    calc_mode_value = bcar_tags.get("MATLANTIS_CALC_MODE") or bcar_tags.get("CALC_MODE")
    calc_mode = _resolve_matlantis_calc_mode(calc_mode_value or "PBE")

    estimator_kwargs: Dict[str, Any] = {
        "model_version": model_version,
        "priority": priority,
        "calc_mode": calc_mode,
    }

    return MatlantisASECalculator(MatlantisEstimator(**estimator_kwargs))


def get_calculator(bcar_tags: Dict[str, str]):
    """Return ASE calculator based on BCAR tags."""
    nnp = bcar_tags.get("NNP", "CHGNET").upper()
    if nnp == "CHGNET":
        if CHGNetCalculator is None:
            raise RuntimeError("CHGNetCalculator not available. Install chgnet.")
        model_path = bcar_tags.get("MODEL")
        if model_path and os.path.exists(model_path):
            return CHGNetCalculator(model_path)
        return CHGNetCalculator()
    if nnp == "MATGL":
        if M3GNetCalculator is None:
            raise RuntimeError(
                "M3GNetCalculator not available. Install matgl and dependencies."
            )
        model_path = bcar_tags.get("MODEL")
        if model_path and os.path.exists(model_path):
            return M3GNetCalculator(model_path)
        return M3GNetCalculator()
    if nnp == "MACE":
        if MACECalculator is None:
            raise RuntimeError(
                "MACECalculator not available. Install mace-torch and dependencies."
            )
        model_path = bcar_tags.get("MODEL")
        device = bcar_tags.get("DEVICE")
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        if model_path and os.path.exists(model_path):
            return MACECalculator(model_path, device=device)
        return MACECalculator(device=device)
    if nnp == "MATTERSIM":
        if MatterSimCalculator is None:
            raise RuntimeError(
                "MatterSimCalculator not available. Install mattersim and dependencies."
            )
        model_path = bcar_tags.get("MODEL")
        if model_path and os.path.exists(model_path):
            return MatterSimCalculator(model_path)
        return MatterSimCalculator()
    if nnp == "MATLANTIS":
        return _build_matlantis_calculator(bcar_tags)
    raise ValueError(f"Unsupported NNP type: {nnp}")


def run_single_point(atoms, calculator):
    atoms.calc = calculator
    e = atoms.get_potential_energy()
    print(f"  energy  without entropy=     {e:10.6f} eV")
    return e


KBAR_TO_EV_PER_A3 = 0.1 / 160.21766208


@dataclass(frozen=True)
class IncarSettings:
    """Container for the INCAR parameters that drive the simulation."""

    nsw: int = 0
    ibrion: int = -1
    ediffg: float = -0.02
    isif: int = 2
    pstress: float | None = None
    tebeg: float = 300.0
    teend: float = 300.0
    potim: float = 2.0
    mdalgo: int = 0
    smass: float | None = None
    thermostat_params: Dict[str, float] = field(default_factory=dict)

    @property
    def energy_tolerance(self) -> float | None:
        """Energy convergence threshold in eV when EDIFFG>0."""

        return self.ediffg if self.ediffg > 0 else None

    @property
    def force_limit(self) -> float:
        """Return ASE ``fmax`` argument derived from EDIFFG semantics."""

        if self.ediffg > 0:
            return -abs(self.ediffg)
        if self.ediffg < 0:
            return abs(self.ediffg)
        return 0.05


SUPPORTED_INCAR_TAGS = {
    "ISIF",
    "IBRION",
    "NSW",
    "EDIFFG",
    "PSTRESS",
    "TEBEG",
    "TEEND",
    "POTIM",
    "MDALGO",
    "SMASS",
    "ANDERSEN_PROB",
    "LANGEVIN_GAMMA",
    "CSVR_PERIOD",
    "NHC_NCHAINS",
    "MAGMOM",
}

SUPPORTED_ISIF_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8}


_NUMERIC_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _load_incar(path: str):
    """Return ``Incar`` contents when available, falling back to ``{}``."""

    if os.path.exists(path):
        return Incar.from_file(path)
    return {}


def _warn_for_unsupported_incar_tags(incar) -> None:
    """Emit warnings for INCAR options that are silently ignored."""

    for key in getattr(incar, "keys", lambda: [])():
        if key not in SUPPORTED_INCAR_TAGS:
            print(f"INCAR tag {key} is not supported and will be ignored")


def _parse_optional_float(value, *, key: str):
    """Attempt to convert ``value`` to ``float`` with warning on failure."""

    if value is None:
        return None
    candidate = value
    if isinstance(value, str):
        match = _NUMERIC_RE.search(value)
        if match is not None:
            candidate = match.group(0)
        else:
            candidate = value.strip()
    try:
        return float(candidate)
    except (TypeError, ValueError):
        print(f"Warning: Unable to parse {key}; ignoring value {value}")
        return None


def _normalize_isif(requested: int) -> int:
    """Map request to supported ISIF behaviour while preserving warnings."""

    if requested not in SUPPORTED_ISIF_VALUES:
        print(
            "Warning: ISIF="
            f"{requested} is not fully supported; defaulting to ISIF=2 behavior."
        )
        return 2
    if requested in (0, 1, 2):
        return 2
    return requested


def _extract_thermostat_parameters(incar) -> Dict[str, float]:
    """Collect thermostat keywords from ``incar`` with validation."""

    params: Dict[str, float] = {}
    keys = ("ANDERSEN_PROB", "LANGEVIN_GAMMA", "CSVR_PERIOD", "NHC_NCHAINS")
    for key in keys:
        if hasattr(incar, "__contains__") and key in incar:
            value = incar[key]
            if key == "NHC_NCHAINS":
                try:
                    coerced = int(float(value))
                except (TypeError, ValueError):
                    print(f"Warning: Unable to parse {key}; ignoring value {value}")
                    parsed = None
                else:
                    parsed = _parse_optional_float(coerced, key=key)
            else:
                parsed = _parse_optional_float(value, key=key)
            if parsed is not None:
                params[key] = float(parsed)
    return params


def _load_incar_settings(incar) -> IncarSettings:
    """Translate INCAR dictionary-like object into :class:`IncarSettings`."""

    if not hasattr(incar, "get"):
        return IncarSettings()

    nsw = int(float(incar.get("NSW", 0)))
    ibrion = int(float(incar.get("IBRION", -1)))
    ediffg = float(incar.get("EDIFFG", -0.02))
    pstress = None
    if "PSTRESS" in incar:
        pstress = _parse_optional_float(incar.get("PSTRESS", 0.0), key="PSTRESS")
    tebeg_default = 300.0
    tebeg_value = incar.get("TEBEG", tebeg_default)
    parsed_tebeg = _parse_optional_float(tebeg_value, key="TEBEG")
    tebeg = parsed_tebeg if parsed_tebeg is not None else tebeg_default

    teend_value = incar.get("TEEND", tebeg)
    parsed_teend = _parse_optional_float(teend_value, key="TEEND")
    teend = parsed_teend if parsed_teend is not None else tebeg
    potim = float(incar.get("POTIM", 2.0))
    mdalgo = int(float(incar.get("MDALGO", 0)))
    smass = (
        _parse_optional_float(incar.get("SMASS"), key="SMASS")
        if "SMASS" in incar
        else None
    )
    thermostat_params = _extract_thermostat_parameters(incar)
    requested_isif = int(float(incar.get("ISIF", 2)))
    normalized_isif = _normalize_isif(requested_isif)

    return IncarSettings(
        nsw=nsw,
        ibrion=ibrion,
        ediffg=ediffg,
        isif=normalized_isif,
        pstress=pstress,
        tebeg=tebeg,
        teend=teend,
        potim=potim,
        mdalgo=mdalgo,
        smass=smass,
        thermostat_params=thermostat_params,
    )


def _should_write_energy_csv(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests CSV output of ionic energies."""

    value = str(bcar_tags.get("WRITE_ENERGY_CSV", "0")).lower()
    return value in {"1", "true", "yes", "on"}


class _EnergyConvergenceMonitor:
    """Track ionic step energies and test for convergence."""

    def __init__(self, atoms, tolerance: float):
        self._atoms = atoms
        self._tolerance = tolerance
        self._previous: float | None = None

    def update(self) -> bool:
        """Return True when the total energy change falls below the tolerance."""

        energy = self._atoms.get_potential_energy()
        if self._previous is None:
            self._previous = energy
            return False
        delta = abs(energy - self._previous)
        self._previous = energy
        return delta <= self._tolerance


def _make_relaxation_builder(
    isif: int,
    scalar_pressure: float | None,
    scalar_pressure_kwarg: float,
) -> tuple[Callable[[object], object], bool]:
    """Return a factory for the relaxation object and freeze requirement."""

    def build_identity(atoms):
        return atoms

    if isif == 3:
        if scalar_pressure is None:
            return UnitCellFilter, False

        def build_ucf(atoms):
            return UnitCellFilter(atoms, scalar_pressure=scalar_pressure)

        return build_ucf, False

    if isif == 4:

        def build_constant_volume(atoms):
            return UnitCellFilter(
                atoms,
                constant_volume=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_constant_volume, False

    if isif == 5:

        def build_constant_volume_frozen(atoms):
            return UnitCellFilter(
                atoms,
                constant_volume=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_constant_volume_frozen, True

    if isif == 6:
        return StrainFilter, False

    if isif == 7:

        def build_hydrostatic_frozen(atoms):
            return UnitCellFilter(
                atoms,
                mask=[1, 1, 1, 0, 0, 0],
                hydrostatic_strain=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_hydrostatic_frozen, True

    if isif == 8:

        def build_hydrostatic(atoms):
            return UnitCellFilter(
                atoms,
                mask=[1, 1, 1, 0, 0, 0],
                hydrostatic_strain=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_hydrostatic, False

    return build_identity, False


@contextmanager
def _temporarily_freeze_atoms(atoms, freeze_required: bool):
    """Temporarily constrain ionic positions when required by ISIF."""

    if not freeze_required:
        yield
        return

    current_constraints = getattr(atoms, "constraints", None)
    if current_constraints is None:
        original_constraints = None
        base_constraints: list[object] = []
    else:
        try:
            base_constraints = list(current_constraints)
        except TypeError:
            base_constraints = [current_constraints]
        original_constraints = base_constraints

    frozen = FixAtoms(indices=list(range(len(atoms))))
    atoms.set_constraint(base_constraints + [frozen])
    try:
        yield
    finally:
        if original_constraints is None:
            atoms.set_constraint()
        else:
            atoms.set_constraint(original_constraints)
def run_relaxation(
    atoms,
    calculator,
    steps: int,
    fmax: float,
    write_energy_csv: bool = False,
    isif: int = 2,
    pstress: float | None = None,
    energy_tolerance: float | None = None,
):
    atoms.calc = calculator
    energies: List[float] = []
    scalar_pressure = pstress * KBAR_TO_EV_PER_A3 if pstress is not None else None
    scalar_pressure_kwarg = scalar_pressure if scalar_pressure is not None else 0.0

    builder, freeze_required = _make_relaxation_builder(
        isif, scalar_pressure, scalar_pressure_kwarg
    )

    with _temporarily_freeze_atoms(atoms, freeze_required):
        relax_object = builder(atoms)
        dyn = BFGS(relax_object, logfile="OUTCAR")
        if write_energy_csv:
            dyn.attach(lambda: energies.append(atoms.get_potential_energy()))
        if energy_tolerance is None:
            dyn.run(fmax=fmax, steps=steps)
        else:
            monitor = _EnergyConvergenceMonitor(atoms, energy_tolerance)
            dyn.fmax = fmax
            for force_converged in dyn.irun(steps=steps):
                energy_converged = monitor.update()
                if energy_converged or force_converged:
                    break

    target_atoms = getattr(relax_object, "atoms", atoms)
    target_atoms.wrap()
    write("CONTCAR", target_atoms, direct=True)
    if write_energy_csv:
        with open("energy.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for energy in energies:
                writer.writerow([energy])
    return target_atoms.get_potential_energy()


def _rescale_velocities(atoms, target_temperature: float) -> None:
    """Scale velocities so that kinetic temperature approaches target."""

    if target_temperature <= 0:
        velocities = atoms.get_velocities()
        if velocities is None:
            zeros = [[0.0, 0.0, 0.0] for _ in range(len(atoms))]
            atoms.set_velocities(zeros)
        else:
            atoms.set_velocities(velocities * 0.0)
        return

    ndof = getattr(atoms, "get_number_of_degrees_of_freedom", lambda: 0)()
    if ndof <= 0:
        velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return

    kinetic_energy = atoms.get_kinetic_energy()
    if kinetic_energy <= 0:
        velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return

    current_temperature = 2.0 * kinetic_energy / (ndof * units.kB)
    if current_temperature <= 0:
        velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return

    scaling = (target_temperature / current_temperature) ** 0.5
    velocities = atoms.get_velocities()
    if velocities is None:
        velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return
    atoms.set_velocities(velocities * scaling)


def _estimate_tdamp(smass: float | None, timestep: float) -> float:
    """Return Nose-Hoover time constant (in fs)."""

    if smass is None or smass == 0:
        return max(100.0 * timestep, timestep)
    return abs(smass)


def _select_md_dynamics(
    atoms,
    mdalgo: int,
    timestep: float,
    initial_temperature: float,
    smass: float | None,
    thermostat_params: Dict[str, float],
):
    """Create ASE molecular dynamics driver and temperature updater."""

    timestep_ase = timestep * units.fs

    def default_update(temp: float) -> None:
        _rescale_velocities(atoms, temp)

    def make_update(dyn, *, allow_attribute_update: bool = False):
        def update(temp: float) -> None:
            try:
                dyn.set_temperature(temperature_K=temp)
            except TypeError:
                dyn.set_temperature(temp)
            except AttributeError:
                if not allow_attribute_update:
                    raise
                dyn.temp = temp * units.kB
                dyn.target_kinetic_energy = 0.5 * dyn.temp * dyn.ndof
            _rescale_velocities(atoms, temp)

        return update

    if mdalgo == 1:
        if Andersen is None:
            raise RuntimeError(
                "Andersen thermostat requested but ase.md.andersen.Andersen "
                "is unavailable. Install the optional ASE thermostat "
                "dependencies or choose a supported MDALGO value."
            )
        andersen_prob = float(thermostat_params.get("ANDERSEN_PROB", 0.1))
        dyn = Andersen(
            atoms,
            timestep_ase,
            temperature_K=initial_temperature,
            andersen_prob=andersen_prob,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn)

    if mdalgo in (2, 4) and NoseHooverChainNVT is not None:
        tdamp_fs = _estimate_tdamp(smass, timestep)
        if mdalgo == 2:
            chain_length = int(thermostat_params.get("NHC_NCHAINS", 1))
        else:
            chain_length = int(thermostat_params.get("NHC_NCHAINS", 3))
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=timestep_ase,
            temperature_K=initial_temperature,
            tdamp=tdamp_fs * units.fs,
            tchain=chain_length,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn)
    if mdalgo in (2, 4) and NoseHooverChainNVT is None and mdalgo != 0:
        raise RuntimeError(
            "Nose-Hoover thermostat requested but ase.md.nose_hoover_chain.NoseHooverChainNVT "
            "is unavailable. Install the optional ASE thermostat dependencies or choose "
            "a supported MDALGO value."
        )

    if mdalgo == 3:
        if Langevin is None:
            raise RuntimeError(
                "Langevin thermostat requested but ase.md.langevin.Langevin "
                "is unavailable. Install the optional ASE thermostat dependencies or "
                "choose a supported MDALGO value."
            )
        gamma = thermostat_params.get("LANGEVIN_GAMMA")
        if gamma is None and smass is not None and smass < 0:
            gamma = abs(smass)
        if gamma is None:
            gamma = 1.0
        friction = (float(gamma) / 1000.0) / units.fs
        dyn = Langevin(
            atoms,
            timestep_ase,
            temperature_K=initial_temperature,
            friction=friction,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn)

    if mdalgo == 5:
        if Bussi is None:
            raise RuntimeError(
                "CSVR thermostat requested but ase.md.bussi.Bussi is unavailable. "
                "Install the optional ASE thermostat dependencies or choose a supported "
                "MDALGO value."
            )
        taut = thermostat_params.get("CSVR_PERIOD")
        if taut is None:
            taut = max(100.0 * timestep, timestep)
        dyn = Bussi(
            atoms,
            timestep_ase,
            temperature_K=initial_temperature,
            taut=float(taut) * units.fs,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn, allow_attribute_update=True)

    dyn = VelocityVerlet(atoms, timestep_ase, logfile="OUTCAR")
    return dyn, default_update


def run_md(
    atoms,
    calculator,
    steps: int,
    temperature: float,
    timestep: float,
    *,
    mdalgo: int = 0,
    teend: float | None = None,
    smass: float | None = None,
    thermostat_params: Dict[str, float] | None = None,
):
    atoms.calc = calculator
    if temperature <= 0:
        velocities = atoms.get_velocities()
        if velocities is None:
            zeros = [[0.0, 0.0, 0.0] for _ in range(len(atoms))]
            atoms.set_velocities(zeros)
        else:
            atoms.set_velocities(velocities * 0.0)
    else:
        velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=temperature
        )
    params = thermostat_params or {}
    dyn, update_temperature = _select_md_dynamics(
        atoms,
        mdalgo,
        timestep,
        temperature,
        smass,
        params,
    )
    target_end = temperature if teend is None else teend
    for i in range(steps):
        dyn.run(1)
        atoms.wrap()
        _write_xdatcar_step("XDATCAR", atoms, i)
        if steps > 1 and i + 1 < steps and target_end != temperature:
            next_temp = temperature + (target_end - temperature) * (i + 1) / (steps - 1)
            update_temperature(next_temp)
    atoms.wrap()
    write("CONTCAR", atoms, direct=True)
    return atoms.get_potential_energy()


def main():
    parser = argparse.ArgumentParser(description="Run NNP with VASP style inputs")
    parser.add_argument("--dir", default=".", help="Input directory")
    args = parser.parse_args()
    workdir = args.dir

    poscar_path = os.path.join(workdir, "POSCAR")
    incar_path = os.path.join(workdir, "INCAR")
    potcar_path = os.path.join(workdir, "POTCAR")
    bcar_path = os.path.join(workdir, "BCAR")

    for fname in ["KPOINTS", "WAVECAR", "CHGCAR"]:
        if os.path.exists(os.path.join(workdir, fname)):
            print(f"Note: {fname} detected but not used in NNP calculations.")

    if not os.path.exists(poscar_path):
        print("POSCAR not found.")
        sys.exit(1)

    structure = read_structure(poscar_path, potcar_path if os.path.exists(potcar_path) else None)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.wrap()

    incar = _load_incar(incar_path)
    _apply_initial_magnetization(atoms, incar)
    bcar = parse_key_value_file(bcar_path) if os.path.exists(bcar_path) else {}

    _warn_for_unsupported_incar_tags(incar)
    settings = _load_incar_settings(incar)

    calculator = get_calculator(bcar)
    write_energy_csv = _should_write_energy_csv(bcar)

    if settings.nsw <= 0 or settings.ibrion < 0:
        run_single_point(atoms, calculator)
    elif settings.ibrion == 0:
        run_md(
            atoms,
            calculator,
            settings.nsw,
            settings.tebeg,
            settings.potim,
            mdalgo=settings.mdalgo,
            teend=settings.teend,
            smass=settings.smass,
            thermostat_params=settings.thermostat_params,
        )
    else:
        run_relaxation(
            atoms,
            calculator,
            settings.nsw,
            settings.force_limit,
            write_energy_csv,
            isif=settings.isif,
            pstress=settings.pstress,
            energy_tolerance=settings.energy_tolerance,
        )

    print("Calculation completed.")


if __name__ == "__main__":
    main()
