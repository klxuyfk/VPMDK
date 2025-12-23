"""vpmdk: Run machine-learning potentials using VASP style inputs.

The utility consumes VASP-style inputs (POSCAR, INCAR, POTCAR, BCAR) and
executes single-point, relaxation, or molecular dynamics runs with the selected
neural-network potential.  Multiple ASE calculators are supported (CHGNet,
M3GNet/MatGL, MACE, MatterSim, Matlantis) and the expected VASP outputs such as
CONTCAR and OUTCAR-style energy logs are produced.
"""

import argparse
import csv
import importlib
import importlib.util
import os
import re
import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
from pymatgen.io.vasp import Incar, Poscar, Potcar
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from chgnet.model import CHGNetCalculator
except Exception:  # pragma: no cover - optional dependency
    CHGNetCalculator = None  # type: ignore

LegacyM3GNet = None
LegacyM3GNetPotential = None

try:
    from matgl.ext.ase import M3GNetCalculator  # type: ignore

    _USING_LEGACY_M3GNET = False
except Exception:  # pragma: no cover - optional dependency
    try:
        from m3gnet.models import M3GNet as LegacyM3GNet  # type: ignore
        from m3gnet.models import M3GNetCalculator  # type: ignore
        from m3gnet.models import Potential as LegacyM3GNetPotential  # type: ignore

        _USING_LEGACY_M3GNET = True
    except Exception:  # pragma: no cover - optional dependency
        M3GNetCalculator = None  # type: ignore
        LegacyM3GNet = None  # type: ignore
        LegacyM3GNetPotential = None  # type: ignore
        _USING_LEGACY_M3GNET = False

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

try:
    from orb_models.forcefield.calculator import ORBCalculator
    from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS
except Exception:  # pragma: no cover - optional dependency
    ORBCalculator = None  # type: ignore
    ORB_PRETRAINED_MODELS = None  # type: ignore

try:
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FAIRChemCalculator = None  # type: ignore

FAIRChemV1Calculator = None  # type: ignore

try:
    from tensorpotential.calculator.asecalculator import TPCalculator
except Exception:  # pragma: no cover - optional dependency
    TPCalculator = None  # type: ignore

try:
    from tensorpotential.calculator.foundation_models import (
        MODELS_NAME_LIST as GRACE_MODEL_NAMES,
        grace_fm,
    )
except Exception:  # pragma: no cover - optional dependency
    GRACE_MODEL_NAMES: List[str] = []
    grace_fm = None  # type: ignore

try:
    from deepmd.calculator import DP as DeePMDCalculator
except Exception:  # pragma: no cover - optional dependency
    DeePMDCalculator = None  # type: ignore

_sevennet_spec = importlib.util.find_spec("sevennet")
if _sevennet_spec is not None:  # pragma: no cover - optional dependency
    try:
        from sevennet.ase import SevenNetCalculator
    except Exception:  # pragma: no cover - handled dynamically
        SevenNetCalculator = None  # type: ignore
else:  # pragma: no cover - optional dependency
    SevenNetCalculator = None  # type: ignore

from ase import units
from ase.io import write
from ase.io.lammpsdata import Prism
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

_nequip_spec = importlib.util.find_spec("nequip")
_nequip_ase_spec = importlib.util.find_spec("nequip.ase") if _nequip_spec else None
if _nequip_ase_spec is not None:  # pragma: no cover - optional dependency
    from nequip.ase import NequIPCalculator
else:  # pragma: no cover - optional dependency
    NequIPCalculator = None  # type: ignore

DEFAULT_ORB_MODEL = "orb-v3-conservative-20-omat"
DEFAULT_FAIRCHEM_MODEL = "esen-sm-direct-all-oc25"
DEFAULT_GRACE_MODEL = "GRACE-2L-MP-r6"


def _build_nequip_family_calculator(
    bcar_tags: Dict[str, str],
    *,
    require_allegro: bool = False,
    missing_message: str,
):
    """Create NequIP-based calculators that require deployed model files."""

    if require_allegro and importlib.util.find_spec("allegro") is None:
        raise RuntimeError(
            "Allegro calculator not available. Install allegro and dependencies."
        )
    if NequIPCalculator is None:
        raise RuntimeError(missing_message)

    model_path = bcar_tags.get("MODEL")
    model_name = "Allegro" if require_allegro else "NequIP"
    if not model_path:
        raise ValueError(f"{model_name} requires MODEL pointing to a deployed model file.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_name} model not found: {model_path}")

    device = bcar_tags.get("DEVICE")
    if device:
        return NequIPCalculator.from_deployed_model(model_path, device=device)
    return NequIPCalculator.from_deployed_model(model_path)


def _build_nequip_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create a NequIP calculator from a deployed model."""

    return _build_nequip_family_calculator(
        bcar_tags,
        missing_message="NequIPCalculator not available. Install nequip and dependencies.",
    )


def _build_allegro_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create an Allegro calculator from a deployed model."""

    return _build_nequip_family_calculator(
        bcar_tags,
        require_allegro=True,
        missing_message="NequIPCalculator not available. Install nequip and dependencies.",
    )


def _build_chgnet_calculator(bcar_tags: Dict[str, str]):
    """Create a CHGNet calculator with optional DEVICE hint."""

    if CHGNetCalculator is None:
        raise RuntimeError("CHGNetCalculator not available. Install chgnet.")

    model_path = bcar_tags.get("MODEL")
    device = _resolve_device(bcar_tags.get("DEVICE"))
    kwargs = {"use_device": device} if device is not None else {}

    if model_path and os.path.exists(model_path):
        try:
            return CHGNetCalculator(model_path, **kwargs)
        except TypeError:
            return CHGNetCalculator(model_path)

    try:
        return CHGNetCalculator(**kwargs)
    except TypeError:
        return CHGNetCalculator()


def _resolve_device(device: str | None) -> str | None:
    """Return user-specified device or best-effort autodetection."""

    if device is not None:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _build_mace_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create a MACE calculator with optional ``MODEL`` override."""

    if MACECalculator is None:
        raise RuntimeError("MACECalculator not available. Install mace-torch and dependencies.")

    model_path = bcar_tags.get("MODEL")
    device = _resolve_device(bcar_tags.get("DEVICE"))

    if model_path and os.path.exists(model_path):
        return MACECalculator(model_path, device=device)
    return MACECalculator(device=device)


def _build_m3gnet_calculator(bcar_tags: Dict[str, str]):
    """Create a MatGL or legacy M3GNet calculator based on availability."""

    if M3GNetCalculator is None:
        raise RuntimeError("M3GNetCalculator not available. Install matgl or m3gnet.")

    model_path = bcar_tags.get("MODEL")
    device = _resolve_device(bcar_tags.get("DEVICE"))

    if not _USING_LEGACY_M3GNET:
        kwargs = {"device": device} if device is not None else {}
        if model_path and os.path.exists(model_path):
            try:
                return M3GNetCalculator(model_path, **kwargs)
            except TypeError:
                return M3GNetCalculator(model_path)
        try:
            return M3GNetCalculator(**kwargs)
        except TypeError:
            return M3GNetCalculator()

    potential = None
    if model_path and os.path.exists(model_path) and LegacyM3GNetPotential is not None:
        try:
            potential = LegacyM3GNetPotential.from_checkpoint(model_path)
        except Exception:
            try:
                if LegacyM3GNet is not None:
                    potential = LegacyM3GNetPotential(
                        LegacyM3GNet.load(model_path)  # type: ignore[arg-type]
                    )
            except Exception:
                potential = None

    if (
        potential is None
        and LegacyM3GNetPotential is not None
        and LegacyM3GNet is not None
    ):
        potential = LegacyM3GNetPotential(LegacyM3GNet.load())

    if potential is None:
        raise RuntimeError("Legacy M3GNet calculator could not be initialized from available models.")

    if device is not None:
        try:
            return M3GNetCalculator(potential=potential, device=device)
        except TypeError:
            pass

    return M3GNetCalculator(potential=potential)


def _build_simple_model_calculator(
    calculator_cls,
    bcar_tags: Dict[str, str],
    missing_message: str,
):
    """Return calculator initialized with optional ``MODEL`` path."""

    if calculator_cls is None:
        raise RuntimeError(missing_message)

    model_path = bcar_tags.get("MODEL")
    if model_path and os.path.exists(model_path):
        return calculator_cls(model_path)
    return calculator_cls()


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
        text: str = ""
        if isinstance(symbol, str):
            text = symbol.strip()
        elif hasattr(symbol, "symbol"):
            text = str(getattr(symbol, "symbol", "")).strip()
        else:
            try:
                text = str(symbol).strip()
            except Exception:
                continue
        if not text:
            continue
        base = text.split("_", 1)[0].strip()
        normalized.append(base or text)
    return normalized


def _infer_type_map(structure) -> List[str]:
    """Infer a DeePMD type map from the provided structure when possible."""

    labels: List[str] = []
    for attr in ("site_symbols", "species"):
        symbols = getattr(structure, attr, None)
        if symbols:
            labels = _normalize_species_labels(symbols)
            if labels:
                break

    unique: List[str] = []
    for label in labels:
        if label and label not in unique:
            unique.append(label)

    return unique


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


def _count_symbols_in_order(atoms) -> tuple[list[str], list[int]]:
    """Return unique symbols in order of appearance with their counts."""

    counts: OrderedDict[str, int] = OrderedDict()
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return list(counts.keys()), list(counts.values())


_XDATCAR_STATE: Dict[str, Dict[str, Any]] = {}


def _initialize_xdatcar_state(path: str, atoms) -> None:
    """Capture header metadata needed for XDATCAR appends."""

    symbols, counts = _count_symbols_in_order(atoms)
    comment = atoms.info.get("comment", "Generated by ASE")
    scaling = "1.0"
    species_line = " ".join(symbols)
    counts_line = " ".join(str(value) for value in counts)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            header_lines = handle.readlines()
    except FileNotFoundError:
        header_lines = []

    if len(header_lines) >= 1:
        comment = header_lines[0].rstrip("\n")
    if len(header_lines) >= 2:
        scaling = header_lines[1].strip()
    if len(header_lines) >= 6:
        species_line = header_lines[5].strip()
    if len(header_lines) >= 7:
        counts_line = header_lines[6].strip()

    _XDATCAR_STATE[path] = {
        "initial_cell": atoms.get_cell().array.copy(),
        "previous_cell": atoms.get_cell().array.copy(),
        "comment": comment,
        "scaling": scaling,
        "species_line": species_line,
        "counts_line": counts_line,
        "variable_cell": False,
    }


def _append_variable_cell_configuration(path: str, atoms, frame_number: int) -> None:
    """Append a POSCAR-style block when the lattice changes during MD."""

    state = _XDATCAR_STATE.get(path)
    if state is None:
        _initialize_xdatcar_state(path, atoms)
        state = _XDATCAR_STATE[path]

    float_string = "{:11.8f}"
    cell_string = "{:16.10f}"
    scaled_positions = atoms.get_scaled_positions()

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{state['comment']}\n")
        handle.write(f"{state['scaling']}\n")
        for vector in atoms.get_cell().array:
            handle.write(" ".join(cell_string.format(value) for value in vector))
            handle.write("\n")
        handle.write(f"{state['species_line']}\n")
        handle.write(f"{state['counts_line']}\n")
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
        _initialize_xdatcar_state(path, atoms)
        return

    state = _XDATCAR_STATE.get(path)
    if state is None:
        _initialize_xdatcar_state(path, atoms)
        state = _XDATCAR_STATE[path]

    cell_changed = not np.allclose(
        atoms.get_cell().array, state["previous_cell"], rtol=1e-10, atol=1e-12
    )
    state["previous_cell"] = atoms.get_cell().array.copy()

    if state["variable_cell"] or cell_changed:
        state["variable_cell"] = True
        _append_variable_cell_configuration(path, atoms, frame_number)
        return

    _append_xdatcar_configuration(path, atoms, frame_number)


def _write_lammps_trajectory_step(path: str, atoms, step_index: int) -> None:
    """Write or append a LAMMPS trajectory frame for the given MD step."""

    append = step_index != 0
    file_mode = "a" if append else "w"

    prism = Prism(atoms.get_cell().array, atoms.get_pbc())
    lx, ly, lz, xy, xz, yz = prism.get_lammps_prism()
    # Convert the prism representation (box lengths and tilt factors) into the
    # bounds expected by the LAMMPS dump format. See "How a triclinic box is
    # defined" in the LAMMPS documentation for the bound transformation.
    x_tilt_min = min(0.0, xy, xz, xy + xz)
    x_tilt_max = max(0.0, xy, xz, xy + xz)
    xlo = 0.0 - x_tilt_min
    xhi = lx - x_tilt_max
    y_tilt_min = min(0.0, yz)
    y_tilt_max = max(0.0, yz)
    ylo = 0.0 - y_tilt_min
    yhi = ly - y_tilt_max
    zlo = 0.0
    zhi = lz
    pbc_flags = ["pp" if periodic else "ff" for periodic in atoms.get_pbc()]

    species_to_type: Dict[str, int] = {}
    symbols = atoms.get_chemical_symbols()
    for symbol in symbols:
        if symbol not in species_to_type:
            species_to_type[symbol] = len(species_to_type) + 1

    # Obtain atomic positions in the LAMMPS coordinate system without wrapping
    lammps_positions = prism.vector_to_lammps(atoms.get_positions(), wrap=False)
    cell_matrix = prism.cell

    # Convert to fractional coordinates in the LAMMPS cell and extract image flags
    fractional = np.linalg.solve(cell_matrix.T, lammps_positions.T).T
    pbc = np.array(atoms.get_pbc(), dtype=bool)
    image_flags = (np.floor(fractional).astype(int)) * pbc
    scaled_positions = fractional - image_flags

    velocities = atoms.get_velocities()
    velocity_data = None
    if velocities is not None:
        velocity_data = prism.vector_to_lammps(velocities, wrap=False)

    with open(path, file_mode, encoding="utf-8") as handle:
        handle.write("ITEM: TIMESTEP\n")
        handle.write(f"{step_index + 1}\n")
        handle.write("ITEM: NUMBER OF ATOMS\n")
        handle.write(f"{len(atoms)}\n")
        handle.write(
            "ITEM: BOX BOUNDS xy xz yz " + " ".join(pbc_flags) + "\n"
        )
        handle.write(f"{xlo} {xhi} {xy}\n")
        handle.write(f"{ylo} {yhi} {xz}\n")
        handle.write(f"{zlo} {zhi} {yz}\n")

        columns = ["id", "type", "xs", "ys", "zs", "ix", "iy", "iz"]
        if velocity_data is not None:
            columns.extend(["vx", "vy", "vz"])
        handle.write("ITEM: ATOMS " + " ".join(columns) + "\n")

        for index, (scaled, images, symbol) in enumerate(
            zip(scaled_positions, image_flags, symbols), start=1
        ):
            type_id = species_to_type[symbol]
            values = [index, type_id, *scaled.tolist(), *images.tolist()]
            if velocity_data is not None:
                values.extend(velocity_data[index - 1].tolist())
            handle.write(" ".join(str(value) for value in values) + "\n")


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


def _coerce_bool_tag(value: str, tag_name: str) -> bool:
    """Parse boolean-like BCAR tags with descriptive errors."""

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {tag_name} value: {value!r}")


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


def _build_orb_calculator(bcar_tags: Dict[str, str]):
    """Create the ORB ASE calculator configured from BCAR tags."""

    if ORBCalculator is None or ORB_PRETRAINED_MODELS is None:
        raise RuntimeError("ORB calculator not available. Install orb-models and dependencies.")

    model_name = bcar_tags.get("ORB_MODEL") or DEFAULT_ORB_MODEL
    model_factory = ORB_PRETRAINED_MODELS.get(model_name)
    if model_factory is None:
        supported = ", ".join(sorted(ORB_PRETRAINED_MODELS))
        raise ValueError(f"Unsupported ORB model '{model_name}'. Available: {supported}")

    device = bcar_tags.get("DEVICE")
    precision = bcar_tags.get("ORB_PRECISION") or "float32-high"
    compile_value = bcar_tags.get("ORB_COMPILE")
    compile_flag = None if compile_value is None else _coerce_bool_tag(compile_value, "ORB_COMPILE")
    weights_path = bcar_tags.get("MODEL")

    model = model_factory(
        weights_path=weights_path or None,
        device=device,
        precision=precision,
        compile=compile_flag,
        train=False,
    )

    return ORBCalculator(model, device=device)


_FAIRCHEM_V1_IMPORT_PATHS = (
    "ocpmodels.common.relaxation.ase_utils",
    "fairchem.core.common.relaxation.ase_utils",
    "fairchem.common.relaxation.ase_utils",
)


def _get_fairchem_v1_calculator_cls():
    """Return FAIRChem v1 calculator class if installed."""

    global FAIRChemV1Calculator

    first_import_error: Exception | None = None

    if FAIRChemV1Calculator is not None:
        return FAIRChemV1Calculator

    for module_name in _FAIRCHEM_V1_IMPORT_PATHS:
        try:
            spec = importlib.util.find_spec(module_name)
        except Exception as exc:  # pragma: no cover - importlib edge case
            if first_import_error is None:
                first_import_error = exc
            continue

        if spec is None:
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - optional dependency
            if first_import_error is None:
                first_import_error = exc
            continue

        candidate = getattr(module, "OCPCalculator", None)
        if candidate is not None:
            FAIRChemV1Calculator = candidate
            return candidate

    if first_import_error is not None:
        raise RuntimeError(
            "FAIRChem v1 calculator not available due to an import failure."
        ) from first_import_error

    return None


def _build_fairchem_calculator(bcar_tags: Dict[str, str]):
    """Create the FAIRChem ASE calculator configured from BCAR tags."""

    if FAIRChemCalculator is None:
        raise RuntimeError("FAIRChemCalculator not available. Install fairchem and dependencies.")

    model_name = bcar_tags.get("MODEL") or DEFAULT_FAIRCHEM_MODEL
    task_name = bcar_tags.get("FAIRCHEM_TASK")
    inference_settings = bcar_tags.get("FAIRCHEM_INFERENCE_SETTINGS") or "default"
    device = bcar_tags.get("DEVICE")

    return FAIRChemCalculator.from_model_checkpoint(
        model_name,
        task_name=task_name,
        inference_settings=inference_settings,
        device=device,
    )


def _build_fairchem_v1_calculator(bcar_tags: Dict[str, str]):
    """Create the FAIRChem v1 OCPCalculator configured from BCAR tags."""

    calculator_cls = _get_fairchem_v1_calculator_cls()
    if calculator_cls is None:
        raise RuntimeError(
            "FAIRChem v1 calculator not available. Install fairchem v1 (OCP) dependencies."
        )

    model_path = bcar_tags.get("MODEL")
    if not model_path:
        raise ValueError("FAIRChem v1 requires MODEL pointing to a checkpoint file.")

    config_path = bcar_tags.get("FAIRCHEM_CONFIG")
    device = bcar_tags.get("DEVICE")
    cpu_flag = device is not None and device.lower() == "cpu"

    kwargs: Dict[str, Any] = {"checkpoint_path": model_path, "cpu": cpu_flag}
    if config_path:
        kwargs["config_yml"] = config_path

    return calculator_cls(**kwargs)


def _build_grace_calculator(bcar_tags: Dict[str, str]):
    """Create a GRACE (TensorPotential) ASE calculator."""

    if TPCalculator is None:
        raise RuntimeError(
            "TPCalculator not available. Install grace-tensorpotential and dependencies."
        )

    grace_kwargs: Dict[str, Any] = {}

    pad_fraction = _parse_optional_float(
        bcar_tags.get("GRACE_PAD_NEIGHBORS_FRACTION"), key="GRACE_PAD_NEIGHBORS_FRACTION"
    )
    if pad_fraction is not None:
        grace_kwargs["pad_neighbors_fraction"] = pad_fraction

    pad_atoms_raw = bcar_tags.get("GRACE_PAD_ATOMS_NUMBER")
    if pad_atoms_raw is not None:
        grace_kwargs["pad_atoms_number"] = _coerce_int_tag(
            pad_atoms_raw, "GRACE_PAD_ATOMS_NUMBER"
        )

    recompilation_raw = bcar_tags.get("GRACE_MAX_RECOMPILATION")
    if recompilation_raw is not None:
        grace_kwargs["max_number_reduction_recompilation"] = _coerce_int_tag(
            recompilation_raw, "GRACE_MAX_RECOMPILATION"
        )

    min_dist = _parse_optional_float(bcar_tags.get("GRACE_MIN_DIST"), key="GRACE_MIN_DIST")
    if min_dist is not None:
        grace_kwargs["min_dist"] = min_dist

    float_dtype = bcar_tags.get("GRACE_FLOAT_DTYPE")
    if float_dtype:
        grace_kwargs["float_dtype"] = float_dtype

    model_value = bcar_tags.get("MODEL")
    if model_value and os.path.exists(model_value):
        return TPCalculator(model_value, **grace_kwargs)

    available_models = GRACE_MODEL_NAMES
    default_model = DEFAULT_GRACE_MODEL
    if available_models:
        default_model = default_model if default_model in available_models else available_models[0]

    if grace_fm is not None and available_models:
        selected = model_value or default_model
        if selected not in available_models:
            print(
                f"Warning: Unknown GRACE model '{selected}', using default {default_model} instead."
            )
            selected = default_model
        return grace_fm(selected, **grace_kwargs)

    if model_value:
        raise FileNotFoundError(f"GRACE model not found: {model_value}")

    raise RuntimeError(
        "GRACE calculator requires a MODEL path or available foundation models (grace_fm)."
    )


def _build_deepmd_calculator(bcar_tags: Dict[str, str], structure=None):
    """Create a DeePMD-kit calculator configured from BCAR tags."""

    if DeePMDCalculator is None:
        raise RuntimeError(
            "DeePMD-kit calculator not available. Install deepmd-kit and dependencies."
        )

    model_path = bcar_tags.get("MODEL")
    if not model_path:
        raise ValueError("DeePMD-kit requires MODEL pointing to a frozen model file.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DeePMD-kit model not found: {model_path}")

    type_map_value = bcar_tags.get("DEEPMD_TYPE_MAP")
    type_map: List[str] = []
    if type_map_value:
        type_map = [item for item in re.split(r"[\s,]+", type_map_value.strip()) if item]
    elif structure is not None:
        type_map = _infer_type_map(structure)

    kwargs: Dict[str, Any] = {}
    if type_map:
        kwargs["type_map"] = type_map

    return DeePMDCalculator(model=model_path, **kwargs)


_SIMPLE_CALCULATORS: Dict[str, tuple[Any, str]] = {
    "SEVENNET": (
        SevenNetCalculator,
        "SevenNetCalculator not available. Install sevennet.",
    ),
    "MATTERSIM": (
        MatterSimCalculator,
        "MatterSimCalculator not available. Install mattersim and dependencies.",
    ),
}


_CALCULATOR_BUILDERS: Dict[str, Callable[[Dict[str, str]], Any]] = {
    "CHGNET": _build_chgnet_calculator,
    "MATGL": _build_m3gnet_calculator,
    "M3GNET": _build_m3gnet_calculator,
    "MACE": _build_mace_calculator,
    "ALLEGRO": _build_allegro_calculator,
    "NEQUIP": _build_nequip_calculator,
    "MATLANTIS": _build_matlantis_calculator,
    "ORB": _build_orb_calculator,
    "FAIRCHEM": _build_fairchem_calculator,
    "FAIRCHEM_V2": _build_fairchem_calculator,
    "ESEN": _build_fairchem_calculator,
    "FAIRCHEM_V1": _build_fairchem_v1_calculator,
    "GRACE": _build_grace_calculator,
    "DEEPMD": _build_deepmd_calculator,
}

for nnp_name, (calculator_cls, message) in _SIMPLE_CALCULATORS.items():
    _CALCULATOR_BUILDERS[nnp_name] = (
        lambda bcar_tags, *, calculator_cls=calculator_cls, message=message: _build_simple_model_calculator(
            calculator_cls,
            bcar_tags,
            message,
        )
    )


def get_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Return ASE calculator based on BCAR tags."""

    nnp = bcar_tags.get("NNP", "CHGNET").upper()
    builder = _CALCULATOR_BUILDERS.get(nnp)

    if builder is None:
        raise ValueError(f"Unsupported NNP type: {nnp}")

    if builder is _build_deepmd_calculator:
        return builder(bcar_tags, structure=structure)
    return builder(bcar_tags)


def _format_energy_value(value: float) -> str:
    """Return energy in VASP-like ``E`` notation with mantissa < 1."""

    if value == 0:
        return "+.00000000E+00"

    mantissa_str, exponent_str = f"{value:.8e}".split("e")
    mantissa = float(mantissa_str) / 10.0
    exponent = int(exponent_str) + 1
    formatted = f"{mantissa:+.8f}".replace("+0.", "+.").replace("-0.", "-.")
    return f"{formatted}E{exponent:+03d}"


def _extract_numeric_attribute(obj, names: Iterable[str]) -> float:
    """Return first numeric attribute or method result from ``names``."""

    for name in names:
        value = getattr(obj, name, None)
        if callable(value):
            try:
                result = value()
            except Exception:
                continue
            try:
                return float(result)
            except (TypeError, ValueError):
                continue
        else:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def run_single_point(atoms, calculator):
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    delta = 0.0
    print(
        f"{1:4d} F= {_format_energy_value(energy)} "
        f"E0= {_format_energy_value(energy)}  d E ={_format_energy_value(delta)}"
    )
    return energy


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
    smass = (
        _parse_optional_float(incar.get("SMASS"), key="SMASS")
        if "SMASS" in incar
        else None
    )
    mdalgo = int(float(incar.get("MDALGO", 0)))
    if mdalgo == 0 and smass is not None:
        if smass < 0:
            mdalgo = 3
        elif smass > 0:
            mdalgo = 2
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


def _should_write_lammps_trajectory(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests LAMMPS-style trajectory output."""

    value = str(bcar_tags.get("WRITE_LAMMPS_TRAJ", "0")).lower()
    return value in {"1", "true", "yes", "on"}


def _get_lammps_trajectory_interval(bcar_tags: Dict[str, str]) -> int:
    """Return the LAMMPS trajectory write interval requested in BCAR."""

    raw = bcar_tags.get("LAMMPS_TRAJ_INTERVAL", "1")
    interval = _coerce_int_tag(raw, "LAMMPS_TRAJ_INTERVAL")
    if interval <= 0:
        raise ValueError("LAMMPS_TRAJ_INTERVAL must be at least 1")
    return interval


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
    previous_energy: float | None = None
    step_counter = 0
    scalar_pressure = pstress * KBAR_TO_EV_PER_A3 if pstress is not None else None
    scalar_pressure_kwarg = scalar_pressure if scalar_pressure is not None else 0.0

    builder, freeze_required = _make_relaxation_builder(
        isif, scalar_pressure, scalar_pressure_kwarg
    )

    with _temporarily_freeze_atoms(atoms, freeze_required):
        relax_object = builder(atoms)
        dyn = BFGS(relax_object, logfile="OUTCAR")

        def _log_relaxation_energy() -> None:
            nonlocal previous_energy, step_counter
            target = getattr(relax_object, "atoms", atoms)
            energy = target.get_potential_energy()
            delta = 0.0 if previous_energy is None else energy - previous_energy
            previous_energy = energy
            step_counter += 1
            print(
                f"{step_counter:4d} F= {_format_energy_value(energy)} "
                f"E0= {_format_energy_value(energy)}  d E ={_format_energy_value(delta)}"
            )

        if write_energy_csv:
            dyn.attach(lambda: energies.append(atoms.get_potential_energy()))
        dyn.attach(_log_relaxation_energy)
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
    write_lammps_traj: bool = False,
    lammps_traj_interval: int = 1,
    lammps_traj_path: str = "lammps.lammpstrj",
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
    md_step = 0

    def _log_md_state() -> None:
        nonlocal md_step
        md_step += 1
        potential_energy = atoms.get_potential_energy()
        try:
            kinetic_energy = atoms.get_kinetic_energy()
        except Exception:
            kinetic_energy = 0.0
        thermostat_potential = _extract_numeric_attribute(
            dyn,
            (
                "thermostat_potential_energy",
                "thermostat_potential",
                "nose_potential_energy",
                "nhc_potential_energy",
            ),
        )
        thermostat_kinetic = _extract_numeric_attribute(
            dyn,
            (
                "thermostat_kinetic_energy",
                "thermostat_kinetic",
                "nose_kinetic_energy",
                "nhc_kinetic_energy",
            ),
        )
        total_energy = potential_energy + kinetic_energy + thermostat_potential + thermostat_kinetic
        try:
            temperature_inst = atoms.get_temperature()
        except Exception:
            temperature_inst = 0.0
        print(
            f"{md_step:7d} T={temperature_inst:7.1f} "
            f"E= {_format_energy_value(total_energy)} "
            f"F= {_format_energy_value(potential_energy)} "
            f"E0= {_format_energy_value(potential_energy)}  "
            f"EK= {_format_energy_value(kinetic_energy)} "
            f"SP= {_format_energy_value(thermostat_potential)} "
            f"SK= {_format_energy_value(thermostat_kinetic)}"
        )

    for i in range(steps):
        dyn.run(1)
        atoms.wrap()
        _log_md_state()
        _write_xdatcar_step("XDATCAR", atoms, i)
        if write_lammps_traj and i % lammps_traj_interval == 0:
            _write_lammps_trajectory_step(lammps_traj_path, atoms, i)
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

    calculator = get_calculator(bcar, structure=structure)
    write_energy_csv = _should_write_energy_csv(bcar)
    write_lammps_traj = _should_write_lammps_trajectory(bcar)
    lammps_traj_interval = _get_lammps_trajectory_interval(bcar) if write_lammps_traj else 1

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
            write_lammps_traj=write_lammps_traj,
            lammps_traj_interval=lammps_traj_interval,
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
