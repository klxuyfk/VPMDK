"""Charge-density prediction helpers and CHGCAR-compatible writers."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.vasp import VaspChargeDensity

from .models import ChargeDensityResult


def _root():
    return sys.modules["vpmdk_core"]


_CHGCAR_GRID_INCAR_TAGS = frozenset(
    {
        "PREC",
        "ENCUT",
        "NGX",
        "NGY",
        "NGZ",
        "NGXF",
        "NGYF",
        "NGZF",
    }
)

_CHARGE_BACKEND_ALIASES = {
    "CHARGE3NET": "CHARGE3NET",
    "CHARGEE3NET": "CHARGE3NET",
}

_PREC_ALIASES = {
    "N": "NORMAL",
    "NORMAL": "NORMAL",
    "M": "MEDIUM",
    "MEDIUM": "MEDIUM",
    "L": "LOW",
    "LOW": "LOW",
    "A": "ACCURATE",
    "ACCURATE": "ACCURATE",
    "H": "HIGH",
    "HIGH": "HIGH",
    "S": "SINGLE",
    "SINGLE": "SINGLE",
    "SN": "SINGLEN",
    "SINGLEN": "SINGLEN",
}

_PREC_GRID_RULES: dict[str, tuple[float, float]] = {
    "LOW": (1.5, 2.0),
    "MEDIUM": (1.5, 2.0),
    "NORMAL": (1.5, 2.0),
    "HIGH": (2.0, 2.0),
    "ACCURATE": (2.0, 2.0),
    # Match VASP 5.4.x semantics, which is the reference available in this repo.
    "SINGLE": (1.5, 1.0),
    "SINGLEN": (1.5, 1.0),
}

_RY_TO_EV = 13.605693009
_ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
_DEFAULT_CHARGE_CUTOFF = 4.0
_DEFAULT_MAX_PROBES_PER_BATCH = 2500
_CHARGE_ENV_BASE_DIR_VAR = "VPMDK_CHARGE_ENV_BASE_DIR"
_CHARGE_MODEL_CONFIG_TAGS = {
    "CHARGE_NUM_INTERACTIONS": ("num_interactions", int),
    "CHARGE_NUM_NEIGHBORS": ("num_neighbors", float),
    "CHARGE_MUL": ("mul", int),
    "CHARGE_LMAX": ("lmax", int),
    "CHARGE_BASIS": ("basis", str),
    "CHARGE_NUM_BASIS": ("num_basis", int),
    "CHARGE_SPIN": ("spin", bool),
}


def _coerce_mapping_value(mapping: Mapping[str, Any], key: str):
    if hasattr(mapping, "get"):
        return mapping.get(key)
    return None


def _coerce_float(value: Any, *, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {key} value: {value!r}") from None


def _coerce_optional_float(value: Any, *, key: str) -> float | None:
    if value is None:
        return None
    return _coerce_float(value, key=key)


def _coerce_grid_shape(grid_shape: Any) -> tuple[int, int, int]:
    try:
        values = tuple(int(v) for v in grid_shape)
    except Exception:
        raise ValueError(f"Invalid grid shape: {grid_shape!r}") from None
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError(f"Invalid grid shape: {grid_shape!r}")
    return values


def _normalize_charge_backend_name(name: str | None) -> str:
    candidate = "CHARGE3NET" if name is None else str(name).strip().upper()
    if not candidate:
        candidate = "CHARGE3NET"
    return _CHARGE_BACKEND_ALIASES.get(candidate, candidate)


def _normalize_prec(value: Any) -> str:
    token = "NORMAL" if value is None else str(value).strip().upper()
    if not token:
        token = "NORMAL"
    normalized = _PREC_ALIASES.get(token, token)
    if normalized not in _PREC_GRID_RULES:
        supported = ", ".join(sorted(_PREC_GRID_RULES))
        raise ValueError(f"Unsupported PREC value {value!r}; expected one of: {supported}.")
    return normalized


def _largest_prime_factor(value: int) -> int:
    n = int(value)
    factor = 2
    largest = 1
    while factor * factor <= n:
        while n % factor == 0:
            largest = factor
            n //= factor
        factor += 1 if factor == 2 else 2
    return max(largest, n)


def _next_even_smooth_number(minimum: float) -> int:
    candidate = max(2, int(np.ceil(float(minimum))))
    if candidate % 2:
        candidate += 1
    while _largest_prime_factor(candidate) > 7:
        candidate += 2
    return candidate


def _get_reference_cell(reference) -> np.ndarray:
    if hasattr(reference, "get_cell"):
        cell = np.asarray(reference.get_cell(), dtype=float)
    elif hasattr(reference, "lattice") and hasattr(reference.lattice, "matrix"):
        cell = np.asarray(reference.lattice.matrix, dtype=float)
    else:
        cell = np.asarray(reference, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 lattice matrix, got shape {cell.shape!r}.")
    return cell


def _coarse_fft_shape_from_cell(cell: np.ndarray, *, encut: float, prec: str) -> tuple[int, int, int]:
    coarse_factor, _ = _PREC_GRID_RULES[prec]
    g_cut = float(np.sqrt(encut / _RY_TO_EV))
    lengths_bohr = np.linalg.norm(cell, axis=1) * _ANGSTROM_TO_BOHR
    counts = [
        _next_even_smooth_number(coarse_factor * g_cut * length / np.pi)
        for length in lengths_bohr
    ]
    return int(counts[0]), int(counts[1]), int(counts[2])


def determine_vasp_fft_grid(reference, incar: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return VASP-like fine FFT-grid dimensions from INCAR and a reference cell."""

    cell = _get_reference_cell(reference)
    explicit_fine = [
        _coerce_optional_float(_coerce_mapping_value(incar, tag), key=tag)
        for tag in ("NGXF", "NGYF", "NGZF")
    ]
    if all(value is not None for value in explicit_fine):
        return tuple(int(value) for value in explicit_fine)  # type: ignore[return-value]

    explicit_coarse = [
        _coerce_optional_float(_coerce_mapping_value(incar, tag), key=tag)
        for tag in ("NGX", "NGY", "NGZ")
    ]
    if any(value is not None for value in explicit_coarse):
        encut = _coerce_optional_float(_coerce_mapping_value(incar, "ENCUT"), key="ENCUT")
        if encut is None:
            if not all(value is not None for value in explicit_coarse):
                raise ValueError(
                    "Unable to determine CHGCAR grid from INCAR without ENCUT or explicit "
                    "NGX/NGY/NGZ or NGXF/NGYF/NGZF."
                )
            coarse_shape = tuple(int(value) for value in explicit_coarse)
        else:
            prec = _normalize_prec(_coerce_mapping_value(incar, "PREC"))
            coarse_shape_list = list(_coarse_fft_shape_from_cell(cell, encut=encut, prec=prec))
            for index, explicit in enumerate(explicit_coarse):
                if explicit is not None:
                    coarse_shape_list[index] = int(explicit)
            coarse_shape = tuple(coarse_shape_list)
    else:
        encut = _coerce_optional_float(_coerce_mapping_value(incar, "ENCUT"), key="ENCUT")
        if encut is None:
            raise ValueError(
                "Unable to determine CHGCAR grid from INCAR without ENCUT or explicit "
                "NGX/NGY/NGZ or NGXF/NGYF/NGZF."
            )
        prec = _normalize_prec(_coerce_mapping_value(incar, "PREC"))
        coarse_shape = _coarse_fft_shape_from_cell(cell, encut=encut, prec=prec)

    prec = _normalize_prec(_coerce_mapping_value(incar, "PREC"))
    _, fine_multiplier = _PREC_GRID_RULES[prec]
    fine_shape = [int(round(value * fine_multiplier)) for value in coarse_shape]
    for index, explicit in enumerate(explicit_fine):
        if explicit is not None:
            fine_shape[index] = int(explicit)
    return int(fine_shape[0]), int(fine_shape[1]), int(fine_shape[2])


def _charge_density_options_from_bcar(bcar_tags: Mapping[str, Any]) -> dict[str, Any]:
    root = _root()
    options: dict[str, Any] = {
        "backend": bcar_tags.get("CHARGE_BACKEND", "CHARGE3NET"),
        "model_path": bcar_tags.get("CHARGE_MODEL"),
        "device": bcar_tags.get("CHARGE_DEVICE"),
        "source_dir": bcar_tags.get("CHARGE_SOURCE_DIR"),
        "python_executable": bcar_tags.get("CHARGE_PYTHON"),
    }

    cutoff = bcar_tags.get("CHARGE_CUTOFF")
    if cutoff is not None:
        options["cutoff"] = _coerce_float(cutoff, key="CHARGE_CUTOFF")
    max_batch = bcar_tags.get("CHARGE_MAX_PROBES_PER_BATCH")
    if max_batch is not None:
        options["max_probes_per_batch"] = _validate_max_probes_per_batch(
            root._coerce_int_tag(
                str(max_batch),
                "CHARGE_MAX_PROBES_PER_BATCH",
            ),
            raw_value=max_batch,
        )
    for tag_name, (option_name, value_type) in _CHARGE_MODEL_CONFIG_TAGS.items():
        raw_value = bcar_tags.get(tag_name)
        if raw_value is None:
            continue
        if value_type is bool:
            options[option_name] = root._parse_optional_bool_tag(dict(bcar_tags), tag_name)
        elif value_type is int:
            options[option_name] = root._coerce_int_tag(str(raw_value), tag_name)
        elif value_type is float:
            options[option_name] = _coerce_float(raw_value, key=tag_name)
        else:
            options[option_name] = str(raw_value)
    return options


def _validate_max_probes_per_batch(
    value: int,
    *,
    raw_value: object | None = None,
) -> int:
    if value <= 0:
        invalid_value = value if raw_value is None else raw_value
        raise ValueError(
            "Invalid CHARGE_MAX_PROBES_PER_BATCH value: "
            f"{invalid_value!r}. Expected a positive integer."
        )
    return int(value)


def _resolve_charge_python(python_executable: str | None) -> str:
    if python_executable:
        return str(Path(python_executable).expanduser())
    env_python = (
        os.environ.get("VPMDK_CHARGE_PYTHON")
        or os.environ.get("VPMDK_CHARGE3NET_PYTHON")
    )
    if env_python:
        resolved = _resolve_charge_env_path(env_python)
        if resolved is not None:
            return resolved
    return sys.executable


def _resolve_charge_env_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    expanded = Path(path_value).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    base_dir = os.environ.get(_CHARGE_ENV_BASE_DIR_VAR)
    if not base_dir:
        return path_value
    return str((Path(base_dir).expanduser() / expanded).resolve())


def _resolve_charge_source_dir(source_dir: str | None) -> str | None:
    if source_dir:
        return str(Path(source_dir).expanduser())
    env_source_dir = (
        os.environ.get("VPMDK_CHARGE_SOURCE_DIR")
        or os.environ.get("VPMDK_CHARGE3NET_SOURCE_DIR")
    )
    return _resolve_charge_env_path(env_source_dir)


def _resolve_charge_model_path(model_path: str | None, source_dir: str | None) -> str | None:
    if model_path:
        return str(Path(model_path).expanduser())
    env_model = os.environ.get("VPMDK_CHARGE_MODEL") or os.environ.get("VPMDK_CHARGE3NET_MODEL")
    if env_model:
        return _resolve_charge_env_path(env_model)
    if source_dir:
        default_model = Path(source_dir) / "models" / "charge3net_mp.pt"
        if default_model.exists():
            return str(default_model)
    return None


def _run_charge3net_backend(
    atoms,
    *,
    grid_shape: tuple[int, int, int],
    model_path: str | None = None,
    device: str | None = None,
    source_dir: str | None = None,
    python_executable: str | None = None,
    cutoff: float | None = None,
    max_probes_per_batch: int = _DEFAULT_MAX_PROBES_PER_BATCH,
    num_interactions: int | None = None,
    num_neighbors: float | None = None,
    mul: int | None = None,
    lmax: int | None = None,
    basis: str | None = None,
    num_basis: int | None = None,
    spin: bool | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    source_dir = _resolve_charge_source_dir(source_dir)
    model_path = _resolve_charge_model_path(model_path, source_dir)
    max_probes_per_batch = _validate_max_probes_per_batch(max_probes_per_batch)
    if not model_path:
        raise RuntimeError(
            "ChargE3Net model checkpoint not found. Set CHARGE_MODEL (or "
            "VPMDK_CHARGE_MODEL). When CHARGE_SOURCE_DIR is set, VPMDK also checks "
            "<CHARGE_SOURCE_DIR>/models/charge3net_mp.pt."
        )

    runner_path = Path(__file__).with_name("charge3net_runner.py")
    python_path = _resolve_charge_python(python_executable)

    with tempfile.TemporaryDirectory(prefix="vpmdk_charge3net_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.npz"
        output_path = Path(tmp_dir) / "density.npz"
        np.savez(
            input_path,
            numbers=np.asarray(atoms.get_atomic_numbers(), dtype=np.int64),
            positions=np.asarray(atoms.get_positions(), dtype=float),
            cell=np.asarray(atoms.get_cell(), dtype=float),
            pbc=np.asarray(atoms.get_pbc(), dtype=bool),
            grid_shape=np.asarray(grid_shape, dtype=np.int64),
        )
        command = [
            python_path,
            str(runner_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model-path",
            str(model_path),
            "--max-probes-per-batch",
            str(int(max_probes_per_batch)),
        ]
        if cutoff is not None:
            command.extend(["--cutoff", str(float(cutoff))])
        if source_dir:
            command.extend(["--source-dir", str(source_dir)])
        if device is not None:
            command.extend(["--device", str(_root()._resolve_device(device))])
        if num_interactions is not None:
            command.extend(["--num-interactions", str(int(num_interactions))])
        if num_neighbors is not None:
            command.extend(["--num-neighbors", str(float(num_neighbors))])
        if mul is not None:
            command.extend(["--mul", str(int(mul))])
        if lmax is not None:
            command.extend(["--lmax", str(int(lmax))])
        if basis is not None:
            command.extend(["--basis", str(basis)])
        if num_basis is not None:
            command.extend(["--num-basis", str(int(num_basis))])
        if spin is not None:
            command.extend(["--spin", "1" if spin else "0"])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or "no output"
            raise RuntimeError(f"ChargE3Net prediction failed: {details}")
        with np.load(output_path) as payload:
            density = np.asarray(payload["density"])
            spin_density = (
                None if "spin_density" not in payload.files else np.asarray(payload["spin_density"])
            )
        return density, spin_density


def predict_charge_density(
    atoms,
    *,
    grid_shape: tuple[int, int, int] | None = None,
    incar: Mapping[str, Any] | None = None,
    reference: Any | None = None,
    backend: str = "CHARGE3NET",
    model_path: str | None = None,
    device: str | None = None,
    source_dir: str | None = None,
    python_executable: str | None = None,
    cutoff: float | None = None,
    max_probes_per_batch: int = _DEFAULT_MAX_PROBES_PER_BATCH,
    num_interactions: int | None = None,
    num_neighbors: float | None = None,
    mul: int | None = None,
    lmax: int | None = None,
    basis: str | None = None,
    num_basis: int | None = None,
    spin: bool | None = None,
) -> ChargeDensityResult:
    """Predict charge density on a user-specified or INCAR-derived grid."""

    if grid_shape is None:
        if incar is None:
            raise ValueError("grid_shape or incar must be provided for charge-density prediction.")
        grid_shape = determine_vasp_fft_grid(reference if reference is not None else atoms, incar)
    grid_shape = _coerce_grid_shape(grid_shape)

    backend_name = _normalize_charge_backend_name(backend)
    if backend_name == "CHARGE3NET":
        density, spin_density = _run_charge3net_backend(
            atoms,
            grid_shape=grid_shape,
            model_path=model_path,
            device=device,
            source_dir=source_dir,
            python_executable=python_executable,
            cutoff=cutoff,
            max_probes_per_batch=max_probes_per_batch,
            num_interactions=num_interactions,
            num_neighbors=num_neighbors,
            mul=mul,
            lmax=lmax,
            basis=basis,
            num_basis=num_basis,
            spin=spin,
        )
    else:
        raise ValueError(f"Unsupported charge-density backend: {backend_name}")

    return ChargeDensityResult(
        atoms=atoms,
        density=density,
        grid_shape=grid_shape,
        backend=backend_name,
        spin_density=spin_density,
        metadata={
            "model_path": model_path,
            "device": "auto" if device is None else _root()._resolve_device(device),
            "source_dir": _resolve_charge_source_dir(source_dir),
            "model_config": {
                key: value
                for key, value in {
                    "num_interactions": num_interactions,
                    "num_neighbors": num_neighbors,
                    "mul": mul,
                    "lmax": lmax,
                    "basis": basis,
                    "num_basis": num_basis,
                    "spin": spin,
                    "cutoff": cutoff,
                    "spin_output": spin_density is not None,
                }.items()
                if value is not None
            },
        },
    )


def charge_density(*args, **kwargs) -> ChargeDensityResult:
    """Backward-compatible alias for :func:`predict_charge_density`."""

    return predict_charge_density(*args, **kwargs)


def write_chgcar(
    path: str | os.PathLike[str],
    atoms,
    density: np.ndarray,
    *,
    spin_density: np.ndarray | None = None,
) -> None:
    """Write a CHGCAR-like file from density arrays in ASE/ChargE3Net units."""

    density_array = np.asarray(density, dtype=float)
    if density_array.ndim != 3:
        raise ValueError(f"CHGCAR density must be a 3D array, got shape {density_array.shape!r}.")
    if spin_density is not None:
        spin_array = np.asarray(spin_density, dtype=float)
        if spin_array.shape != density_array.shape:
            raise ValueError(
                "Spin density shape must match charge density shape: "
                f"{spin_array.shape!r} != {density_array.shape!r}."
            )
    else:
        spin_array = None

    charge = VaspChargeDensity(filename=None)
    charge.atoms.append(atoms.copy())
    charge.chg.append(density_array)
    charge.aug = ""
    charge.augdiff = ""
    if spin_array is not None:
        charge.chgdiff.append(spin_array)
    charge.write(str(path), format="chgcar")
