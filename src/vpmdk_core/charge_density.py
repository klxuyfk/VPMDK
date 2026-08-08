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
    "DEEPDFT": "DEEPDFT",
    "DEEP_DFT": "DEEPDFT",
    "DEEPCDP": "DEEPCDP",
    "DEEP_CDP": "DEEPCDP",
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

# The DeePCDP option tags _charge_density_options_from_bcar reads
# INDIVIDUALLY (not via a dict): kept as a set beside those reads so the
# BCAR vocabulary harvest (_known_bcar_tags) cannot drift from the consumers
# again -- the R161 harvest missed exactly these eight and falsely warned
# "not recognized" for a documented, fully-consumed configuration.
_DEEPCDP_OPTION_TAGS = frozenset(
    {
        "CHARGE_DEEPCDP_METADATA",
        "CHARGE_DEEPCDP_SPECIES",
        "CHARGE_DEEPCDP_RCUT",
        "CHARGE_DEEPCDP_NMAX",
        "CHARGE_DEEPCDP_LMAX",
        "CHARGE_DEEPCDP_SIGMA",
        "CHARGE_DEEPCDP_PERIODIC",
        "CHARGE_DEEPCDP_ACTIVATION",
    }
)

_DEEPCDP_WEIGHTING_KEYS = {
    "CHARGE_DEEPCDP_WEIGHTING_FUNCTION": ("function", str),
    "CHARGE_DEEPCDP_WEIGHTING_R0": ("r0", float),
    "CHARGE_DEEPCDP_WEIGHTING_C": ("c", float),
    "CHARGE_DEEPCDP_WEIGHTING_M": ("m", float),
    "CHARGE_DEEPCDP_WEIGHTING_D": ("d", float),
}

_CHARGE_BACKEND_ENV_VARS = {
    "CHARGE3NET": {
        "python": ("VPMDK_CHARGE3NET_PYTHON",),
        "source_dir": ("VPMDK_CHARGE3NET_SOURCE_DIR",),
        "model": ("VPMDK_CHARGE3NET_MODEL",),
    },
    "DEEPDFT": {
        "python": ("VPMDK_DEEPDFT_PYTHON",),
        "source_dir": ("VPMDK_DEEPDFT_SOURCE_DIR",),
        "model": ("VPMDK_DEEPDFT_MODEL",),
    },
    "DEEPCDP": {
        "python": ("VPMDK_DEEPCDP_PYTHON",),
        "source_dir": ("VPMDK_DEEPCDP_SOURCE_DIR",),
        "model": ("VPMDK_DEEPCDP_MODEL",),
    },
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


def _coerce_positive_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {key} value: {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {key} value: {value!r}") from None
    if not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"Invalid {key} value: {value!r}")
    return int(numeric)


def _coerce_int_option(value: Any, *, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {key} value: {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {key} value: {value!r}") from None
    if not numeric.is_integer():
        raise ValueError(f"Invalid {key} value: {value!r}")
    return int(numeric)


def _coerce_grid_shape(grid_shape: Any) -> tuple[int, int, int]:
    try:
        values = tuple(
            _coerce_positive_int(value, key="grid_shape")
            for value in grid_shape
        )
    except Exception:
        raise ValueError(f"Invalid grid shape: {grid_shape!r}") from None
    if len(values) != 3:
        raise ValueError(f"Invalid grid shape: {grid_shape!r}")
    return values


def _coerce_csv_tokens(value: Any, *, key: str) -> list[str]:
    if value is None:
        return []
    tokens = [token.strip() for token in str(value).split(",")]
    result = [token for token in tokens if token]
    if not result:
        raise ValueError(f"Invalid {key} value: {value!r}")
    return result


# Derived from the alias table's values so a new backend cannot drift out of the
# input-phase validation below.
_SUPPORTED_CHARGE_BACKENDS = frozenset(_CHARGE_BACKEND_ALIASES.values())


def _normalize_charge_backend_name(name: str | None) -> str:
    candidate = "CHARGE3NET" if name is None else str(name).strip().upper()
    if not candidate:
        candidate = "CHARGE3NET"
    return _CHARGE_BACKEND_ALIASES.get(candidate, candidate)


def _charge_backend_env_vars(kind: str, backend: str | None) -> tuple[str, ...]:
    backend_name = _normalize_charge_backend_name(backend)
    return _CHARGE_BACKEND_ENV_VARS.get(backend_name, {}).get(kind, ())


def _resolve_charge_env_var(
    *,
    generic_env_var: str,
    kind: str,
    backend: str | None,
) -> str | None:
    backend_name = None if backend is None else _normalize_charge_backend_name(backend)
    if backend_name is not None:
        for env_var in _charge_backend_env_vars(kind, backend_name):
            value = os.environ.get(env_var)
            if value:
                return value
    return os.environ.get(generic_env_var)


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


# Ceiling for a single FFT-grid axis. Realistic grids stay in the low
# thousands (a 1000 A cell at ENCUT=1000 needs ~8000); 100000 per axis is
# >10x beyond that while keeping the smooth-number search below trivially
# fast. Without a bound, a finite-but-absurd ENCUT (1e30 passes every
# is-it-a-number check) pushed `minimum` to ~1e17 and the search into a
# multi-year spin: _largest_prime_factor trial-divides to sqrt(candidate),
# so each probe alone took hours and a server worker wedged permanently on
# one request. Same class as _MAX_REQUEST_TIMEOUT: values handed to
# unbounded machinery must reject huge-but-finite, not just inf/nan.
_MAX_FFT_GRID_POINTS_PER_AXIS = 100_000

# The per-axis cap alone is not enough: NGXF=NGYF=NGZF=99999 is per-axis legal
# but a 1e15-point (8 PB) grid, which passed the input phase and then died at
# CHGCAR-write time as a "retryable" MemoryError after the whole calculation
# was paid for. Bound the RESOURCE (total points), not just the parameter
# axis. 1e9 (~1000^3) is >10x beyond the largest realistic fine grid
# (~500^3) while still allocatable.
_MAX_FFT_GRID_TOTAL_POINTS = 1_000_000_000


def _next_even_smooth_number(minimum: float) -> int:
    numeric = float(minimum)
    if not np.isfinite(numeric) or numeric > _MAX_FFT_GRID_POINTS_PER_AXIS:
        raise ValueError(
            f"CHGCAR grid dimension {numeric!r} exceeds the supported maximum "
            f"of {_MAX_FFT_GRID_POINTS_PER_AXIS} points per axis; check ENCUT "
            "and the cell size."
        )
    candidate = max(2, int(np.ceil(numeric)))
    if candidate % 2:
        candidate += 1
    while _largest_prime_factor(candidate) > 7:
        candidate += 2
    return candidate


def _validated_fft_grid_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Reject grids no machine can hold, so they classify as input errors.

    An explicit NGXF=1e12 sails through _coerce_positive_int, and the failure
    then happens at CHGCAR-write time as a MemoryError -- a "retryable"
    calculation failure (exit 2) that a batch driver resubmits forever. The
    grid is pure input, so judge it at input time (exit 1).
    """

    for count in shape:
        if count > _MAX_FFT_GRID_POINTS_PER_AXIS:
            raise ValueError(
                f"CHGCAR grid {shape!r} exceeds the supported maximum of "
                f"{_MAX_FFT_GRID_POINTS_PER_AXIS} points per axis."
            )
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if total > _MAX_FFT_GRID_TOTAL_POINTS:
        raise ValueError(
            f"CHGCAR grid {shape!r} has {total} points, exceeding the "
            f"supported maximum of {_MAX_FFT_GRID_TOTAL_POINTS} total points."
        )
    return shape


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
        return _validated_fft_grid_shape(
            tuple(
                _coerce_positive_int(value, key=tag)
                for tag, value in zip(("NGXF", "NGYF", "NGZF"), explicit_fine, strict=False)
            )
        )

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
            coarse_shape = tuple(
                _coerce_positive_int(value, key=tag)
                for tag, value in zip(("NGX", "NGY", "NGZ"), explicit_coarse, strict=False)
            )
        else:
            prec = _normalize_prec(_coerce_mapping_value(incar, "PREC"))
            coarse_shape_list = list(_coarse_fft_shape_from_cell(cell, encut=encut, prec=prec))
            for index, explicit in enumerate(explicit_coarse):
                if explicit is not None:
                    coarse_shape_list[index] = _coerce_positive_int(
                        explicit,
                        key=("NGX", "NGY", "NGZ")[index],
                    )
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
            fine_shape[index] = _coerce_positive_int(
                explicit,
                key=("NGXF", "NGYF", "NGZF")[index],
            )
    return _validated_fft_grid_shape(
        (int(fine_shape[0]), int(fine_shape[1]), int(fine_shape[2]))
    )


def _charge_density_options_from_bcar(bcar_tags: Mapping[str, Any]) -> dict[str, Any]:
    root = _root()
    requested_backend = bcar_tags.get(
        "CHARGE_MLP",
        bcar_tags.get("CHARGE_BACKEND", "CHARGE3NET"),
    )
    # Validate the backend SELECTOR here, in the input phase. Every other CHARGE_*
    # value is validated in this function (so run_workdir's _read_workdir_input
    # reports a bad one as input_error / exit 1), but the selector was only
    # rejected later, at dispatch inside predict_charge_density -- after the
    # single point had already completed and written OUTCAR/CONTCAR. A typo like
    # CHARGE_MLP=CHARGE3NE therefore escaped as a plain ValueError and server mode
    # reported calculation_error (exit 2), which SERVER_MODE_SPEC 2.5 documents as
    # RETRYABLE, so a retry driver resubmits a permanently broken BCAR forever.
    # _normalize_charge_backend_name is a pure alias lookup that never raises, so
    # the check has to be explicit. The dispatch-time raise stays as defense for
    # direct API callers.
    normalized_backend = root._normalize_charge_backend_name(requested_backend)
    if normalized_backend not in root._SUPPORTED_CHARGE_BACKENDS:
        supported = ", ".join(sorted(root._SUPPORTED_CHARGE_BACKENDS))
        raise ValueError(
            f"Unsupported charge-density backend: {normalized_backend}. "
            f"Supported: {supported}"
        )
    options: dict[str, Any] = {
        "backend": requested_backend,
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
        options["max_probes_per_batch"] = _validate_max_probes_per_batch(max_batch)
    for tag_name, (option_name, value_type) in _CHARGE_MODEL_CONFIG_TAGS.items():
        raw_value = bcar_tags.get(tag_name)
        if raw_value is None:
            continue
        if value_type is bool:
            options[option_name] = root._parse_optional_bool_tag(dict(bcar_tags), tag_name)
        elif value_type is int:
            options[option_name] = _coerce_int_option(raw_value, key=tag_name)
        elif value_type is float:
            options[option_name] = _coerce_float(raw_value, key=tag_name)
        else:
            options[option_name] = str(raw_value)
    deepcdp_metadata = bcar_tags.get("CHARGE_DEEPCDP_METADATA")
    if deepcdp_metadata is not None:
        options["metadata_path"] = str(deepcdp_metadata)
    deepcdp_species = bcar_tags.get("CHARGE_DEEPCDP_SPECIES")
    if deepcdp_species is not None:
        options["charge_species"] = _coerce_csv_tokens(
            deepcdp_species,
            key="CHARGE_DEEPCDP_SPECIES",
        )
    soap_rcut = bcar_tags.get("CHARGE_DEEPCDP_RCUT")
    if soap_rcut is not None:
        options["soap_rcut"] = _coerce_float(soap_rcut, key="CHARGE_DEEPCDP_RCUT")
    soap_nmax = bcar_tags.get("CHARGE_DEEPCDP_NMAX")
    if soap_nmax is not None:
        options["soap_nmax"] = _coerce_int_option(soap_nmax, key="CHARGE_DEEPCDP_NMAX")
    soap_lmax = bcar_tags.get("CHARGE_DEEPCDP_LMAX")
    if soap_lmax is not None:
        options["soap_lmax"] = _coerce_int_option(soap_lmax, key="CHARGE_DEEPCDP_LMAX")
    soap_sigma = bcar_tags.get("CHARGE_DEEPCDP_SIGMA")
    if soap_sigma is not None:
        options["soap_sigma"] = _coerce_float(soap_sigma, key="CHARGE_DEEPCDP_SIGMA")
    if "CHARGE_DEEPCDP_PERIODIC" in bcar_tags:
        options["soap_periodic"] = root._parse_optional_bool_tag(
            dict(bcar_tags),
            "CHARGE_DEEPCDP_PERIODIC",
        )
    activation = bcar_tags.get("CHARGE_DEEPCDP_ACTIVATION")
    if activation is not None:
        options["activation"] = str(activation)
    weighting: dict[str, Any] = {}
    for tag_name, (option_name, value_type) in _DEEPCDP_WEIGHTING_KEYS.items():
        raw_value = bcar_tags.get(tag_name)
        if raw_value is None:
            continue
        if value_type is float:
            weighting[option_name] = _coerce_float(raw_value, key=tag_name)
        else:
            weighting[option_name] = str(raw_value)
    if weighting:
        options["weighting"] = weighting
    return options


def _validate_max_probes_per_batch(
    value: object,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            "Invalid CHARGE_MAX_PROBES_PER_BATCH value: "
            f"{value!r}. Expected a positive integer."
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "Invalid CHARGE_MAX_PROBES_PER_BATCH value: "
            f"{value!r}. Expected a positive integer."
        ) from None
    if not numeric.is_integer() or numeric <= 0:
        raise ValueError(
            "Invalid CHARGE_MAX_PROBES_PER_BATCH value: "
            f"{value!r}. Expected a positive integer."
        )
    return int(numeric)


def _resolve_charge_python(
    python_executable: str | None,
    backend: str | None = None,
) -> str:
    if python_executable:
        return str(Path(python_executable).expanduser())
    env_python = _resolve_charge_env_var(
        generic_env_var="VPMDK_CHARGE_PYTHON",
        kind="python",
        backend=backend,
    )
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


def _resolve_charge_source_dir(source_dir: str | None, backend: str | None = None) -> str | None:
    if source_dir:
        return str(Path(source_dir).expanduser())
    env_source_dir = _resolve_charge_env_var(
        generic_env_var="VPMDK_CHARGE_SOURCE_DIR",
        kind="source_dir",
        backend=backend,
    )
    return _resolve_charge_env_path(env_source_dir)


def _resolve_charge_model_path(
    model_path: str | None,
    source_dir: str | None,
    backend: str | None = None,
) -> str | None:
    if model_path:
        return str(Path(model_path).expanduser())
    env_model = _resolve_charge_env_var(
        generic_env_var="VPMDK_CHARGE_MODEL",
        kind="model",
        backend=backend,
    )
    if env_model:
        return _resolve_charge_env_path(env_model)
    if source_dir and _normalize_charge_backend_name(backend) == "CHARGE3NET":
        default_model = Path(source_dir) / "models" / "charge3net_mp.pt"
        if default_model.exists():
            return str(default_model)
    return None


def _resolve_charge_metadata_path(
    metadata_path: str | None,
    model_path: str | None,
    backend: str | None = None,
) -> str | None:
    if metadata_path:
        return str(Path(metadata_path).expanduser())

    backend_name = _normalize_charge_backend_name(backend)
    if backend_name != "DEEPCDP":
        return None

    if model_path:
        model_location = Path(model_path).expanduser()
        candidate_dir = model_location if model_location.is_dir() else model_location.parent
        for filename in ("deepcdp_config.json", "metadata.json", "config.json"):
            candidate = candidate_dir / filename
            if candidate.exists():
                return str(candidate)
    env_value = os.environ.get("VPMDK_DEEPCDP_METADATA")
    if env_value:
        return _resolve_charge_env_path(env_value)
    return None


def _resolve_deepdft_model_dir(model_path: str | None, source_dir: str | None) -> str | None:
    resolved = _resolve_charge_model_path(model_path, source_dir, backend="DEEPDFT")
    if not resolved:
        return None
    model_dir = Path(resolved).expanduser()
    if model_dir.is_file() or model_dir.name in {"best_model.pth", "arguments.json"}:
        model_dir = model_dir.parent
    return str(model_dir)


def _resolve_deepcdp_checkpoint_path(model_path: str | None, source_dir: str | None) -> str | None:
    resolved = _resolve_charge_model_path(model_path, source_dir, backend="DEEPCDP")
    if not resolved:
        return None
    checkpoint_path = Path(resolved).expanduser()
    if checkpoint_path.is_file():
        return str(checkpoint_path)
    explicit_candidate = checkpoint_path / "model.pt"
    if explicit_candidate.exists():
        return str(explicit_candidate)
    pt_files = sorted(checkpoint_path.glob("*.pt"))
    if len(pt_files) == 1:
        return str(pt_files[0])
    if not pt_files:
        # Pure path-resolution failures: permanent configuration, decidable
        # without executing anything -> input_error (exit 1), never the
        # RETRYABLE exit 2 (WorkdirInputError subclasses RuntimeError, so
        # callers catching RuntimeError keep working).
        raise _root().WorkdirInputError(
            "DeepCDP model path must point to a .pt checkpoint or a directory containing one."
        )
    raise _root().WorkdirInputError(
        "DeepCDP model directory contains multiple .pt checkpoints. Set CHARGE_MODEL "
        "to the exact checkpoint path."
    )


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
    source_dir = _resolve_charge_source_dir(source_dir, backend="CHARGE3NET")
    model_path = _resolve_charge_model_path(model_path, source_dir, backend="CHARGE3NET")
    max_probes_per_batch = _validate_max_probes_per_batch(max_probes_per_batch)
    if num_interactions is not None:
        num_interactions = _coerce_int_option(num_interactions, key="num_interactions")
    if mul is not None:
        mul = _coerce_int_option(mul, key="mul")
    if lmax is not None:
        lmax = _coerce_int_option(lmax, key="lmax")
    if num_basis is not None:
        num_basis = _coerce_int_option(num_basis, key="num_basis")
    if not model_path:
        # A permanently unconfigured charge backend (no CHARGE_MODEL anywhere)
        # is decidable without executing anything, exactly like the backend
        # SELECTOR typo fixed in _charge_density_options_from_bcar: as a plain
        # RuntimeError, server mode reported calculation_error (exit 2), which
        # SERVER_MODE_SPEC 2.5 documents as RETRYABLE, so a retry driver
        # re-ran the FULL calculation (the ionic loop completes before this
        # raise) on the same broken directory forever, while one-shot exited 1
        # for the byte-identical input. WorkdirInputError classifies both
        # paths as input_error / exit 1. Raising at the use-site (not by
        # pre-resolving paths in the input phase) keeps monkeypatched
        # predict_charge_density flows untouched -- the R124[3] blocker.
        raise _root().WorkdirInputError(
            "ChargE3Net model checkpoint not found. Set CHARGE_MODEL (or "
            "VPMDK_CHARGE_MODEL). When CHARGE_SOURCE_DIR is set, VPMDK also checks "
            "<CHARGE_SOURCE_DIR>/models/charge3net_mp.pt."
        )

    runner_path = Path(__file__).with_name("charge3net_runner.py")
    python_path = _resolve_charge_python(python_executable, backend="CHARGE3NET")

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
            command.extend(["--num-interactions", str(num_interactions)])
        if num_neighbors is not None:
            command.extend(["--num-neighbors", str(float(num_neighbors))])
        if mul is not None:
            command.extend(["--mul", str(mul)])
        if lmax is not None:
            command.extend(["--lmax", str(lmax)])
        if basis is not None:
            command.extend(["--basis", str(basis)])
        if num_basis is not None:
            command.extend(["--num-basis", str(num_basis)])
        if spin is not None:
            command.extend(["--spin", "1" if spin else "0"])
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            # The configured interpreter itself is missing (CHARGE_PYTHON /
            # VPMDK_CHARGE_PYTHON names a non-existent executable) -- broken
            # configuration that no retry can heal, so it must classify as
            # input_error (exit 1), not the RETRYABLE calculation_error
            # (exit 2), for the same reason as the missing-checkpoint raise
            # above.
            raise _root().WorkdirInputError(
                f"Charge-density python interpreter not found: {command[0]}. "
                "Set CHARGE_PYTHON (or VPMDK_CHARGE_PYTHON) to an existing "
                "Python executable."
            ) from exc
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


def _run_deepdft_backend(
    atoms,
    *,
    grid_shape: tuple[int, int, int],
    model_path: str | None = None,
    device: str | None = None,
    source_dir: str | None = None,
    python_executable: str | None = None,
    max_probes_per_batch: int = _DEFAULT_MAX_PROBES_PER_BATCH,
) -> tuple[np.ndarray, np.ndarray | None]:
    source_dir = _resolve_charge_source_dir(source_dir, backend="DEEPDFT")
    model_dir = _resolve_deepdft_model_dir(model_path, source_dir)
    probe_count = _validate_max_probes_per_batch(max_probes_per_batch)
    if not model_dir:
        # Same classification as the ChargE3Net missing-checkpoint raise:
        # permanent configuration error -> input_error, never exit 2.
        raise _root().WorkdirInputError(
            "DeepDFT model directory not found. Set CHARGE_MODEL (or "
            "VPMDK_CHARGE_MODEL / VPMDK_DEEPDFT_MODEL) to a directory containing "
            "arguments.json and best_model.pth."
        )

    runner_path = Path(__file__).with_name("deepdft_runner.py")
    python_path = _resolve_charge_python(python_executable, backend="DEEPDFT")

    with tempfile.TemporaryDirectory(prefix="vpmdk_deepdft_") as tmp_dir:
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
            "--model-dir",
            str(model_dir),
            "--probe-count",
            str(int(probe_count)),
        ]
        if source_dir:
            command.extend(["--source-dir", str(source_dir)])
        if device is not None:
            command.extend(["--device", str(_root()._resolve_device(device))])
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            # The configured interpreter itself is missing (CHARGE_PYTHON /
            # VPMDK_CHARGE_PYTHON names a non-existent executable) -- broken
            # configuration that no retry can heal, so it must classify as
            # input_error (exit 1), not the RETRYABLE calculation_error
            # (exit 2), for the same reason as the missing-checkpoint raise
            # above.
            raise _root().WorkdirInputError(
                f"Charge-density python interpreter not found: {command[0]}. "
                "Set CHARGE_PYTHON (or VPMDK_CHARGE_PYTHON) to an existing "
                "Python executable."
            ) from exc
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or "no output"
            raise RuntimeError(f"DeepDFT prediction failed: {details}")
        with np.load(output_path) as payload:
            density = np.asarray(payload["density"])
        return density, None


def _run_deepcdp_backend(
    atoms,
    *,
    grid_shape: tuple[int, int, int],
    model_path: str | None = None,
    device: str | None = None,
    source_dir: str | None = None,
    python_executable: str | None = None,
    max_probes_per_batch: int = _DEFAULT_MAX_PROBES_PER_BATCH,
    metadata_path: str | None = None,
    charge_species: list[str] | tuple[str, ...] | str | None = None,
    soap_rcut: float | None = None,
    soap_nmax: int | None = None,
    soap_lmax: int | None = None,
    soap_sigma: float | None = None,
    soap_periodic: bool | None = None,
    activation: str | None = None,
    weighting: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    source_dir = _resolve_charge_source_dir(source_dir, backend="DEEPCDP")
    checkpoint_path = _resolve_deepcdp_checkpoint_path(model_path, source_dir)
    resolved_metadata_path = _resolve_charge_metadata_path(
        metadata_path,
        checkpoint_path,
        backend="DEEPCDP",
    )
    probe_count = _validate_max_probes_per_batch(max_probes_per_batch)
    if not checkpoint_path:
        # Same classification as the ChargE3Net missing-checkpoint raise:
        # permanent configuration error -> input_error, never exit 2.
        raise _root().WorkdirInputError(
            "DeepCDP checkpoint not found. Set CHARGE_MODEL (or VPMDK_CHARGE_MODEL / "
            "VPMDK_DEEPCDP_MODEL) to a DeepCDP .pt file or directory."
        )

    runner_path = Path(__file__).with_name("deepcdp_runner.py")
    python_path = _resolve_charge_python(python_executable, backend="DEEPCDP")

    with tempfile.TemporaryDirectory(prefix="vpmdk_deepcdp_") as tmp_dir:
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
            str(checkpoint_path),
            "--probe-count",
            str(int(probe_count)),
        ]
        if source_dir:
            command.extend(["--source-dir", str(source_dir)])
        if device is not None:
            command.extend(["--device", str(_root()._resolve_device(device))])
        if resolved_metadata_path:
            command.extend(["--metadata-path", str(resolved_metadata_path)])
        if charge_species is not None:
            if isinstance(charge_species, str):
                species_tokens = _coerce_csv_tokens(charge_species, key="charge_species")
            else:
                species_tokens = [str(token).strip() for token in charge_species if str(token).strip()]
            command.extend(["--species", ",".join(species_tokens)])
        if soap_rcut is not None:
            command.extend(["--soap-rcut", str(float(soap_rcut))])
        if soap_nmax is not None:
            command.extend(["--soap-nmax", str(_coerce_int_option(soap_nmax, key="soap_nmax"))])
        if soap_lmax is not None:
            command.extend(["--soap-lmax", str(_coerce_int_option(soap_lmax, key="soap_lmax"))])
        if soap_sigma is not None:
            command.extend(["--soap-sigma", str(float(soap_sigma))])
        if soap_periodic is not None:
            command.extend(["--soap-periodic", "1" if soap_periodic else "0"])
        if activation is not None:
            command.extend(["--activation", str(activation)])
        if weighting:
            for key, value in weighting.items():
                command.extend([f"--weighting-{key.replace('_', '-')}", str(value)])
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            # The configured interpreter itself is missing (CHARGE_PYTHON /
            # VPMDK_CHARGE_PYTHON names a non-existent executable) -- broken
            # configuration that no retry can heal, so it must classify as
            # input_error (exit 1), not the RETRYABLE calculation_error
            # (exit 2), for the same reason as the missing-checkpoint raise
            # above.
            raise _root().WorkdirInputError(
                f"Charge-density python interpreter not found: {command[0]}. "
                "Set CHARGE_PYTHON (or VPMDK_CHARGE_PYTHON) to an existing "
                "Python executable."
            ) from exc
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or "no output"
            raise RuntimeError(f"DeepCDP prediction failed: {details}")
        with np.load(output_path) as payload:
            density = np.asarray(payload["density"])
        return density, None


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
    metadata_path: str | None = None,
    charge_species: list[str] | tuple[str, ...] | str | None = None,
    soap_rcut: float | None = None,
    soap_nmax: int | None = None,
    soap_lmax: int | None = None,
    soap_sigma: float | None = None,
    soap_periodic: bool | None = None,
    activation: str | None = None,
    weighting: Mapping[str, Any] | None = None,
) -> ChargeDensityResult:
    """Predict charge density on a user-specified or INCAR-derived grid."""

    if grid_shape is None:
        if incar is None:
            raise ValueError("grid_shape or incar must be provided for charge-density prediction.")
        grid_shape = determine_vasp_fft_grid(reference if reference is not None else atoms, incar)
    grid_shape = _coerce_grid_shape(grid_shape)

    backend_name = _normalize_charge_backend_name(backend)
    max_probes_per_batch = _validate_max_probes_per_batch(max_probes_per_batch)
    if backend_name == "CHARGE3NET":
        resolved_source_dir = _resolve_charge_source_dir(source_dir, backend=backend_name)
        resolved_model_path = _resolve_charge_model_path(
            model_path,
            resolved_source_dir,
            backend=backend_name,
        )
        density, spin_density = _run_charge3net_backend(
            atoms,
            grid_shape=grid_shape,
            model_path=resolved_model_path,
            device=device,
            source_dir=resolved_source_dir,
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
        model_config = {
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
        }
    elif backend_name == "DEEPDFT":
        resolved_source_dir = _resolve_charge_source_dir(source_dir, backend=backend_name)
        resolved_model_path = _resolve_deepdft_model_dir(model_path, resolved_source_dir)
        density, spin_density = _run_deepdft_backend(
            atoms,
            grid_shape=grid_shape,
            model_path=resolved_model_path,
            device=device,
            source_dir=resolved_source_dir,
            python_executable=python_executable,
            max_probes_per_batch=max_probes_per_batch,
        )
        model_config = {
            "probe_count": max_probes_per_batch,
            "spin_output": False,
        }
    elif backend_name == "DEEPCDP":
        resolved_source_dir = _resolve_charge_source_dir(source_dir, backend=backend_name)
        resolved_model_path = _resolve_deepcdp_checkpoint_path(model_path, resolved_source_dir)
        resolved_metadata_path = _resolve_charge_metadata_path(
            metadata_path,
            resolved_model_path,
            backend=backend_name,
        )
        density, spin_density = _run_deepcdp_backend(
            atoms,
            grid_shape=grid_shape,
            model_path=resolved_model_path,
            device=device,
            source_dir=resolved_source_dir,
            python_executable=python_executable,
            max_probes_per_batch=max_probes_per_batch,
            metadata_path=resolved_metadata_path,
            charge_species=charge_species,
            soap_rcut=soap_rcut,
            soap_nmax=soap_nmax,
            soap_lmax=soap_lmax,
            soap_sigma=soap_sigma,
            soap_periodic=soap_periodic,
            activation=activation,
            weighting=weighting,
        )
        model_config = {
            key: value
            for key, value in {
                "probe_count": max_probes_per_batch,
                "metadata_path": resolved_metadata_path,
                "charge_species": charge_species,
                "soap_rcut": soap_rcut,
                "soap_nmax": soap_nmax,
                "soap_lmax": soap_lmax,
                "soap_sigma": soap_sigma,
                "soap_periodic": soap_periodic,
                "activation": activation,
                "weighting": dict(weighting) if weighting else None,
                "spin_output": False,
            }.items()
            if value is not None
        }
    else:
        raise ValueError(f"Unsupported charge-density backend: {backend_name}")

    return ChargeDensityResult(
        atoms=atoms,
        density=density,
        grid_shape=grid_shape,
        backend=backend_name,
        spin_density=spin_density,
        metadata={
            "model_path": resolved_model_path,
            "device": "auto" if device is None else _root()._resolve_device(device),
            "source_dir": resolved_source_dir,
            "model_config": model_config,
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
    _rewrite_chgcar_header_precision(path, atoms)


def _rewrite_chgcar_header_precision(path: str | os.PathLike[str], atoms) -> None:
    """Write a VASP-equivalent header that Henkelman bader reads reliably."""

    cell = np.asarray(atoms.cell.array, dtype=float)
    if cell.shape != (3, 3):
        return
    if not np.all(np.isfinite(cell)):
        return

    lines = Path(path).read_text().splitlines()
    if len(lines) < 5:
        return

    lines[1] = "   1.0000000000000000"
    for index, vector in enumerate(cell, start=2):
        lines[index] = "".join(f"{component:13.8f}" for component in vector)
    count_line_index = 5
    if len(lines) >= 7:
        symbol_fields = lines[5].split()
        count_fields = lines[6].split()
        symbols_are_counts = all(_is_integer_token(field) for field in symbol_fields)
        next_line_is_counts = all(_is_integer_token(field) for field in count_fields)
        if symbol_fields and count_fields and not symbols_are_counts and next_line_is_counts:
            count_line_index = 6
    _rewrite_chgcar_coordinate_precision(lines, count_line_index)
    Path(path).write_text("\n".join(lines) + "\n")


def _is_integer_token(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _rewrite_chgcar_coordinate_precision(lines: list[str], count_line_index: int) -> None:
    if count_line_index >= len(lines):
        return
    try:
        atom_count = sum(int(value) for value in lines[count_line_index].split())
    except ValueError:
        return

    coordinate_mode_index = count_line_index + 1
    if coordinate_mode_index >= len(lines):
        return
    if lines[coordinate_mode_index].strip().lower().startswith("s"):
        coordinate_mode_index += 1
    atom_start = coordinate_mode_index + 1
    for index in range(atom_start, min(atom_start + atom_count, len(lines))):
        fields = lines[index].split()
        if len(fields) < 3:
            continue
        try:
            coordinates = [float(value) for value in fields[:3]]
        except ValueError:
            continue
        suffix = fields[3:]
        prefix = "".join(f"{value:12.6f}" for value in coordinates)
        lines[index] = f"{prefix}{(' ' + ' '.join(suffix)) if suffix else ''}"
